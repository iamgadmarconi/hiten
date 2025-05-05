import symengine as se
import numpy as np
import math
import time
import logging

from .core import Polynomial, Monomial, _poisson_bracket, _poisson_bracket_term_by_term, _dot_product, _monomial_from_key
from ..variables import physical_vars, real_normal_vars, canonical_normal_vars, linear_modes_vars, get_vars
from .factory import (
    hamiltonian, 
    physical_to_real_normal, 
    real_normal_to_complex_canonical, 
    complex_canonical_to_real_normal
)
from system.libration import LibrationPoint

from log_config import logger


x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)


def real_normal_center_manifold(point: LibrationPoint, symbolic: bool = False, max_degree: int = None, 
                               poisson_method: str = 'auto', use_cache: bool = True) -> tuple[Polynomial, Polynomial]:
    """
    Compute the center manifold of a libration point up to a given degree in real normal form.

    Parameters
    ----------
    point : LibrationPoint
        The libration point to compute the center manifold for.
    symbolic : bool
        Whether to use symbolic expressions or numeric values.
    max_degree : int
        The maximum degree of the center manifold.
    poisson_method : str, optional
        Method to use for Poisson bracket calculations:
        - 'auto': Automatically select the best method
        - 'standard': Use the standard implementation
        - 'term_by_term': Use term-by-term differentiation
    use_cache : bool, optional
        Whether to cache Poisson bracket results for reuse

    Returns
    -------
    Polynomial
        The center manifold Hamiltonian in real normal form.
    """
    H_cnr = reduce_center_manifold(point, symbolic, max_degree, poisson_method, use_cache)
    H_rnr = complex_canonical_to_real_normal(point, H_cnr, symbolic, max_degree)
    return H_rnr

def reduce_center_manifold(point: LibrationPoint, symbolic: bool = False, max_degree: int = None,
                          poisson_method: str = 'auto', use_cache: bool = True) -> Polynomial:
    """
    Reduce the transformed center manifold Hamiltonian in complex normal 
    form to a given degree.

    Parameters
    ----------
    point : LibrationPoint
        The libration point to compute the center manifold for.
    symbolic : bool
        Whether to use symbolic expressions or numeric values.
    max_degree : int
        The maximum degree of the center manifold.
    poisson_method : str, optional
        Method to use for Poisson bracket calculations
    use_cache : bool, optional
        Whether to cache Poisson bracket results for reuse

    Returns
    -------
    Polynomial
        The reduced center manifold Hamiltonian in reduced complex normal form.
    """
    H_cnt = compute_center_manifold(point, symbolic, max_degree, poisson_method, use_cache)[0]
    H_cnr = H_cnt.subs({q1: 0, p1: 0}).truncate(max_degree)
    return H_cnr

def compute_center_manifold(point: LibrationPoint, symbolic: bool = False, max_degree: int = None,
                           poisson_method: str = 'auto', use_cache: bool = True) -> tuple[Polynomial, Polynomial]:
    """
    Compute the transformed center manifold Hamiltonian and the generating 
    function in complex normal form up to a given degree.

    Parameters
    ----------
    point : LibrationPoint
        The libration point to compute the center manifold for.
    symbolic : bool
        Whether to use symbolic expressions or numeric values.
    max_degree : int
        The maximum degree of the center manifold.
    poisson_method : str, optional
        Method to use for Poisson bracket calculations
    use_cache : bool, optional
        Whether to cache Poisson bracket results for reuse

    Returns
    -------
    tuple[Polynomial, Polynomial]
        The transformed center manifold Hamiltonian in complex normal form
        and the generating function.
    """
    H_phys = hamiltonian(point, max_degree)
    H_rn   = physical_to_real_normal(point, H_phys, symbolic, max_degree)
    H_cn   = real_normal_to_complex_canonical(point, H_rn, symbolic, max_degree)
    H_cnt, G_total = _lie_transform(point, H_cn, max_degree, poisson_method, use_cache)
    return H_cnt, G_total

def _lie_transform(point: LibrationPoint, H_init: Polynomial, max_degree: int,
                  poisson_method: str = 'auto', use_cache: bool = True) -> tuple[Polynomial, Polynomial]:
    """
    Bring *H_init* to the Jorba-Masdemont partial normal form up to
    *max_degree* using a Lie-series expansion.

    Parameters
    ----------
    point : LibrationPoint
        The LibrationPoint object (needed for linear modes)
    H_init : Polynomial
        Hamiltonian already expressed in complex canonical variables
        (q1,q2,q3,p1,p2,p3), with a working ``build_by_degree`` helper.
    max_degree : int
        Highest total degree to normalise and keep.
    poisson_method : str, optional
        Method to use for Poisson bracket calculations:
        - 'auto': Automatically select the best method
        - 'standard': Use the standard implementation
        - 'term_by_term': Use term-by-term differentiation
    use_cache : bool, optional
        Whether to cache Poisson bracket results for reuse

    Returns
    -------
    tuple[Polynomial, Polynomial]
        (H_normalized, G_total) where H_normalized is the transformed
        Hamiltonian and G_total is the accumulated generating function.
    """
    # Clear all caches at the beginning for a fresh start
    Polynomial.clear_all_caches()
    
    H_transformed = H_init.copy()
    G_total = Polynomial(H_init.variables, se.Integer(0))
    lambda1, omega1, omega2 = point.linear_modes()

    eta_vector = [se.sympify(lambda1), se.I*se.sympify(omega1), se.I*se.sympify(omega2)]

    # H_2 is the homogeneous part of H_init of degree 2
    H_2 = _get_homogeneous_terms(H_init, 2)

    for n in range(3, max_degree + 1):
        logger.info(f"Normalizing order {n}...") # Progress indicator

        # Step 6: Extract homogeneous terms of degree n
        H_n_current = _get_homogeneous_terms(H_transformed, n)
        logger.debug(f"--- n = {n} ---")
        logger.debug(f"H_{n}_current:\n{H_n_current.expression}\n")
        if H_n_current.total_degree() == 0: # If H_n is zero, nothing to eliminate
            logger.warning(f"Order {n} is zero, skipping normalization for this order.")
            continue

        # Step 7: Select monomials for elimination
        H_n_to_eliminate = _select_monomials_for_elimination(H_n_current)
        logger.debug(f"H_{n}_to_eliminate:\n{H_n_to_eliminate.expression}\n")
        # If there are no terms to eliminate at this order, skip
        if H_n_to_eliminate.total_degree() == 0:
            logger.warning(f"No terms to eliminate at order {n}.")
            continue

        # Step 8: Solve homological equation to find G_n
        G_n = _solve_homological_equation(H_n_to_eliminate, H_2, eta_vector, H_init.variables)
        logger.debug(f"G_{n}:\n{G_n.expression}\n")
        
        # Clear caches before applying Lie transform to avoid memory overflow
        # and limit cache size during complex operations
        Polynomial.clear_all_caches()
        
        # Step 9: Apply Lie series transform generated by G_n to H_transformed
        H_transformed = _apply_lie_transform(H_transformed, G_n, max_degree, poisson_method, use_cache)

        # Step 10: Add G_n to total generating function
        G_total = G_total + G_n
        
        # Limit cache size after each order to prevent memory issues
        Polynomial.limit_cache_size()

    # Final cleanup of caches
    Polynomial.clear_all_caches()
    
    # After the loop, H_transformed is the normalized Hamiltonian up to max_degree
    # and G_total is the sum of generators G_3 + ... + G_n
    return H_transformed, G_total

def _get_homogeneous_terms(H: Polynomial, n: int) -> Polynomial:
    """
    Extracts the homogeneous polynomial of degree n from Hamiltonian H.

    Parameters
    ----------
    H : Polynomial
        The input polynomial.
    n : int
        The desired degree.

    Returns
    -------
    Polynomial
        The homogeneous part of H of degree n.
    """
    # build_by_degree returns a dict {degree: [monomials]}
    monomials_by_degree = H.build_by_degree()

    # Get the list of monomials for the specified degree n
    # If degree n is not present, it will return an empty list from defaultdict
    homogeneous_monomials = monomials_by_degree.get(n, [])

    # Construct a new Polynomial from these monomials
    # Pass the variables from the original polynomial
    return Polynomial.from_monomials(H.variables, homogeneous_monomials)

def _select_monomials_for_elimination(H_n_current: Polynomial) -> Polynomial:
    """
    Selects monomials from H_n_current for elimination based on the partial
    normalization condition (kq[0] != kp[0]).

    Parameters
    ----------
    H_n_current : Polynomial
        The homogeneous polynomial of degree n.

    Returns
    -------
    Polynomial
        A polynomial containing only the terms from H_n_current to be eliminated.
    """
    # Use the more efficient monomial extraction through get_monomials()
    monomials_to_eliminate = []
    
    # Get all monomials at once rather than iterating term by term
    all_monomials = H_n_current.get_monomials()
    
    # Filter the monomials based on our criterion
    for monomial in all_monomials:
        # ONLY keep this condition based on Section 3.1 for partial uncoupling
        if monomial.kq[0] != monomial.kp[0]:
            monomials_to_eliminate.append(monomial)

    # Construct a new Polynomial from the selected monomials
    return Polynomial.from_monomials(H_n_current.variables, monomials_to_eliminate)

def _solve_homological_equation(H_n_to_eliminate: Polynomial, H_2: Polynomial, eta_vector: list[se.Basic], variables: list[se.Symbol]) -> Polynomial:
    """
    Solves {H_2, G_n} = -H_n_to_eliminate for G_n.

    Parameters
    ----------
    H_n_to_eliminate : Polynomial
        The terms of degree n to be eliminated.
    H_2 : Polynomial
        The second-order part of the Hamiltonian (diagonal form).
    eta_vector : list[se.Basic]
        The vector of eigenvalues [eta_1, eta_2, eta_3] for the dot product denominator.
    variables : list[se.Symbol]
        The list of variables (q1, q2, q3, p1, p2, p3) needed to reconstruct monomials.

    Returns
    -------
    Polynomial
        The generating function G_n of degree n.

    Notes
    - H_2 is assumed to be in the form eta_1 q1 p1 + eta_2 q2 p2 + eta_3 q3 p3
    - The dot product <k_p - k_q, eta> uses eta = (eta_1, eta_2, eta_3)
    - k_q, k_p are tuples of exponents for (q1, q2, q3) and (p1, p2, p3).

    """
    g_n_monomials = []

    for h_coeff, h_monomial in H_n_to_eliminate.iter_terms():
        kq = h_monomial.kq
        kp = h_monomial.kp

        # Compute the vector k_p - k_q
        kp_minus_kq = tuple(kp[i] - kq[i] for i in range(len(kq))) # Assuming kq and kp have same length (3)

        # Compute the denominator: <k_p - k_q, eta>
        denominator = _dot_product(kp_minus_kq, eta_vector)

        if denominator == 0:
            # Check for small divisors / zero denominator
            # The paper states this should not be zero for the selected terms.
            err = f"Zero denominator encountered for term with exponents kq={kq}, kp={kp}. This term cannot be eliminated."
            logger.error(err)
            raise ValueError(err)

        # Compute the coefficient for G_n
        g_coeff = -h_coeff / denominator

        # Create the Monomial for G_n
        # Need to reconstruct the symbolic expression for the monomial
        g_sym_expr = g_coeff * _monomial_from_key(kq, kp, [q1,q2,q3], [p1,p2,p3])

        g_n_monomials.append(Monomial(coeff=g_coeff, kq=kq, kp=kp, sym=g_sym_expr))

    # Construct the Polynomial for G_n
    return Polynomial.from_monomials(variables, g_n_monomials)

def _apply_lie_transform(H_current: Polynomial, G_n: Polynomial, N_max: int, 
                        poisson_method: str = 'auto', use_cache: bool = True) -> Polynomial:
    """
    Applies the Lie series transform generated by G_n to H_current up to order N_max.
    Implements equation (16) from Jorba-Masdemont paper:
        H_hat ≡ H + {H, G} + (1/2!) {{H, G}, G} + (1/3!) {{{H, G}, G}, G} + ...

    Parameters
    ----------
    H_current : Polynomial
        The Hamiltonian before applying the transform.
    G_n : Polynomial
        The generating function for this order.
    N_max : int
        The maximum total degree to keep in the transformed Hamiltonian.
    poisson_method : str, optional
        Method to use for Poisson bracket calculations:
        - 'auto': Automatically select the best method
        - 'standard': Use the standard implementation
        - 'term_by_term': Use term-by-term differentiation
    use_cache : bool, optional
        Whether to cache Poisson bracket results for reuse

    Returns
    -------
    Polynomial
        The transformed Hamiltonian, truncated at N_max.
    """
    # Start with the initial Hamiltonian
    H_old = H_current.copy()
    H_new = H_old.copy()
    PB_term = H_old.copy() # Start with H_old
    G_degree = G_n.total_degree()

    # Dynamic MAX_LIE_ITERATIONS based on the order of G_n
    # From observation: terms of order n typically converge after n-2 iterations
    # Minimum of 1 iteration to ensure at least one transform is applied
    MAX_LIE_ITERATIONS = max(1, G_degree - 1)
    
    logger.debug(f"Applying transform for G of degree {G_degree}")
    logger.info(f"MAX_LIE_ITERATIONS = {MAX_LIE_ITERATIONS}")
    
    # Clear memoization cache at the beginning of each transform application
    if use_cache:
        Polynomial.memoized_poisson.cache_clear()
        
    # Pre-expand polynomials for better performance in repeated operations
    H_old_expanded = H_old.expansion
    G_n_expanded = G_n.expansion
    PB_term = H_old_expanded.copy()

    # Pre-compute and store factorial values to avoid repeated calculations
    factorials = [math.factorial(i) for i in range(MAX_LIE_ITERATIONS + 1)]

    # Process and reuse gradient calculations if needed
    if poisson_method == 'standard':
        # Pre-compute gradients to avoid recalculation in each iteration
        G_n_expanded._gradient()

    for k in range(1, MAX_LIE_ITERATIONS + 1):
        logger.info(f"Applying transform for G of degree {G_degree} at iteration {k}...")
        
        # Periodically check and limit cache sizes to prevent memory overflow
        if k % 2 == 0:
            Polynomial.limit_cache_size()
        
        # Calculate the next Poisson bracket term {PB_term, G} using optimized method
        if use_cache:
            # Use memoized version for repeated calculations
            PB_term = PB_term.optimized_poisson(G_n_expanded, method=poisson_method, use_cache=True)
        else:
            # Use specified method without caching
            PB_term = PB_term.optimized_poisson(G_n_expanded, method=poisson_method, use_cache=False)

        is_zero = (PB_term.total_degree() == 0 and PB_term.expression == 0) # Basic check, might need tolerance
        logger.debug(f"is_zero: {is_zero}, total_degree: {PB_term.total_degree()}")
        if is_zero:
            logger.info(f"Stopping Lie series at k={k} because PB_term is zero.")
            break

        term_degree = PB_term.total_degree()
        logger.info(f"Lie series k={k}: PB_term max degree = {term_degree}")

        # Use pre-computed factorial value
        factorial_k = factorials[k]
        term_coefficient = se.Integer(1) / se.Integer(factorial_k)

        # Pre-expand the term before adding to improve expansion efficiency
        term_to_add = (PB_term * term_coefficient).expansion
        
        # Check if the term has degree beyond N_max and truncate early if needed
        if term_to_add.total_degree() > N_max:
            term_to_add = term_to_add.truncate(N_max)

        H_new = H_new + term_to_add

        # --- Check if max iterations reached ---
        if k == MAX_LIE_ITERATIONS:
            logger.warning(f"Reached MAX_LIE_ITERATIONS = {k}. Stopping series.")
            break

    # Final truncation to ensure we stay within N_max
    H_final = H_new.truncate(N_max)

    logger.debug(f"Final H after transform (truncated to {N_max}):\n{H_final.expression}\n")

    return H_final

def extract_coeffs_up_to_degree(H_cm: Polynomial, max_deg: int) -> list[tuple[int, int, int, int, se.Basic]]:
    table = []

    degree_monomials = H_cm.build_by_degree()

    for degree, monomials in degree_monomials.items():
        for monomial in monomials:
            coef = monomial.coeff
            kq = monomial.kq
            kp = monomial.kp
            total_deg = sum(kq[1:]+kp[1:])  # skipping index 0 (q1,p1)
            if 1 <= total_deg <= max_deg:
                table.append((kq[1], kp[1], kq[2], kp[2], coef))
    # Don't sort the table as it may contain complex numbers that can't be compared
    return table


def coefficients_to_table(coeffs: list[tuple[int, int, int, int, se.Basic]], save=False, filename=__file__) -> str:
    """
    Format the coefficients of the center manifold Hamiltonian as a nice table.
    
    Parameters
    ----------
    coeffs : list[tuple[int, int, int, int, se.Basic]]
        The coefficients to format.
    save : bool
        Whether to save the table to a file.
    filename : str
        The filename to save the table to.
        
    Returns
    -------
    str
        A formatted table string showing the coefficients.
    """    
    # Sort by degree and then by exponents for nicer display
    def sort_key(entry):
        k1, k2, k3, k4, _ = entry
        total_deg = k1 + k2 + k3 + k4
        return (total_deg, k1, k2, k3, k4)
    
    sorted_coeffs = sorted(coeffs, key=sort_key)
    
    # Create the header
    header = "Coefficients of the transformed Hamiltonian restricted to the center manifold\n"
    header += "The exponents (k1, k2, k3, k4) refer to the variables (q2, p2, q3, p3), in this order\n"
    header += "=" * 120 + "\n"
    
    # Column headers
    col_header = "{:<5} {:<5} {:<5} {:<5} {:<30}    {:<5} {:<5} {:<5} {:<5} {:<30}\n".format(
        "k1", "k2", "k3", "k4", "hk", 
        "k1", "k2", "k3", "k4", "hk"
    )
    header += col_header
    
    # Prepare data rows
    rows = []
    
    # Split into two columns
    mid_point = (len(sorted_coeffs) + 1) // 2
    left_col = sorted_coeffs[:mid_point]
    right_col = sorted_coeffs[mid_point:]
    
    # Helper function to format coefficient values safely
    def format_coef(coef):
        try:
            # Try to evaluate to float and use scientific notation
            return "{:<30e}".format(float(coef))
        except (TypeError, ValueError, RuntimeError):
            # For complex numbers or other types that can't be converted to float
            # Just convert to string and format as is
            return "{:<30}".format(str(coef))
    
    # Fill rows
    for i in range(max(len(left_col), len(right_col))):
        row = ""
        
        # Left column
        if i < len(left_col):
            k1, k2, k3, k4, coef = left_col[i]
            row += "{:<5} {:<5} {:<5} {:<5} {}".format(k1, k2, k3, k4, format_coef(coef))
        else:
            row += " " * 50
            
        row += "    "  # Separator
        
        # Right column
        if i < len(right_col):
            k1, k2, k3, k4, coef = right_col[i]
            row += "{:<5} {:<5} {:<5} {:<5} {}".format(k1, k2, k3, k4, format_coef(coef))
        
        rows.append(row)
    
    # Combine everything
    table = header + "\n".join(rows)

    if save:
        with open(filename, 'w') as f:
            f.write(table)
    
        logger.info(f"Table saved to {filename}")

    return table