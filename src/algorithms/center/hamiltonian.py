import symengine as se
import numpy as np
import sympy as sp  # Add sympy import for numerical cleanup

from .core import Polynomial, _clean_numerical_artifacts
from system.libration import LibrationPoint
from log_config import logger

from ..variables import physical_vars, real_normal_vars, canonical_normal_vars, linear_modes_vars, scale_factors_vars, get_vars, create_symbolic_cn
from ..polynomial.base import symengine_to_custom_poly

x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)
s1, s2 = get_vars(scale_factors_vars)


def _build_T_polynomials(N: int) -> list[se.Basic]:
    """Return [T0 … TN] using the Legendre recurrence (paper eq. 6)."""
    if N == 0:
        return [se.Integer(1)]
    if N == 1:
        return [se.Integer(1), x]
        
    T = [se.Integer(1), x]
    for n in range(2, N + 1):
        n_ = se.Integer(n)
        Tn = ( (2*n_-1)/n_ ) * x * T[n-1] - ( (n_-1)/n_ ) * (x**2 + y**2 + z**2) * T[n-2]
        T.append(se.expand(Tn))
    return T


def hamiltonian(point: LibrationPoint, max_degree: int = 6) -> Polynomial:
    """
    Construct the Hamiltonian expansion (eq. 4) around the chosen collinear point,
    already translated & scaled so that (x,y,z)=(0,0,0) is the equilibrium.

    Returns a Polynomial in the *physical* coordinates (x,y,z,px,py,pz).
    """
    T = _build_T_polynomials(max_degree)

    c = [None, None]   # c_0, c_1 unused
    for n in range(2, max_degree + 1):
        # c.append(point._cn(n))
        c.append(create_symbolic_cn(n))

    U = -se.Add(*[c[n] * T[n] for n in range(2, max_degree + 1)])

    K = se.Rational(1,2)*(px**2 + py**2 + pz**2) + y*px - x*py

    H_phys = se.expand(K + U)

    return Polynomial([x, y, z, px, py, pz], H_phys)


def hamiltonian_arrays(point, max_degree, psi, clmo):
    """
    Return H_phys as list[np.ndarray] in the order [x,y,z,px,py,pz].
    """
    expr = hamiltonian(point, max_degree).expansion.expression
    return symengine_to_custom_poly(expr,
                                    [x, y, z, px, py, pz],
                                    max_degree,
                                    psi, clmo,
                                    complex_dtype=False)


def physical_to_real_normal(point: LibrationPoint, H_phys: Polynomial, symbolic: bool = False, max_degree: int = None) -> Polynomial:
    """
    Express the physical CR3BP Hamiltonian around a collinear point in the
    real normal-form variables (x,y,z,px,py,pz) used for the centre-
    manifold normal-form computation (Jorba & Masdemont, 1999).

    Parameters
    ----------
    point : LibrationPoint
        The LibrationPoint object
    H_phys : Polynomial
        The physical Hamiltonian to transform

    Returns
    -------
    Polynomial
        The transformed Hamiltonian in real normal-form variables with symbolic parameters
    """
    # Get the symbolic transformation matrices
    C, Cinv = point._symbolic_normal_form_transform()

    # Create the new coordinate vector
    Z_new = se.Matrix([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn])

    # Build the substitution dictionary from physical to real normal coordinates
    subs_dict = {}
    for i, var in enumerate([x, y, z, px, py, pz]):
        expr = 0
        for j in range(6):
            expr += C[i, j] * Z_new[j]
        subs_dict[var] = se.expand(expr)
    
    H_rn = H_phys.subs(subs_dict).expansion.expression

    if not symbolic:
        if not max_degree:
            err = "Max degree must be provided if symbolic is False"
            logger.error(err)
            raise ValueError(err)

        subs_dict = _generate_subs_dict(point, max_degree)
        H_rn = H_rn.subs(subs_dict)
        H_rn_sp = sp.sympify(H_rn)
        H_rn_clean = _clean_numerical_artifacts(H_rn_sp)
        H_rn = se.sympify(H_rn_clean)

        # logger.debug(f"\n\nH_rn:\n\n{H_rn}\n\n")

        return Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], H_rn)

    return Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], H_rn)

def real_normal_to_complex_canonical(point: LibrationPoint, H_real_normal: Polynomial, symbolic: bool = False, max_degree: int = None) -> Polynomial:
    """
    Express the physical CR3BP Hamiltonian around a collinear point in the
    complex canonical variables (q1,q2,q3,p1,p2,p3) used for the centre-
    manifold normal-form computation (Jorba & Masdemont, 1999).

    Parameters
    ----------
    point : LibrationPoint
        Must implement `normal_form_transform()` → (C, Cinv) with C \subset \mathbb{R}^{6 \times 6}.
    H_real_normal : Polynomial
        Hamiltonian in the real normal-form variables
        (x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn).

    Returns
    -------
    Polynomial
        Same Hamiltonian written in (q1,q2,q3,p1,p2,p3).
    """

    sqrt2 = se.sqrt(2)
    complex_subs = {
        x_rn : q1,
        y_rn : (q2 + se.I*p2) / sqrt2,
        z_rn : (q3 + se.I*p3) / sqrt2,
        px_rn: p1,
        py_rn: (se.I*q2 + p2) / sqrt2,
        pz_rn: (se.I*q3 + p3) / sqrt2,
    }

    H_cn_expr = H_real_normal.subs(complex_subs).expansion.expression

    if not symbolic:
        if not max_degree:
            err = "Max degree must be provided if symbolic is False"
            logger.error(err)
            raise ValueError(err)

        subs_dict = _generate_subs_dict(point, max_degree)
        H_cn_expr = H_cn_expr.subs(subs_dict)

        H_cn_expr_sp = sp.sympify(H_cn_expr)
        H_cn_expr_clean = _clean_numerical_artifacts(H_cn_expr_sp)
        H_cn = se.sympify(H_cn_expr_clean)

        # logger.debug(f"\n\nH_cn_expr_clean:\n\n{H_cn_expr_clean}\n\n")

        return Polynomial([q1, q2, q3, p1, p2, p3], H_cn)

    return Polynomial([q1, q2, q3, p1, p2, p3], H_cn_expr)

def complex_canonical_to_real_normal(point: LibrationPoint, H_complex_canonical: Polynomial, symbolic: bool = False, max_degree: int = None) -> Polynomial:
    """
    Express the Hamiltonian in the real normal-form variables (x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn)
    from the complex canonical variables (q1, q2, q3, p1, p2, p3).
    
    Parameters
    ----------
    point : LibrationPoint
        The LibrationPoint object
    H_complex_canonical : Polynomial
        The Hamiltonian in complex canonical variables
    
    Returns
    -------
    Polynomial
        The transformed Hamiltonian in real normal-form variables
    """
    sqrt2 = se.sqrt(2)
    # Implement the exact inverse of the transformation in real_normal_to_complex_canonical
    # This is directly derived from the equations in (12)

    real_subs = {
        q1: x_rn,
        q2: (y_rn - se.I*py_rn) / sqrt2,
        q3: (z_rn - se.I*pz_rn) / sqrt2, 
        p1: px_rn,
        p2: (py_rn - se.I*y_rn) / sqrt2,
        p3: (pz_rn - se.I*z_rn) / sqrt2
    }

    H_rn_expr = H_complex_canonical.subs(real_subs).expansion.expression

    if not symbolic:
        if not max_degree:
            err = "Max degree must be provided if symbolic is False"
            logger.error(err)
            raise ValueError(err)

        subs_dict = _generate_subs_dict(point, max_degree)
        H_rn = H_rn_expr.subs(subs_dict)

        H_rn_sp = sp.sympify(H_rn)
        H_rn_clean = _clean_numerical_artifacts(H_rn_sp)
        H_rn = se.sympify(H_rn_clean)

        # logger.debug(f"\n\nH_rn:\n\n{H_rn}\n\n")

        return Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], H_rn)

    return Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], H_rn_expr)

def _generate_subs_dict(point: LibrationPoint, max_degree: int) -> dict:

    # Generate all necessary c symbols based on degree
    c_symbols = {c2: c2}  # Start with c2 which is already defined
    for n in range(3, max_degree+1):
        c_symbols[create_symbolic_cn(n)] = create_symbolic_cn(n)

    lambda1_num, omega1_num, omega2_num = point.linear_modes()
    s1_num, s2_num = point._scale_factor(lambda1_num, omega1_num, omega2_num)

    c_nums = {}
    for n in range(2, max_degree+1):
        c_sym = create_symbolic_cn(n)
        c_num = point._cn(n)
        c_nums[c_sym] = c_num

    subs_dict = {
        lambda1: lambda1_num, 
        omega1: omega1_num, 
        omega2: omega2_num,
        s1: s1_num, 
        s2: s2_num
    }

    for c_sym, c_val in c_nums.items():
        subs_dict[c_sym] = c_val

    return subs_dict
