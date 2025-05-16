from dataclasses import dataclass
import numpy as np
from numba import types, njit
from numba.experimental import jitclass
from numba.typed import List

# Assuming these imports are resolvable relative to polynomial.py
from algorithms.center.polynomial.base import init_index_tables
# N_VARS is used by polynomial_degree and other underlying functions implicitly.
# Ensure N_VARS from algorithms.variables is accessible in the environment where these are run.
from algorithms.center.polynomial.operations import (
    polynomial_zero_list,
    polynomial_variable,
    polynomial_add_inplace,
    polynomial_multiply,
    polynomial_power,
    polynomial_poisson_bracket,
    polynomial_clean,
    polynomial_degree,
    polynomial_differentiate
)
from algorithms.center.polynomial.algebra import _poly_scale, _poly_diff # _get_degree is used by polynomial_degree


@njit
def _njit_deep_copy_typed_list_of_arrays_uint32(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Deep copies a Numba Typed List of 1D uint32 NumPy arrays."""
    copied_list = List.empty_list(types.uint32[::1])
    for arr in list_of_arrays:
        copied_list.append(arr.copy())
    return copied_list

@njit
def _njit_deep_copy_typed_list_of_arrays_complex128(list_of_arrays: List[np.ndarray]) -> List[np.ndarray]:
    """Deep copies a Numba Typed List of 1D complex128 NumPy arrays."""
    copied_list = List.empty_list(types.complex128[::1])
    for arr in list_of_arrays:
        copied_list.append(arr.copy())
    return copied_list

# jitclass specification
jit_polynomial_spec = [
    ('polynomials', types.ListType(types.complex128[::1])), # List of 1D complex arrays
    ('max_deg', types.int64),
    ('psi_table', types.int64[:,::1]), # 2D int64 array
    ('clmo_table', types.ListType(types.uint32[::1])) # List of 1D uint32 arrays
]

@jitclass(jit_polynomial_spec)
class JITPolynomial:
    def __init__(self, polynomials_list: List[np.ndarray], 
                 max_deg_val: int, 
                 psi_table_val: np.ndarray, 
                 clmo_table_val: List[np.ndarray]):
        self.polynomials = polynomials_list
        self.max_deg = max_deg_val
        self.psi_table = psi_table_val
        self.clmo_table = clmo_table_val

    def copy(self) -> 'JITPolynomial':
        """Creates a deep copy of the polynomial."""
        copied_polys = _njit_deep_copy_typed_list_of_arrays_complex128(self.polynomials)
        copied_clmo = _njit_deep_copy_typed_list_of_arrays_uint32(self.clmo_table)
        # psi_table is a NumPy array, .copy() is sufficient.
        return JITPolynomial(copied_polys, self.max_deg, self.psi_table.copy(), copied_clmo)

    def __add__(self, other: 'JITPolynomial') -> 'JITPolynomial':
        """Adds two polynomials: self + other."""
        res_max_deg = max(self.max_deg, other.max_deg)
        
        res_psi_table, res_clmo_table_list = init_index_tables(res_max_deg)
        # Initialize result coefficients list
        res_coeffs = polynomial_zero_list(res_max_deg, res_psi_table)
        
        # Add self to result
        polynomial_add_inplace(res_coeffs, self.polynomials, 1.0, self.max_deg)
        # Add other to result
        polynomial_add_inplace(res_coeffs, other.polynomials, 1.0, other.max_deg)
        
        return JITPolynomial(res_coeffs, res_max_deg, res_psi_table, res_clmo_table_list)

    def __sub__(self, other: 'JITPolynomial') -> 'JITPolynomial':
        """Subtracts two polynomials: self - other."""
        res_max_deg = max(self.max_deg, other.max_deg)
        res_psi_table, res_clmo_table_list = init_index_tables(res_max_deg)
        res_coeffs = polynomial_zero_list(res_max_deg, res_psi_table)

        polynomial_add_inplace(res_coeffs, self.polynomials, 1.0, self.max_deg)
        polynomial_add_inplace(res_coeffs, other.polynomials, -1.0, other.max_deg) # Scale by -1 for subtraction
        
        return JITPolynomial(res_coeffs, res_max_deg, res_psi_table, res_clmo_table_list)

    def __mul__(self, other: 'JITPolynomial') -> 'JITPolynomial':
        """Multiplies two polynomials: self * other."""
        # The result can have degree up to self.max_deg + other.max_deg.
        # This will be the max_deg for the new polynomial instance and its tables.
        res_max_deg = self.max_deg + other.max_deg
        
        res_psi_table, res_clmo_table_list = init_index_tables(res_max_deg)
        
        # polynomial_multiply's max_deg argument is the truncation degree for the output list.
        res_coeffs = polynomial_multiply(self.polynomials, other.polynomials, 
                                         res_max_deg, res_psi_table, res_clmo_table_list)
        
        return JITPolynomial(res_coeffs, res_max_deg, res_psi_table, res_clmo_table_list)

    def __pow__(self, k_exponent: int) -> 'JITPolynomial':
        """Raises the polynomial to the power k_exponent: self ** k_exponent."""
        if k_exponent < 0:
            # Or handle appropriately (e.g. inverse if applicable, or error)
            # For now, Numba does not support raising Python exceptions directly in all contexts like this.
            # A consuming function might check or this might lead to a Numba error.
            # For simplicity, assume k_exponent is non-negative.
            pass # Consider raising error if this were pure Python

        # The result is truncated at self.max_deg, using self's tables for the operation.
        # The new polynomial instance will also have self.max_deg.
        res_coeffs = polynomial_power(self.polynomials, k_exponent, self.max_deg,
                                      self.psi_table, self.clmo_table)
        
        return JITPolynomial(res_coeffs, self.max_deg, self.psi_table.copy(), 
                             _njit_deep_copy_typed_list_of_arrays_uint32(self.clmo_table))

    def scale(self, factor: complex) -> 'JITPolynomial':
        """Scales the polynomial by a factor."""
        # Initialize a new list for scaled coefficients, matching self's structure.
        scaled_coeffs_list = polynomial_zero_list(self.max_deg, self.psi_table)
        
        for d in range(self.max_deg + 1):
            # Check ensures we don't go out of bounds if self.polynomials is somehow shorter
            # than self.max_deg + 1, though it shouldn't be by construction.
            if d < len(self.polynomials) and d < len(scaled_coeffs_list):
                original_term_coeffs = self.polynomials[d]
                # _poly_scale(a_coeffs, alpha, out_coeffs)
                _poly_scale(original_term_coeffs, factor, scaled_coeffs_list[d])
        
        return JITPolynomial(scaled_coeffs_list, self.max_deg, self.psi_table.copy(), 
                             _njit_deep_copy_typed_list_of_arrays_uint32(self.clmo_table))

    def poisson_bracket(self, other: 'JITPolynomial') -> 'JITPolynomial':
        """Computes the Poisson bracket {self, other}."""
        # Theoretical max degree of the result {A,B} is deg(A) + deg(B) - 2.
        # Use property for actual degree.
        actual_deg_self = polynomial_degree(self.polynomials)
        actual_deg_other = polynomial_degree(other.polynomials)
        
        theoretic_res_deg = actual_deg_self + actual_deg_other - 2
        if theoretic_res_deg < 0:
            theoretic_res_deg = 0 # Result is a constant (likely zero)

        # The polynomial_poisson_bracket function truncates its output list at its 'max_deg' arg.
        # The JITPolynomial instance needs a max_deg for its tables.
        # Choose a truncation_max_deg that can hold the result.
        # Max of operands' max_deg or the theoretic_res_deg, whichever is larger.
        truncation_max_deg = max(self.max_deg, other.max_deg)
        truncation_max_deg = max(truncation_max_deg, theoretic_res_deg)

        res_psi_table, res_clmo_table_list = init_index_tables(truncation_max_deg)

        res_coeffs = polynomial_poisson_bracket(self.polynomials, other.polynomials,
                                                truncation_max_deg, # Truncation for the output list
                                                res_psi_table, res_clmo_table_list)
        
        return JITPolynomial(res_coeffs, truncation_max_deg, res_psi_table, res_clmo_table_list)

    def differentiate(self, var_idx: int) -> 'JITPolynomial':
        """Differentiates the polynomial with respect to variable var_idx."""
        
        # Calculate the max degree of the derivative
        derivative_max_deg = self.max_deg - 1
        if derivative_max_deg < 0:
            derivative_max_deg = 0 # Derivative of a constant is a (zero) constant

        # Initialize tables for the derivative polynomial
        derivative_psi_table, derivative_clmo_table = init_index_tables(derivative_max_deg)

        # Call the operations function, passing all required tables
        derivative_coeffs_list, _ = \
            polynomial_differentiate(
                self.polynomials,          # Original coefficients
                var_idx,                  # Variable to differentiate by
                self.max_deg,             # Original max degree
                self.psi_table,           # Original psi_table
                self.clmo_table,          # Original clmo_table
                derivative_psi_table,     # Derivative's psi_table
                derivative_clmo_table     # Derivative's clmo_table
            )
        
        return JITPolynomial(derivative_coeffs_list, derivative_max_deg, derivative_psi_table, derivative_clmo_table)

    def clean(self, tol: float) -> 'JITPolynomial':
        """Cleans small coefficients from the polynomial, returning a new polynomial."""
        # polynomial_clean is already @njit and returns a new Numba Typed List (deep copy).
        cleaned_coeffs = polynomial_clean(self.polynomials, tol)
        
        # New instance uses copies of tables.
        return JITPolynomial(cleaned_coeffs, self.max_deg, self.psi_table.copy(), 
                             _njit_deep_copy_typed_list_of_arrays_uint32(self.clmo_table))

    @property
    def degree(self) -> int:
        """Computes the effective degree of the polynomial."""
        # polynomial_degree takes the list of coefficient arrays.
        return polynomial_degree(self.polynomials)


# --- Factory Functions ---

@njit(fastmath=True)
def create_zero_jitpolynomial(max_deg_val: int) -> JITPolynomial:
    """Creates a JITPolynomial representing zero, up to max_deg_val."""
    psi, clmo = init_index_tables(max_deg_val)
    coeffs = polynomial_zero_list(max_deg_val, psi)
    return JITPolynomial(coeffs, max_deg_val, psi, clmo)

@njit(fastmath=True)
def create_variable_jitpolynomial(var_idx: int, max_deg_val: int) -> JITPolynomial:
    """Creates a JITPolynomial representing a single variable x_var_idx."""
    psi, clmo = init_index_tables(max_deg_val)
    coeffs = polynomial_variable(var_idx, max_deg_val, psi, clmo)
    return JITPolynomial(coeffs, max_deg_val, psi, clmo)
