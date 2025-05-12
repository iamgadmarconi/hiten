import numpy as np
import math
from numba import njit
from numba.typed import List
import symengine as se

from algorithms.variables import N_VARS


def init_index_tables(max_degree: int):
    psi = np.zeros((N_VARS+1, max_degree+1), dtype=np.int64)
    for i in range(1, N_VARS+1):
        for d in range(max_degree+1):
            psi[i, d] = math.comb(d + i - 1, i - 1)
    psi[0, 0] = 1

    clmo = List()
    for d in range(max_degree+1):
        count = psi[N_VARS, d]
        arr = np.empty(count, dtype=np.uint32)
        idx = 0
        for k0 in range(d, -1, -1):
            for k1 in range(d - k0, -1, -1):
                for k2 in range(d - k0 - k1, -1, -1):
                    for k3 in range(d - k0 - k1 - k2, -1, -1):
                        for k4 in range(d - k0 - k1 - k2 - k3, -1, -1):
                            k5 = d - k0 - k1 - k2 - k3 - k4
                            packed = (
                                (k1 & 0x3F)
                                | ((k2 & 0x3F) << 6)
                                | ((k3 & 0x3F) << 12)
                                | ((k4 & 0x3F) << 18)
                                | ((k5 & 0x3F) << 24)
                            )
                            arr[idx] = np.uint32(packed)
                            idx += 1
        clmo.append(arr)
    return psi, clmo


@njit(fastmath=True, cache=True)
def decode_multiindex(pos: int, degree: int, clmo) -> np.ndarray:
    packed = clmo[degree][pos]
    k = np.empty(N_VARS, dtype=np.int64)
    k[1] = packed & 0x3F
    k[2] = (packed >> 6) & 0x3F
    k[3] = (packed >> 12) & 0x3F
    k[4] = (packed >> 18) & 0x3F
    k[5] = (packed >> 24) & 0x3F
    s = k[1] + k[2] + k[3] + k[4] + k[5]
    k[0] = degree - s
    return k


@njit(fastmath=True, cache=True)
def encode_multiindex(k: np.ndarray, degree: int, psi, clmo) -> int:
    packed = (
        (k[1] & 0x3F)
        | ((k[2] & 0x3F) << 6)
        | ((k[3] & 0x3F) << 12)
        | ((k[4] & 0x3F) << 18)
        | ((k[5] & 0x3F) << 24)
    )
    arr = clmo[degree]
    for idx in range(arr.shape[0]):
        if arr[idx] == packed:
            return idx
    return -1


def make_poly(degree: int, psi, complex_dtype: bool=False) -> np.ndarray:
    if complex_dtype:
        return _make_poly_complex(degree, psi)
    else:
        return _make_poly_real(degree, psi)


@njit(fastmath=True, cache=True)
def _make_poly_real(degree: int, psi) -> np.ndarray:
    size = psi[N_VARS, degree]
    return np.zeros(size, dtype=np.float64)


@njit(fastmath=True, cache=True)
def _make_poly_complex(degree: int, psi) -> np.ndarray:
    size = psi[N_VARS, degree]
    return np.zeros(size, dtype=np.complex128)


def _extract_symengine_term_details(term: se.Basic, variables: list[se.Symbol]) -> tuple[complex | float, np.ndarray, int]:
    """
    Extracts coefficient, multi-index, and degree from a single symengine term (monomial).
    The multi-index corresponds to the order of symbols in `variables`.
    Assumes `term` is a monomial (coeff * var1**exp1 * var2**exp2 ...).
    """
    k = np.zeros(N_VARS, dtype=np.int64)
    
    # Handle purely numeric terms first
    if isinstance(term, se.Number): # Includes Integer, Rational, Float, Complex, NumberSymbol
        coeff_expr = term
        var_expr = se.Integer(1) # Represents no variable part
    else:
        try:
            coeff_expr, var_expr = term.as_coeff_mul()
        except AttributeError:
            # This might happen if term is a Symbol or Pow but not in a Mul, treat full term as variable part
            # if it's not a Number. e.g. term = x0 or term = x0**2
            coeff_expr = se.Integer(1) # Coefficient is 1
            var_expr = term

    if isinstance(var_expr, se.Mul):
        factors = var_expr.args
    elif var_expr == se.Integer(1): # No variable part, e.g., term is just a number
        factors = []
    else: # Single variable or Pow instance
        factors = [var_expr]

    internal_coeff_accumulator = 1.0 # Initialize to float 1.0

    for factor in factors:
        if isinstance(factor, se.Pow):
            base, exp_obj = factor.args
            # Ensure exponent is a SymEngine Integer and convert to Python int
            if not isinstance(exp_obj, se.Integer):
                raise ValueError(f"Exponent in Pow '{factor}' is not an integer: {exp_obj}")
            exp = int(exp_obj)
            try:
                var_idx = variables.index(base)
                k[var_idx] = exp
            except ValueError:
                raise ValueError(f"Variable '{base}' in term '{term}' not found in variables list: {variables}")
        elif isinstance(factor, se.Symbol):
            try:
                var_idx = variables.index(factor)
                # Use direct assignment for safety, assuming expand() handles x*x -> x**2
                # If x appears multiple times in a Mul not fully simplified, this needs k[var_idx] += 1
                # However, se.expand() prior to this should make terms canonical.
                if k[var_idx] != 0: # Check if exponent for this var was already set (e.g. by a Pow)
                     raise ValueError(f"Variable '{factor}' appears in multiple forms in term '{term}'. Expression might not be canonical.")
                k[var_idx] = 1 # Assuming it appears once as a symbol factor after expand
            except ValueError:
                raise ValueError(f"Variable '{factor}' in term '{term}' not found in variables list: {variables}")
        elif isinstance(factor, se.Number):
            # This numeric factor is part of var_expr, accumulate its value.
            num_factor_val = factor.evalf()
            if isinstance(internal_coeff_accumulator, complex) or isinstance(num_factor_val, (se.ComplexMPC, se.ComplexDouble)):
                if not isinstance(internal_coeff_accumulator, complex):
                    internal_coeff_accumulator = complex(internal_coeff_accumulator)
                internal_coeff_accumulator *= complex(num_factor_val)
            else:
                internal_coeff_accumulator *= float(num_factor_val)
        else:
            # This case should not be reached if var_expr is correctly structured (product of Pows and Symbols)
            raise ValueError(f"Unexpected factor type '{type(factor)}' (value: {factor}) in variable part of term '{term}'")

    numeric_coeff_val: float | complex
    try:
        # Evaluate the coefficient part to a numerical value.
        # .evalf() helps resolve symbolic constants like pi, E and ensures float/complex representation.
        eval_coeff = coeff_expr.evalf() 
        if isinstance(eval_coeff, (se.ComplexMPC, se.ComplexDouble)):
            numeric_coeff_val = complex(eval_coeff)
        elif isinstance(eval_coeff, (se.Integer, se.Rational, se.Float)):
            numeric_coeff_val = float(eval_coeff)
        elif isinstance(eval_coeff, se.Symbol) and eval_coeff.is_Symbol:
             # This implies an unresolved symbolic coefficient that wasn't substituted.
             raise ValueError(f"Term '{term}' has an unresolved symbolic coefficient '{eval_coeff}'. All symbolic parameters should be substituted with numerical values before conversion.")
        else:
            # If eval_coeff is still some other SymEngine type not convertible, it's an error.
            raise TypeError(f"Coefficient '{coeff_expr}' (type {type(coeff_expr)}) of term '{term}' evaluated to unhandled numeric type {type(eval_coeff)}.")
    except AttributeError: # If coeff_expr doesn't have evalf (e.g. already Python type)
        if isinstance(coeff_expr, (int, float, complex)):
            numeric_coeff_val = coeff_expr
        else:
            raise TypeError(f"Coefficient '{coeff_expr}' (type {type(coeff_expr)}) of term '{term}' is not a SymEngine numeric type and lacks .evalf().")
    except Exception as e: # Catch other errors during evalf or conversion
        raise TypeError(f"Coefficient '{coeff_expr}' (type {type(coeff_expr)}) of term '{term}' could not be converted to a numeric value. Original error: {e}")

    # Combine the coefficient from coeff_expr with any numeric factors found in var_expr
    if isinstance(numeric_coeff_val, complex) or isinstance(internal_coeff_accumulator, complex):
        if not isinstance(numeric_coeff_val, complex):
            numeric_coeff_val = complex(numeric_coeff_val)
        if not isinstance(internal_coeff_accumulator, complex):
            internal_coeff_accumulator = complex(internal_coeff_accumulator)
    
    final_numeric_coeff = numeric_coeff_val * internal_coeff_accumulator

    term_degree = int(sum(k))
    return final_numeric_coeff, k, term_degree


def symengine_to_custom_poly(
    expr: se.Basic, 
    variables: list[se.Symbol], 
    max_degree: int, 
    psi,  # from init_index_tables
    clmo, # from init_index_tables
    complex_dtype: bool = False,
    tolerance: float = 1e-18
) -> list[np.ndarray]:
    """
    Converts a symengine expression to a list of custom polynomial arrays.
    Each element in the list corresponds to a homogeneous polynomial of a specific degree.
    `variables` must be a list of N_VARS symengine symbols in the desired order.
    Coefficients smaller than `tolerance` are treated as zero.
    """
    if len(variables) != N_VARS:
        raise ValueError(f"Expected {N_VARS} variables, got {len(variables)}")

    poly_list = [make_poly(d, psi, complex_dtype) for d in range(max_degree + 1)]

    expanded_expr = se.expand(expr)
    
    terms_to_process = []
    if isinstance(expanded_expr, se.Add):
        terms_to_process.extend(expanded_expr.args)
    elif expanded_expr != 0: # Handle single term expressions or non-zero constants
        terms_to_process.append(expanded_expr)
    # If expanded_expr is 0, terms_to_process remains empty, and an empty list of zero polynomials is returned.

    for term in terms_to_process:
        coeff, k_multi_index, term_degree = _extract_symengine_term_details(term, variables)
        
        if term_degree > max_degree:
            # Optionally, log a warning here if terms are being truncated.
            continue

        # Handle complex coefficients and complex_dtype consistency
        current_coeff_value: float | complex
        if isinstance(coeff, complex):
            if not complex_dtype and abs(coeff.imag) > tolerance:
                raise ValueError(
                    f"Complex coefficient {coeff} encountered for term {term} (degree {term_degree}), "
                    f"but complex_dtype is False. Set complex_dtype=True or ensure expression is real."
                )
            current_coeff_value = coeff if complex_dtype else coeff.real
        else: # coeff is float
            current_coeff_value = coeff
        
        if abs(current_coeff_value) > tolerance: # Check magnitude against tolerance
            flat_idx = encode_multiindex(k_multi_index, term_degree, psi, clmo)
            if flat_idx == -1: # Should not happen if k and term_degree are valid
                raise ValueError(f"Failed to encode multi-index {k_multi_index} for degree {term_degree} of term {term}")
            
            try:
                poly_list[term_degree][flat_idx] += current_coeff_value
            except IndexError:
                 # This might happen if flat_idx is out of bounds for poly_list[term_degree]
                 # which indicates an issue with encode_multiindex or psi table sizing.
                 raise IndexError(f"Index {flat_idx} out of bounds for polynomial of degree {term_degree} (size {poly_list[term_degree].size}). Term: {term}, Multi-index: {k_multi_index}")


    return poly_list
