r"""
polynomial.base
===============

Low-level helpers for manipulating multivariate polynomial coefficient arrays
used throughout the library.  The routines defined here are performance-critical
Numba-JIT compiled kernels that:

* build combinatorial lookup tables (:pyfunc:`init_index_tables`) for fast
  encoding/decoding of monomial multi-indices,
* provide packing/unpacking utilities for exponents in a compact
  32-bit representation (:pyfunc:`pack_multiindex`,
  :pyfunc:`decode_multiindex`, :pyfunc:`encode_multiindex`,
  :pyfunc:`fill_exponents`),
* expose lightweight algebraic helpers such as factorial and binomial
  coefficients (:pyfunc:`_factorial`, :pyfunc:`_combinations`), and
* create suitably-shaped zero coefficient arrays (:pyfunc:`make_poly`).

The module is intentionally dependency-light (only *NumPy* and *Numba*) so that
these utilities can be reused by both CPU and GPU back-ends without circular
dependencies.
"""

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from hiten.utils.config import FASTMATH, N_VARS


@njit(fastmath=FASTMATH,cache=False)
def _factorial(n: int) -> int:
    r"""
    Calculate the factorial of a non-negative integer.
    
    Parameters
    ----------
    n : int
        Non-negative integer to calculate factorial for
        
    Returns
    -------
    int
        The factorial :math:`n! = n * (n-1) * ... * 2 * 1`
        
    Notes
    -----
    Optimized for Numba with fastmath and caching
    """
    if n < 0:
        pass
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res

@njit(fastmath=FASTMATH, cache=False)
def _combinations(n: int, k: int) -> int:
    r"""
    Calculate the binomial coefficient :math:`C(n,k) = n! / (k! * (n-k)!)`.
    
    Parameters
    ----------
    n : int
        Total number of items
    k : int
        Number of items to choose
        
    Returns
    -------
    int
        The number of ways to choose k items from n items,
        which equals :math:`n! / (k! * (n-k)!)`
        
    Notes
    -----
    Implementation uses an optimized approach to avoid calculating
    full factorials, which could cause numeric overflow for large values.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    if k == 0:
        return 1
    
    res = 1
    for i in range(1, k + 1):
        res = res * (n - i + 1) // i
    return res

@njit(fastmath=FASTMATH,cache=False)
def init_index_tables(max_degree: int):
    r"""
    Initialize lookup tables for polynomial multi-index encoding and decoding.
    
    This function creates two data structures essential for polynomial operations:
    1. A table of combinations (psi) that counts monomials for given degrees
    2. A list of packed multi-indices (clmo) that efficiently stores monomial exponents
    
    Parameters
    ----------
    max_degree : int
        Maximum polynomial degree to initialize tables for
        
    Returns
    -------
    psi : numpy.ndarray
        2D array where psi[i, d] contains the number of monomials of degree d 
        in i variables. Shape is (N_VARS+1, max_degree+1)
    
    clmo : numba.typed.List
        List of arrays where clmo[d] contains packed representations of all
        multi-indices for monomials of degree d. Each multi-index is packed
        into a uint32 value for efficient storage and lookup.
        
    Notes
    -----
    The packing scheme allocates 6 bits for each variable :math:`x_1` through :math:`x_5`,
    with :math:`x_0`'s exponent implicitly determined by the total degree.
    """
    psi = np.zeros((N_VARS+1, max_degree+1), dtype=np.int64)
    for i_vars_count in range(1, N_VARS+1):
        for d_degree in range(max_degree+1):
            # psi[i, d] = math.comb(d + i - 1, i - 1)
            # n = d + i_vars_count - 1
            # k = i_vars_count - 1
            # psi[i_vars_count, d_degree] = _factorial(n) // (_factorial(k) * _factorial(n - k))
            psi[i_vars_count, d_degree] = _combinations(d_degree + i_vars_count - 1, i_vars_count - 1)
    psi[0, 0] = 1

    clmo = List.empty_list(np.uint32[::1]) # Ensure clmo is typed correctly for Numba
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

# -----------------------------------------------------------------------------
#  GLOBAL clmo cache (Numba functions need it at definition time)
# -----------------------------------------------------------------------------
PSI_GLOBAL, CLMO_GLOBAL = init_index_tables(30)  # default; will be overwritten

# -----------------------------------------------------------------------------
# Build a Numba‐typed lookup: for each degree d, a dict mapping
# packed_exponent → index in CLMO_GLOBAL[d]
# -----------------------------------------------------------------------------
ENCODE_DICT_GLOBAL = List.empty_list(
    types.DictType(types.int64, types.int32)
)
for clmo_arr in CLMO_GLOBAL:
    d = Dict.empty(
        key_type=types.int64,
        value_type=types.int32,
    )
    # populate the dict once
    for idx, packed_val in enumerate(clmo_arr):
        # Explicitly cast key to int64 and value to int32 for the dictionary
        d[np.int64(packed_val)] = np.int32(idx)
    ENCODE_DICT_GLOBAL.append(d)

# -----------------------------------------------------------------------------

@njit(fastmath=FASTMATH, cache=False)
def pack_multiindex(k: np.ndarray) -> np.uint32:
    r"""
    Pack the exponents k_1 through k_5 into a 32-bit integer.
    
    Parameters
    ----------
    k : numpy.ndarray
        Array of length N_VARS containing the exponents [k_0, k_1, k_2, k_3, k_4, k_5]
        
    Returns
    -------
    numpy.uint32
        A packed 32-bit integer where:
        - k[1] uses bits 0-5
        - k[2] uses bits 6-11
        - k[3] uses bits 12-17
        - k[4] uses bits 18-23
        - k[5] uses bits 24-29
        
    Notes
    -----
    k[0] is not included in the packed value. Each exponent uses 6 bits,
    limiting its maximum value to 63.
    """
    # This logic is derived from the original encode_multiindex and init_index_tables
    packed = (
        (k[1] & 0x3F)
        | ((k[2] & 0x3F) << 6)
        | ((k[3] & 0x3F) << 12)
        | ((k[4] & 0x3F) << 18)
        | ((k[5] & 0x3F) << 24)
    )
    return np.uint32(packed) # Ensure it returns uint32 as in init_index_tables

@njit(fastmath=FASTMATH, cache=False)
def decode_multiindex(pos: int, degree: int, clmo) -> np.ndarray:
    r"""
    Decode a packed multi-index from its position in the lookup table.
    
    Parameters
    ----------
    pos : int
        Position of the multi-index in the clmo[degree] array
    degree : int
        Degree of the monomial
    clmo : numba.typed.List
        List of arrays containing packed multi-indices, as returned by init_index_tables
        
    Returns
    -------
    k : numpy.ndarray
        Array of length N_VARS containing the exponents [k_0, k_1, k_2, k_3, k_4, k_5]
        where k_0 + k_1 + k_2 + k_3 + k_4 + k_5 = degree
        
    Notes
    -----
    The function unpacks a 32-bit integer where:
    - k_1 uses bits 0-5
    - k_2 uses bits 6-11
    - k_3 uses bits 12-17
    - k_4 uses bits 18-23
    - k_5 uses bits 24-29
    
    k_0 is calculated as (degree - sum of other exponents)
    """
    packed = clmo[degree][pos]
    k1 =  packed        & 0x3F
    k2 = (packed >>  6) & 0x3F
    k3 = (packed >> 12) & 0x3F
    k4 = (packed >> 18) & 0x3F
    k5 = (packed >> 24) & 0x3F
    k0 = degree - (k1+k2+k3+k4+k5)
    return k0, k1, k2, k3, k4, k5

@njit(fastmath=FASTMATH, inline='always', cache=False)
def fill_exponents(pos, degree, clmo, out):
    packed = clmo[degree][pos]
    out[1] =  packed        & 0x3F
    out[2] = (packed >>  6) & 0x3F
    out[3] = (packed >> 12) & 0x3F
    out[4] = (packed >> 18) & 0x3F
    out[5] = (packed >> 24) & 0x3F
    out[0] = degree - (out[1]+out[2]+out[3]+out[4]+out[5])


@njit(fastmath=FASTMATH, cache=False)
def encode_multiindex(k: np.ndarray, degree: int, encode_dict_list: List) -> int:
    r"""
    Encode a multi-index to find its position in the coefficient array.
    
    Parameters
    ----------
    k : numpy.ndarray
        Array of length N_VARS containing the exponents [k_0, k_1, k_2, k_3, k_4, k_5]
    degree : int
        Degree of the monomial (should equal sum of elements in k)
    encode_dict_list : numba.typed.List
        The precomputed list of dictionaries (e.g., ENCODE_DICT_GLOBAL)
        mapping packed exponents (int64) to indices (int32).
        
    Returns
    -------
    int
        The position of the multi-index in the coefficient array for the given degree,
        or -1 if the multi-index is not found.
        
    Notes
    -----
    This function uses a precomputed dictionary list for O(1) lookup.
    It packs the exponents k_1 through k_5 into a 32-bit integer (uint32),
    casts it to int64 for lookup, and expects an int32 index.
    """
    packed_val = pack_multiindex(k) # pack_multiindex returns uint32
    packed_key = np.int64(packed_val) # Explicitly cast to int64 for dict key type

    # Ensure degree is within the bounds of encode_dict_list
    if degree < 0 or degree >= len(encode_dict_list):
        return -1 # Degree out of bounds for the lookup table

    current_dict = encode_dict_list[degree]
    
    # Check if key exists, then get it. This helps Numba with type stability.
    if packed_key in current_dict:
        return current_dict[packed_key]  # This should be int32, returned as int
    else:
        return -1 # Key not found

@njit(fastmath=FASTMATH, cache=False)
def make_poly(degree: int, psi) -> np.ndarray:
    r"""
    Create a new polynomial coefficient array of specified degree with complex128 dtype.
    
    Parameters
    ----------
    degree : int
        Degree of the polynomial
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
        
    Returns
    -------
    numpy.ndarray
        Array of zeros with complex128 data type and size equal to the number
        of monomials of degree 'degree' in N_VARS variables
        
    Notes
    -----
    The size of the array is determined by psi[N_VARS, degree], which gives
    the number of monomials of degree 'degree' in N_VARS variables.
    All polynomials use complex128 data type for consistency.
    """
    size = psi[N_VARS, degree]
    return np.zeros(size, dtype=np.complex128)


# Helper to create encode_dict_list from clmo_table
@njit(fastmath=FASTMATH, cache=False)
def _create_encode_dict_from_clmo(clmo_table: List) -> List:
    r"""
    Create a list of dictionaries mapping packed multi-indices to their positions.
    
    Parameters
    ----------
    clmo_table : numba.typed.List
        List of arrays where each array contains packed multi-indices for a specific degree
        
    Returns
    -------
    numba.typed.List
        List of dictionaries where each dictionary maps a packed multi-index (int64)
        to its position (int32) in the corresponding clmo_table array
        
    Notes
    -----
    This is a helper function used to build the ENCODE_DICT_GLOBAL structure.
    Each dictionary provides O(1) lookup time for finding the position
    of a multi-index in the coefficient array.
    """
    # Create an empty list that will hold dictionaries
    encode_dict_list = List()
    for clmo_arr in clmo_table:
        d_map = Dict()
        for i, packed_val in enumerate(clmo_arr):
            d_map[np.int64(packed_val)] = np.int32(i)
        encode_dict_list.append(d_map)
    return encode_dict_list
