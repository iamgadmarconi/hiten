import numpy as np
import math
from numba import njit
from numba.typed import List
import symengine as se

from algorithms.variables import N_VARS


def init_index_tables(max_degree: int):
    """
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
    The packing scheme allocates 6 bits for each variable x_1 through x_5,
    with x_0's exponent implicitly determined by the total degree.
    """
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

# -----------------------------------------------------------------------------
#  GLOBAL clmo cache (Numba functions need it at definition time)
# -----------------------------------------------------------------------------
PSI_GLOBAL, CLMO_GLOBAL = init_index_tables(30)  # default; will be overwritten

@njit(fastmath=True, cache=True)
def decode_multiindex(pos: int, degree: int, clmo) -> np.ndarray:
    """
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
    """
    Encode a multi-index to find its position in the coefficient array.
    
    Parameters
    ----------
    k : numpy.ndarray
        Array of length N_VARS containing the exponents [k_0, k_1, k_2, k_3, k_4, k_5]
    degree : int
        Degree of the monomial (should equal sum of elements in k)
    psi : numpy.ndarray
        Combinatorial table from init_index_tables
    clmo : numba.typed.List
        List of arrays containing packed multi-indices from init_index_tables
        
    Returns
    -------
    int
        The position of the multi-index in the coefficient array for the given degree,
        or -1 if the multi-index is not found
        
    Notes
    -----
    This function is the inverse of decode_multiindex. It packs the exponents
    k_1 through k_5 into a 32-bit integer and searches for this value in clmo[degree].
    """
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


@njit(fastmath=True, cache=True)
def make_poly(degree: int, psi) -> np.ndarray:
    """
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
