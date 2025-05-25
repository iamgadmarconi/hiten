import random

import numpy as np
import pytest
import symengine as se
from numba import njit, types
from numba.typed import Dict, List

from algorithms.center.polynomial.base import (CLMO_GLOBAL, ENCODE_DICT_GLOBAL,
                                               PSI_GLOBAL, decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables, make_poly,
                                               _create_encode_dict_from_clmo)
from algorithms.variables import N_VARS

MAX_DEGREE = 5
PSI, CLMO = init_index_tables(MAX_DEGREE)
# Corresponding ENCODE_DICT for the global PSI, CLMO used in some tests
ENCODE_DICT = _create_encode_dict_from_clmo(CLMO)

if N_VARS == 6:
    s_vars = se.symbols(f'x0:{N_VARS}') 
else:
    s_vars = se.symbols(','.join([f'x{i}' for i in range(N_VARS)]))

def test_init_index_tables():
    """Test if the index tables are initialized correctly"""
    # Check dimensions of psi
    assert PSI.shape == (N_VARS+1, MAX_DEGREE+1)
    
    # Check a few known values of psi (binomial coefficients)
    assert PSI[1, 3] == 1  # Variables=1, Degree=3: only x^3
    assert PSI[2, 2] == 3  # Variables=2, Degree=2: x^2, xy, y^2
    assert PSI[3, 1] == 3  # Variables=3, Degree=1: x, y, z
    
    # Check clmo list length
    assert len(CLMO) == MAX_DEGREE + 1
    
    # Check sizes of clmo arrays for different degrees
    for d in range(MAX_DEGREE + 1):
        assert len(CLMO[d]) == PSI[N_VARS, d]

def test_make_poly():
    """Test creation of zero polynomials with complex128 dtype"""
    for degree in range(MAX_DEGREE + 1):
        # Test polynomial creation
        poly = make_poly(degree, PSI)
        
        # Check size
        expected_size = PSI[N_VARS, degree]
        assert poly.shape[0] == expected_size
        
        # Check if all coefficients are zero
        assert np.all(poly == 0.0)
        
        # Check data type is always complex128
        assert poly.dtype == np.complex128

def test_decode_multiindex():
    """Test decoding multiindices"""
    for degree in range(1, MAX_DEGREE + 1):
        # Get size of the polynomial for this degree
        size = PSI[N_VARS, degree]
        
        # For small degrees, test all indices
        # For larger degrees, test a subset
        if size <= 50:
            positions = range(size)
        else:
            # Test first 20, last 20, and 10 positions in the middle
            positions = list(range(20))
            positions.extend(range(size-20, size))
            positions.extend(range(size//2-5, size//2+5))
        
        for pos in positions:
            # Decode the position
            k = decode_multiindex(pos, degree, CLMO)
            
            # Verify array shape
            k = np.asarray(k)
            assert k.shape == (N_VARS,)
            
            # Verify sum of exponents equals degree
            assert np.sum(k) == degree
            
            # Verify all exponents are non-negative
            assert np.all(k >= 0)
            
            # Verify no exponent exceeds degree
            assert np.all(k <= degree)

def test_encode_multiindex():
    """Test encoding multiindices"""
    for degree in range(1, MAX_DEGREE + 1):
        size = PSI[N_VARS, degree]
        
        # Test positions using the same pattern as in decode test
        if size <= 50:
            positions = range(size)
        else:
            positions = list(range(20))
            positions.extend(range(size-20, size))
            positions.extend(range(size//2-5, size//2+5))
        
        for pos in positions:
            # Get a known valid multiindex
            k = decode_multiindex(pos, degree, CLMO)
            k = np.asarray(k)
            
            # Encode it and verify it matches the original position
            idx = encode_multiindex(k, degree, ENCODE_DICT)
            assert idx == pos
            
            # Test invalid encoding behavior by modifying the multiindex
            if np.any(k > 0):
                # Create an invalid multiindex by incrementing and decrementing exponents
                k_invalid = k.copy()
                # Find first non-zero exponent
                for i in range(N_VARS):
                    if k[i] > 0:
                        k_invalid[i] -= 1
                        k_invalid[(i+1) % N_VARS] += 1
                        # If this creates a valid but different multiindex, it will encode
                        # to a different position. If it's invalid pattern, it should return -1
                        idx_invalid = encode_multiindex(k_invalid, degree, ENCODE_DICT)
                        assert idx_invalid != pos  # Should be different or -1
                        break

def test_multi_index_roundtrip():
    """Test full encode-decode roundtrip for multiindices"""
    # Try comprehensive testing for all valid indices
    for degree in range(MAX_DEGREE + 1):
        size = PSI[N_VARS, degree]
        
        # Full test for small sizes, sample for larger ones
        if degree <= 2:  # Fully test degrees 0, 1, 2
            test_positions = range(size)
        else:
            # Sample positions across the range
            test_positions = []
            step = max(1, size // 20)  # Sample about 20 positions
            for i in range(0, size, step):
                test_positions.append(i)
            # Always include the first and last
            if size > 0:
                test_positions.append(0)
                test_positions.append(size - 1)
                
        # Test decode -> encode -> decode consistency
        for pos in test_positions:
            k1 = decode_multiindex(pos, degree, CLMO)
            idx = encode_multiindex(k1, degree, ENCODE_DICT)
            k2 = decode_multiindex(idx, degree, CLMO)
            
            # Verify positions match
            assert idx == pos
            
            # Verify multiindices are identical
            np.testing.assert_array_equal(k1, k2)


@pytest.mark.parametrize("deg", [0, 1, 2, 3, 4, 5, 6])
def test_encode_decode_roundtrip(deg):
    psi, clmo = init_index_tables(6) # Max degree for these tables is 6
    encode_dict_for_deg6 = _create_encode_dict_from_clmo(clmo) # Create corresponding encode_dict
    # draw 10 random exponent vectors of total degree = deg
    for _ in range(10):
        k = np.zeros(N_VARS, dtype=np.int64)
        remaining = deg
        for i in range(N_VARS - 1):
            k[i] = random.randint(0, remaining)
            remaining -= k[i]
        k[-1] = remaining
        idx = encode_multiindex(k, deg, encode_dict_for_deg6) # Use the created encode_dict
        assert idx >= 0
        k_back = decode_multiindex(idx, deg, clmo)
        assert np.array_equal(k_back, k)

    k = np.array([0,0,0,0,0,0], dtype=np.int64)
    deg = 0
    psi, clmo = init_index_tables(deg) # Local psi, clmo for this specific test setup
    # Need to build a local encode_dict_list for this specific psi, clmo
    local_encode_dict_list = List.empty_list(types.DictType(types.int64, types.int32))
    for d_arr in clmo:
        d_map = Dict.empty(key_type=types.int64, value_type=types.int32)
        for i, p_val in enumerate(d_arr):
            d_map[np.int64(p_val)] = np.int32(i)
        local_encode_dict_list.append(d_map)

    idx = encode_multiindex(k, deg, local_encode_dict_list) # Changed arguments
    assert idx == 0 # (0,0,0,0,0,0) is the first entry for degree 0