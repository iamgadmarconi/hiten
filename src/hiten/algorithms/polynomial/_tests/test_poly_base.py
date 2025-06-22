import random

import numpy as np
import pytest
from numba import types
from numba.typed import Dict, List

from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables, make_poly)
from hiten.utils.config import N_VARS

TEST_MAX_DEG = 5
PSI, CLMO = init_index_tables(TEST_MAX_DEG)
ENCODE_DICT = _create_encode_dict_from_clmo(CLMO)

def test_init_index_tables():
    """Test if the index tables are initialized correctly"""
    # Check dimensions of psi
    assert PSI.shape == (N_VARS+1, TEST_MAX_DEG+1)
    
    # Check a few known values of psi (binomial coefficients)
    assert PSI[1, 3] == 1  # Variables=1, Degree=3: only x^3
    assert PSI[2, 2] == 3  # Variables=2, Degree=2: x^2, xy, y^2
    assert PSI[3, 1] == 3  # Variables=3, Degree=1: x, y, z
    
    # Check clmo list length
    assert len(CLMO) == TEST_MAX_DEG + 1
    
    # Check sizes of clmo arrays for different degrees
    for d in range(TEST_MAX_DEG + 1):
        assert len(CLMO[d]) == PSI[N_VARS, d]

def test_make_poly():
    """Test creation of zero polynomials with complex128 dtype"""
    for degree in range(TEST_MAX_DEG + 1):
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
    for degree in range(1, TEST_MAX_DEG + 1):
        # Get size of the polynomial for this degree
        size = PSI[N_VARS, degree]
        
        if size <= 50:
            positions = range(size)
        else:
            positions = list(range(20))
            positions.extend(range(size-20, size))
            positions.extend(range(size//2-5, size//2+5))
        
        for pos in positions:
            k = decode_multiindex(pos, degree, CLMO)
            
            k = np.asarray(k)
            assert k.shape == (N_VARS,)
            
            assert np.sum(k) == degree
            
            assert np.all(k >= 0)
            
            assert np.all(k <= degree)

def test_encode_multiindex():
    """Test encoding multiindices"""
    for degree in range(1, TEST_MAX_DEG + 1):
        size = PSI[N_VARS, degree]
        
        if size <= 50:
            positions = range(size)
        else:
            positions = list(range(20))
            positions.extend(range(size-20, size))
            positions.extend(range(size//2-5, size//2+5))
        
        for pos in positions:
            k = decode_multiindex(pos, degree, CLMO)
            k = np.asarray(k)
            
            idx = encode_multiindex(k, degree, ENCODE_DICT)
            assert idx == pos
            
            if np.any(k > 0):
                k_invalid = k.copy()
                for i in range(N_VARS):
                    if k[i] > 0:
                        k_invalid[i] -= 1
                        k_invalid[(i+1) % N_VARS] += 1
                        idx_invalid = encode_multiindex(k_invalid, degree, ENCODE_DICT)
                        assert idx_invalid != pos
                        break

def test_multi_index_roundtrip():
    """Test full encode-decode roundtrip for multiindices"""
    for degree in range(TEST_MAX_DEG + 1):
        size = PSI[N_VARS, degree]
        
        if degree <= 2:
            test_positions = range(size)
        else:
            test_positions = []
            step = max(1, size // 20)
            for i in range(0, size, step):
                test_positions.append(i)
            if size > 0:
                test_positions.append(0)
                test_positions.append(size - 1)
                
        for pos in test_positions:
            k1 = decode_multiindex(pos, degree, CLMO)
            idx = encode_multiindex(k1, degree, ENCODE_DICT)
            k2 = decode_multiindex(idx, degree, CLMO)
            
            assert idx == pos
            
            np.testing.assert_array_equal(k1, k2)


@pytest.mark.parametrize("deg", [0, 1, 2, 3, 4, 5, 6])
def test_encode_decode_roundtrip(deg):
    _, clmo = init_index_tables(6)
    encode_dict_for_deg6 = _create_encode_dict_from_clmo(clmo)
    for _ in range(10):
        k = np.zeros(N_VARS, dtype=np.int64)
        remaining = deg
        for i in range(N_VARS - 1):
            k[i] = random.randint(0, remaining)
            remaining -= k[i]
        k[-1] = remaining
        idx = encode_multiindex(k, deg, encode_dict_for_deg6)
        assert idx >= 0
        k_back = decode_multiindex(idx, deg, clmo)
        assert np.array_equal(k_back, k)

    k = np.array([0,0,0,0,0,0], dtype=np.int64)
    deg = 0
    _, clmo = init_index_tables(deg)
    local_encode_dict_list = List.empty_list(types.DictType(types.int64, types.int32))
    for d_arr in clmo:
        d_map = Dict.empty(key_type=types.int64, value_type=types.int32)
        for i, p_val in enumerate(d_arr):
            d_map[np.int64(p_val)] = np.int32(i)
        local_encode_dict_list.append(d_map)

    idx = encode_multiindex(k, deg, local_encode_dict_list)
    assert idx == 0
