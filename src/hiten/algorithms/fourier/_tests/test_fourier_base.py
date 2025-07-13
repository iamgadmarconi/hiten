import random

import numpy as np
import pytest

from hiten.algorithms.fourier.base import (_MAX_K, _MAX_N,
                                           _create_encode_dict_fourier,
                                           _decode_fourier_index,
                                           _encode_fourier_index,
                                           _init_fourier_tables,
                                           _pack_fourier_index)
from hiten.algorithms.fourier.operations import _make_fourier_poly

TEST_MAX_DEG = 3  # action‐degree limit for tests
TEST_K_MAX = 2   # Fourier index limit for tests (-2 ... +2)

# Build lookup tables once and reuse across tests
PSI_F, CLMO_F = _init_fourier_tables(TEST_MAX_DEG, TEST_K_MAX)
# Build encoding dictionary using the JIT-compiled helper (should succeed)
ENCODE_DICT_F = _create_encode_dict_fourier(CLMO_F)


def _random_valid_index(max_deg: int, k_max: int):
    d = random.randint(0, max_deg)
    # choose (n1,n2,n3) summing to d
    n1 = random.randint(0, d)
    n2 = random.randint(0, d - n1)
    n3 = d - n1 - n2
    k1 = random.randint(-k_max, k_max)
    k2 = random.randint(-k_max, k_max)
    k3 = random.randint(-k_max, k_max)
    return (n1, n2, n3, k1, k2, k3), d

def test_init_fourier_tables():
    # psiF should have length max_degree + 1
    assert PSI_F.shape == (TEST_MAX_DEG + 1,)

    # clmoF list length should match degrees and each array size equals psiF[d]
    assert len(CLMO_F) == TEST_MAX_DEG + 1

    num_fourier_terms = (2 * TEST_K_MAX + 1) ** 3
    for d in range(TEST_MAX_DEG + 1):
        # combinatorial count of action exponents: C(d+2, 2)
        count_actions = (d + 2) * (d + 1) // 2
        expected_size = count_actions * num_fourier_terms
        # Check psiF entry
        assert PSI_F[d] == expected_size
        # Check clmo array size
        assert len(CLMO_F[d]) == expected_size


def test_make_fourier_poly():
    for degree in range(TEST_MAX_DEG + 1):
        poly = _make_fourier_poly(degree, PSI_F)
        # correct length & dtype
        assert poly.shape[0] == PSI_F[degree]
        assert poly.dtype == np.complex128
        # all coefficients zero initially
        assert np.all(poly == 0.0)


def test_pack_unpack_roundtrip():
    for _ in range(50):  # 50 random samples
        idx_tuple, _ = _random_valid_index(TEST_MAX_DEG, TEST_K_MAX)
        packed = _pack_fourier_index(*idx_tuple)
        unpacked = _decode_fourier_index(packed)
        assert unpacked == idx_tuple


@pytest.mark.parametrize("deg", list(range(TEST_MAX_DEG + 1)))
def test_encode_decode_roundtrip(deg):
    size = PSI_F[deg]

    # Choose a representative subset of positions to keep runtime small
    if size <= 50:
        test_positions = range(size)
    else:
        test_positions = list(range(20))  # first few
        test_positions += list(range(size - 20, size))  # last few
        center = size // 2
        test_positions += list(range(center - 5, center + 5))  # middle slice

    for pos in test_positions:
        packed_key = CLMO_F[deg][pos]
        idx_tuple = _decode_fourier_index(packed_key)
        idx_back = _encode_fourier_index(idx_tuple, deg, ENCODE_DICT_F)
        assert idx_back == pos
        # Re-decode to verify deterministic mapping
        idx_tuple_again = _decode_fourier_index(packed_key)
        assert idx_tuple_again == idx_tuple


def test_encode_invalid_conditions():
    # Invalid action exponent (n1 exceeds _MAX_N)
    bad_tuple = (_MAX_N + 1, 0, 0, 0, 0, 0)
    assert _encode_fourier_index(bad_tuple, 0, ENCODE_DICT_F) == -1

    # Invalid Fourier exponent (k1 beyond _MAX_K)
    bad_tuple = (0, 0, 0, _MAX_K + 1, 0, 0)
    assert _encode_fourier_index(bad_tuple, 0, ENCODE_DICT_F) == -1

    # Degree outside encode table range
    good_tuple = (0, 0, 0, 0, 0, 0)
    assert _encode_fourier_index(good_tuple, TEST_MAX_DEG + 5, ENCODE_DICT_F) == -1
