"""Low-level helpers for Fourier-Taylor coefficient indexing.

Provide numba-compatible utilities to pack and unpack exponents and
Fourier indices into 64-bit keys, and to build lookup tables required
by the algebra and evaluation routines in :mod:`hiten.algorithms.fourier`.

Notes
-----
- Use ASCII-only text and plain-text math. Example: use ``sqrt(a^2+b^2)``.
- The packed key layout dedicates 6 bits to each action exponent and 7 bits
  to each Fourier index after a +64 shift. See inline comments below.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from hiten.algorithms.utils.config import FASTMATH

#  6 bits for each action exponent (0 ... 63)
#  7 bits for each Fourier index shifted by +64  (-64 ... +63)
#
#  ┌─────────┬────────┬────────┬────────┬────────┬────────┬────────┐
#  │ bits    │ 0-5    │ 6-11   │ 12-17  │ 18-24  │ 25-31  │ 32-38  │
#  │ field   │ n1     │ n2     │ n3     │ k1     │ k2     │ k3     │
#  └─────────┴────────┴────────┴────────┴────────┴────────┴────────┘
#  (remaining higher bits unused for now)

_N_MASK = 0x3F              # 6 bits
_K_MASK = 0x7F              # 7 bits
_K_OFFSET = 64              # shift applied to store signed kᵢ as unsigned

# upper bounds hard-wired by bit-width
_MAX_N = _N_MASK
_MAX_K = _K_OFFSET - 1


@njit(fastmath=FASTMATH, cache=False)
def _pack_fourier_index(n1: int, n2: int, n3: int, k1: int, k2: int, k3: int) -> np.uint64:  
    """Pack exponents into a 64-bit key for constant-time lookup.

    Parameters
    ----------
    n1, n2, n3 : int
        Non-negative action exponents, each in [0, _MAX_N].
    k1, k2, k3 : int
        Fourier indices in [-_K_OFFSET, _MAX_K].

    Returns
    -------
    numpy.uint64
        Packed key, or ``0xFFFFFFFFFFFFFFFF`` as an invalid sentinel.
    """

    if (n1 < 0 or n1 > _MAX_N or
        n2 < 0 or n2 > _MAX_N or
        n3 < 0 or n3 > _MAX_N):
        return np.uint64(0xFFFFFFFFFFFFFFFF)  # invalid sentinel

    if (k1 < -_K_OFFSET or k1 > _MAX_K or
        k2 < -_K_OFFSET or k2 > _MAX_K or
        k3 < -_K_OFFSET or k3 > _MAX_K):
        return np.uint64(0xFFFFFFFFFFFFFFFF)

    k1_enc = (k1 + _K_OFFSET) & _K_MASK
    k2_enc = (k2 + _K_OFFSET) & _K_MASK
    k3_enc = (k3 + _K_OFFSET) & _K_MASK

    packed = (
        (n1 & _N_MASK)
        | ((n2 & _N_MASK) << 6)
        | ((n3 & _N_MASK) << 12)
        | (k1_enc << 18)
        | (k2_enc << 25)
        | (k3_enc << 32)
    )
    return np.uint64(packed)


@njit(fastmath=FASTMATH, cache=False)
def _decode_fourier_index(key: np.uint64):  
    """Decode a packed key back into exponents and Fourier indices.

    Parameters
    ----------
    key : numpy.uint64
        Packed key produced by
        :func:`hiten.algorithms.fourier.base._pack_fourier_index`.

    Returns
    -------
    tuple
        ``(n1, n2, n3, k1, k2, k3)`` integers.
    """
    key_int = int(key)

    n1 = key_int & _N_MASK
    n2 = (key_int >> 6) & _N_MASK
    n3 = (key_int >> 12) & _N_MASK

    k1 = ((key_int >> 18) & _K_MASK) - _K_OFFSET
    k2 = ((key_int >> 25) & _K_MASK) - _K_OFFSET
    k3 = ((key_int >> 32) & _K_MASK) - _K_OFFSET

    return n1, n2, n3, k1, k2, k3


@njit(fastmath=FASTMATH, cache=False)
def _init_fourier_tables(degree: int, k_max: int):  
    """Build lookup tables for Fourier-Taylor monomials up to a degree.

    Parameters
    ----------
    degree : int
        Maximum total action degree d = n1 + n2 + n3 to include.
    k_max : int
        Limit Fourier indices to the range [-k_max, +k_max] (k_max <= 63).

    Returns
    -------
    psiF : numpy.ndarray
        Array of length ``degree + 1`` where ``psiF[d]`` gives the number of
        monomials with total action degree ``d``.
    clmoF : numba.typed.List
        For each degree ``d``, an array of packed indices (dtype ``uint64``)
        of length ``psiF[d]``.
    """
    if k_max > _MAX_K:
        k_max = _MAX_K  # silently truncate to hard limit

    num_fourier = 2 * k_max + 1  # count per angle dimension
    num_fourier_cubed = num_fourier * num_fourier * num_fourier

    psiF = np.zeros(degree + 1, dtype=np.int64)
    clmoF = List.empty_list(np.uint64[::1])

    for d in range(degree + 1):
        # number of (n1,n2,n3) with sum d = C(d+2,2)
        count_actions = (d + 2) * (d + 1) // 2
        count_terms = count_actions * num_fourier_cubed
        psiF[d] = count_terms

        arr = np.empty(count_terms, dtype=np.uint64)
        idx = 0

        # enumerate all non-negative integer triples summing to d
        for n1 in range(d, -1, -1):
            for n2 in range(d - n1, -1, -1):
                n3 = d - n1 - n2

                # enumerate Fourier indices
                for k1 in range(-k_max, k_max + 1):
                    for k2 in range(-k_max, k_max + 1):
                        for k3 in range(-k_max, k_max + 1):
                            arr[idx] = _pack_fourier_index(n1, n2, n3, k1, k2, k3)
                            idx += 1
        clmoF.append(arr)

    return psiF, clmoF


@njit(fastmath=FASTMATH, cache=False)
def _create_encode_dict_fourier(clmoF: List):  
    """Create index-to-position maps for each degree array in ``clmoF``.

    Parameters
    ----------
    clmoF : numba.typed.List
        Packed index arrays per degree.

    Returns
    -------
    numba.typed.List
        For each degree, a ``Dict[int64, int32]`` mapping packed key to array
        position to support constant-time lookup.
    """
    encode_list = List()
    for arr in clmoF:
        d_map = Dict.empty(key_type=types.int64, value_type=types.int32)
        for pos, key in enumerate(arr):
            d_map[np.int64(key)] = np.int32(pos)
        encode_list.append(d_map)
    return encode_list


@njit(fastmath=FASTMATH, cache=False)
def _encode_fourier_index(idx_tuple, degree: int, encode_dict_list):  
    """Return position of an index tuple within a degree block.

    Parameters
    ----------
    idx_tuple : tuple
        ``(n1, n2, n3, k1, k2, k3)``.
    degree : int
        Target degree block.
    encode_dict_list : numba.typed.List
        List of maps returned by
        :func:`hiten.algorithms.fourier.base._create_encode_dict_fourier`.

    Returns
    -------
    int
        Zero-based position if present; -1 if invalid or not found.
    """
    n1, n2, n3, k1, k2, k3 = idx_tuple
    key = _pack_fourier_index(n1, n2, n3, k1, k2, k3)
    if key == np.uint64(0xFFFFFFFFFFFFFFFF):
        return -1
    if degree < 0 or degree >= len(encode_dict_list):
        return -1
    d_map = encode_dict_list[degree]
    key_int = np.int64(key)
    if key_int in d_map:
        return d_map[key_int]
    return -1