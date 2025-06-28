"""
hiten.algorithms.polynomial.fourier
===================================

Low-level helpers for working with *Fourier-Taylor* coefficient arrays
in action-angle variables
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict, List

from hiten.algorithms.utils.config import FASTMATH

#  6 bits for each action exponent (0 … 63)
#  7 bits for each Fourier index shifted by +64  (-64 … +63)
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
    """Pack exponents into a 64-bit key for constant-time lookup."""

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
    """Inverse of :pyfunc:`_pack_fourier_index`."""
    key_int = int(key)

    n1 = key_int & _N_MASK
    n2 = (key_int >> 6) & _N_MASK
    n3 = (key_int >> 12) & _N_MASK

    k1 = ((key_int >> 18) & _K_MASK) - _K_OFFSET
    k2 = ((key_int >> 25) & _K_MASK) - _K_OFFSET
    k3 = ((key_int >> 32) & _K_MASK) - _K_OFFSET

    return n1, n2, n3, k1, k2, k3


@njit(fastmath=FASTMATH, cache=False)
def _init_fourier_tables(max_degree: int, k_max: int):  
    """
    Build *psiF* and *clmoF* lookup tables for Fourier polynomials.

    Parameters
    ----------
    max_degree : int
        Maximum total action degree *d = n₁+n₂+n₃* to include.
    k_max : int
        Fourier indices kᵢ will be limited to -k_max … +k_max (k_max ≤ 63).

    Returns
    -------
    psiF : numpy.ndarray  (shape ``(max_degree+1,)``)
        psiF[d] = number of terms with total action degree *d*.
    clmoF : numba.typed.List
        For each degree *d*, an array of packed indices (dtype uint64) of size psiF[d].
    """
    if k_max > _MAX_K:
        k_max = _MAX_K  # silently truncate to hard limit

    num_fourier = 2 * k_max + 1  # count per angle dimension
    num_fourier_cubed = num_fourier * num_fourier * num_fourier

    psiF = np.zeros(max_degree + 1, dtype=np.int64)
    clmoF = List.empty_list(np.uint64[::1])

    for d in range(max_degree + 1):
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
    """Create a list of dictionaries mapping packed index -> position for each degree."""
    encode_list = List()
    for arr in clmoF:
        d_map = Dict.empty(key_type=types.int64, value_type=types.int32)
        for pos, key in enumerate(arr):
            d_map[np.int64(key)] = np.int32(pos)
        encode_list.append(d_map)
    return encode_list


@njit(fastmath=FASTMATH, cache=False)
def _encode_fourier_index(idx_tuple, degree: int, encode_dict_list):  
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


@njit(fastmath=FASTMATH, cache=False)
def _make_fourier_poly(degree: int, psiF: np.ndarray):  
    size = psiF[degree]
    return np.zeros(size, dtype=np.complex128)


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_add(p: np.ndarray, q: np.ndarray, out: np.ndarray) -> None:
    for i in range(p.shape[0]):
        out[i] = p[i] + q[i]


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_scale(p: np.ndarray, alpha, out: np.ndarray) -> None:
    for i in range(p.shape[0]):
        out[i] = alpha * p[i]


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_mul(
    p: np.ndarray,
    deg_p: int,
    q: np.ndarray,
    deg_q: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    deg_r = deg_p + deg_q
    out_len = psiF[deg_r]
    out = np.zeros(out_len, dtype=np.complex128)

    for i in range(p.shape[0]):
        ci = p[i]
        if ci == 0.0:
            continue
        n1a, n2a, n3a, k1a, k2a, k3a = _decode_fourier_index(clmoF[deg_p][i])

        for j in range(q.shape[0]):
            cj = q[j]
            if cj == 0.0:
                continue
            n1b, n2b, n3b, k1b, k2b, k3b = _decode_fourier_index(clmoF[deg_q][j])

            # combined exponents / indices
            n1c = n1a + n1b
            n2c = n2a + n2b
            n3c = n3a + n3b
            k1c = k1a + k1b
            k2c = k2a + k2b
            k3c = k3a + k3b

            idx_tuple = (n1c, n2c, n3c, k1c, k2c, k3c)
            pos = _encode_fourier_index(idx_tuple, deg_r, encodeF)
            if pos != -1:
                out[pos] += ci * cj
    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_diff_action(
    p: np.ndarray,
    deg_p: int,
    action_idx: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    if deg_p == 0:
        return np.zeros_like(p)

    out_deg = deg_p - 1
    out = np.zeros(psiF[out_deg], dtype=np.complex128)

    for i in range(p.shape[0]):
        coeff = p[i]
        if coeff == 0.0:
            continue
        n1, n2, n3, k1, k2, k3 = _decode_fourier_index(clmoF[deg_p][i])
        n = (n1, n2, n3)
        exp_val = n[action_idx]
        if exp_val == 0:
            continue
        n_list = [n1, n2, n3]
        n_list[action_idx] = exp_val - 1
        idx_tuple = (n_list[0], n_list[1], n_list[2], k1, k2, k3)
        pos = _encode_fourier_index(idx_tuple, out_deg, encodeF)
        if pos != -1:
            out[pos] += coeff * exp_val
    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_diff_angle(
    p: np.ndarray,
    deg_p: int,
    angle_idx: int,
    clmoF,
) -> np.ndarray:
    out = np.zeros_like(p)

    for i in range(p.shape[0]):
        coeff = p[i]
        if coeff == 0.0:
            continue

        _n1, _n2, _n3, k1, k2, k3 = _decode_fourier_index(clmoF[deg_p][i])

        k_tuple = (k1, k2, k3)
        k_val = k_tuple[angle_idx]
        if k_val == 0:
            continue

        out[i] = 1j * k_val * coeff

    return out

@njit(fastmath=FASTMATH, cache=False)
def _fpoly_poisson(
    p: np.ndarray,
    deg_p: int,
    q: np.ndarray,
    deg_q: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    if deg_p == 0 and deg_q == 0:
        return np.zeros(psiF[0], dtype=np.complex128)

    deg_r = deg_p + deg_q - 1  # derivative w.r.t I_j lowers deg by 1
    if deg_r >= psiF.shape[0]:
        # allocate on the fly if beyond current table (rare)
        return np.zeros(1, dtype=np.complex128)

    out = np.zeros(psiF[deg_r], dtype=np.complex128)

    for j in range(3):  # loop over 3 action-angle pairs
        dp_dtheta = _fpoly_diff_angle(p, deg_p, j, clmoF)
        dq_dI     = _fpoly_diff_action(q, deg_q, j, psiF, clmoF, encodeF)

        # Skip invalid combinations that would require negative degrees
        if deg_q > 0:
            term1 = _fpoly_mul(dp_dtheta, deg_p, dq_dI, deg_q - 1, psiF, clmoF, encodeF)
        else:
            term1 = np.zeros_like(out)

        dp_dI   = _fpoly_diff_action(p, deg_p, j, psiF, clmoF, encodeF)
        dq_dtheta = _fpoly_diff_angle(q, deg_q, j, clmoF)

        if deg_p > 0:
            term2 = _fpoly_mul(dp_dI, deg_p - 1, dq_dtheta, deg_q, psiF, clmoF, encodeF)
        else:
            term2 = np.zeros_like(out)

        # Addition/subtraction into out
        for idx in range(out.shape[0]):
            out[idx] += term1[idx] - term2[idx]

    return out
