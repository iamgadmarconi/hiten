import numpy as np
import math
from numba import njit
from numba.typed import List

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


@njit(fastmath=True, cache=True)
def make_poly(degree: int, psi) -> np.ndarray:
    size = psi[N_VARS, degree]
    return np.zeros(size, dtype=np.float64)


@njit(fastmath=True, cache=True)
def make_poly_complex(degree: int, psi) -> np.ndarray:
    size = psi[N_VARS, degree]
    return np.zeros(size, dtype=np.complex128)

