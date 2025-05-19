import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def scale_series(series: np.ndarray, factor: float) -> np.ndarray:
    """Return *factor · series* (copy)."""
    return series * factor

@njit(fastmath=True, cache=True)
def _omega0(c2: float) -> float:
    return np.sqrt((2-c2+np.sqrt(9*c2**2-8*c2))/2)

@njit(fastmath=True, cache=True)
def _nu0(c2: float) -> float:
    return np.sqrt(c2)

@njit(fastmath=True, cache=True)
def _kappa(c2: float, omega0: float) -> float:
    """Compute kappa analytically as in the image."""
    return -((omega0**2 + 1 + 2 * c2) / (2 * omega0))

@njit(fastmath=True, cache=True)
def _compute_denominator(k: int, m: int, omega0: float, nu0: float) -> float:
    return -((k * omega0 + m * nu0) ** 2)

@njit(fastmath=True, cache=True)
def _k_to_idx(i: int, k: int) -> int:
    return (k + i) // 2

@njit(fastmath=True, cache=True)
def _m_to_idx(j: int, m: int) -> int:
    return (m + j) // 2

@njit(fastmath=True, cache=True)
def linear_index(i: int, j: int, k_idx: int, m_idx: int, offset_ij: np.ndarray) -> int:
    return offset_ij[i, j] + k_idx * (j + 1) + m_idx

@njit(fastmath=True, cache=True)
def convolve_series(
    A: np.ndarray,
    B: np.ndarray,
    n: int,
    offset_ij: np.ndarray,
    mode_list: list,
) -> np.ndarray:
    out = np.zeros_like(A)

    for n1 in range(0, n + 1):
        list1 = mode_list[n1]
        for idx1_mode in range(list1.shape[0]):
            i1, j1, k_idx1, m_idx1, k1, m1 = list1[idx1_mode]
            idx1 = linear_index(i1, j1, k_idx1, m_idx1, offset_ij)
            a_val = A[idx1]
            if a_val == 0.0:
                continue

            max_n2 = n - (i1 + j1)
            for n2 in range(0, max_n2 + 1):
                list2 = mode_list[n2]
                for idx2_mode in range(list2.shape[0]):
                    i2, j2, k_idx2, m_idx2, k2, m2 = list2[idx2_mode]
                    idx2 = linear_index(i2, j2, k_idx2, m_idx2, offset_ij)
                    b_val = B[idx2]
                    if b_val == 0.0:
                        continue

                    i = i1 + i2
                    j = j1 + j2
                    if i + j > n:
                        continue

                    k = k1 + k2
                    m = m1 + m2

                    if ((k + i) & 1) or ((m + j) & 1):
                        continue
                    if abs(k) > i or abs(m) > j:
                        continue

                    k_out_idx = _k_to_idx(i, k)
                    m_out_idx = _m_to_idx(j, m)
                    idx_out = linear_index(i, j, k_out_idx, m_out_idx, offset_ij)
                    out[idx_out] += a_val * b_val
    return out

