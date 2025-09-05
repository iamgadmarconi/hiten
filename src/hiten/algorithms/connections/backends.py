from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numba import njit


class _BackendBase(ABC):
    pass


@njit(cache=False)
def _pair_counts(query: np.ndarray, ref: np.ndarray, r2: float) -> np.ndarray:
    n_q = query.shape[0]
    n_r = ref.shape[0]
    counts = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        x = query[i, 0]
        y = query[i, 1]
        c = 0
        for j in range(n_r):
            dx = x - ref[j, 0]
            dy = y - ref[j, 1]
            if dx * dx + dy * dy <= r2:
                c += 1
        counts[i] = c
    return counts


@njit(cache=False)
def _exclusive_prefix_sum(a: np.ndarray) -> np.ndarray:
    n = a.size
    out = np.empty(n + 1, dtype=np.int64)
    out[0] = 0
    s = 0
    for i in range(n):
        s += int(a[i])
        out[i + 1] = s
    return out


@njit(cache=False)
def _radpair2d(query: np.ndarray, ref: np.ndarray, radius: float) -> np.ndarray:
    r2 = float(radius) * float(radius)
    counts = _pair_counts(query, ref, r2)
    offs = _exclusive_prefix_sum(counts)
    total = int(offs[-1])
    pairs = np.empty((total, 2), dtype=np.int64)

    n_q = query.shape[0]
    n_r = ref.shape[0]
    for i in range(n_q):
        write = offs[i]
        x = query[i, 0]
        y = query[i, 1]
        for j in range(n_r):
            dx = x - ref[j, 0]
            dy = y - ref[j, 1]
            if dx * dx + dy * dy <= r2:
                pairs[write, 0] = i
                pairs[write, 1] = j
                write += 1
    return pairs


def _radius_pairs_2d(query: np.ndarray, ref: np.ndarray, radius: float) -> np.ndarray:
    """Return pairs (i,j) where ||query[i]-ref[j]|| <= radius on a 2D plane.

    Parameters
    ----------
    query, ref : (N,2) and (M,2) float arrays
        2D plane coordinates.
    radius : float
        Match radius.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(ref, dtype=np.float64)
    return _radpair2d(q, r, float(radius))