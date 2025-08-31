import numpy as np
from numba import njit

from hiten.algorithms.utils.config import FASTMATH


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _interp_linear(t0: float, x0: np.ndarray, t1: float, x1: np.ndarray, t: float) -> np.ndarray:
    s = (t - t0) / (t1 - t0)
    return (1.0 - s) * x0 + s * x1


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _hermite_scalar(s: float, y0: float, y1: float, dy0: float, dy1: float, dt: float) -> float:
    h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
    h10 = s * (1.0 - s) ** 2
    h01 = s ** 2 * (3.0 - 2.0 * s)
    h11 = s ** 2 * (s - 1.0)
    return h00 * y0 + h10 * dy0 * dt + h01 * y1 + h11 * dy1 * dt


@njit(cache=False, fastmath=FASTMATH, inline="always")
def _hermite_der(s: float, y0: float, y1: float, dy0: float, dy1: float, dt_seg: float) -> float:
    # Analytical derivative of the cubic Hermite polynomial
    dh00 = 6.0 * s * (s - 1.0) + (1.0 - s) ** 2 * 2.0 - 2.0 * (1.0 - s) * (1.0 + 2.0 * s)
    dh10 = (1.0 - s) ** 2 + s * (2.0 * (s - 1.0))
    dh01 = 6.0 * s * (1.0 - s) - 2.0 * s * (3.0 - 2.0 * s)
    dh11 = 2.0 * s * (s - 1.0) + s ** 2
    return dh00 * y0 + dh10 * dy0 * dt_seg + dh01 * y1 + dh11 * dy1 * dt_seg
