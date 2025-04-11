import numba
import numpy as np
import mpmath as mp


from log_config import logger
from algorithms.geometry import _gamma_L


@numba.njit(fastmath=True, cache=True)
def hamiltonian_accel(state, mu):
    x, y, z, vx, vy, vz = state


