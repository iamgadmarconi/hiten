import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple, Any
from scipy.integrate import solve_ivp


from orbits.base import PeriodicOrbit, orbitConfig
from system.libration import CollinearPoint, L1Point, L2Point, L3Point
from algorithms.geometry import _gamma_L, _find_y_zero_crossing
from algorithms.dynamics import compute_stm
from log_config import logger


class VerticalLyapunovOrbit(PeriodicOrbit):
