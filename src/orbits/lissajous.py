import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple, Any
from scipy.integrate import solve_ivp


from orbits.base import PeriodicOrbit, orbitConfig
from system.libration import CollinearPoint, L1Point, L2Point, L3Point
from algorithms.geometry import _gamma_L, _find_y_zero_crossing
from algorithms.dynamics import compute_stm
from log_config import logger


class LissajousOrbit(PeriodicOrbit):
    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.Az = config.extra_params['Az']
        super().__init__(config, initial_state)

        if not isinstance(self.libration_point, CollinearPoint):
            msg = f"Expected CollinearPoint, got {type(self.libration_point)}."
            logger.error(msg)
            raise TypeError(msg)

        if isinstance(self.libration_point, L3Point):
            msg = "L3 libration points are not supported for Lissajous orbits."
            logger.error(msg)
            raise NotImplementedError(msg)

    def _initial_guess(self):
        pass

    def differential_correction(self):
        pass

    def eccentricity(self):
        pass

