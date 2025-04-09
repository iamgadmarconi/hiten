import numpy as np
from typing import Optional, Sequence

from system.libration import CollinearPoint
from base import PeriodicOrbit, orbitConfig

from log_config import logger


class LyapunovOrbit(PeriodicOrbit):
    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        super().__init__(config, initial_state)

        self.Ax = config.extra_params['Ax']

        if not isinstance(self.libration_point, CollinearPoint):
            msg = f"Expected CollinearPoint, got {type(self.libration_point)}."
            logger.error(msg)
            raise TypeError(msg)

    def _initial_guess(self) -> np.ndarray:
        L_i = self.libration_point.position
        mu = self._system.mu
        x_L_i = L_i[0]
        mu_bar = mu * np.abs(x_L_i - 1 + mu) ** (-3) + (1 - mu) * np.abs(x_L_i + mu) ** (-3)

        if mu_bar < 0:
            msg = "Error in linearization: mu_bar is negative"
            logger.error(msg)
            raise ValueError(msg)

        alpha_2 = (mu_bar - 2 - np.emath.sqrt(9*mu_bar**2 - 8*mu_bar)) / 2
        if isinstance(alpha_2, np.complex128):
            logger.warning("Alpha 2 is complex")

        eig2 = np.emath.sqrt(-alpha_2)

        nu_1 = eig2

        a = 2 * mu_bar + 1

        tau = - (nu_1 **2 + a) / (2*nu_1)

        u = np.array([1, 0, 0, nu_1 * tau])

        displacement = self.Ax * u
        state = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement
        return np.array([state[0], state[1], 0, state[2], state[3], 0], dtype=np.float64)

    def differential_correction(self, **kwargs):
        pass
