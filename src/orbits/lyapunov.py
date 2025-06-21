from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from algorithms.geometry import _find_y_zero_crossing, _find_z_zero_crossing
from orbits.base import PeriodicOrbit, S, correctionConfig, orbitConfig
from system.libration import CollinearPoint, L3Point
from utils.log_config import logger


class LyapunovOrbit(PeriodicOrbit):
    Ax: float # Amplitude of the Lyapunov orbit

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.Ax = config.extra_params['Ax']
        
        self._initial_state = None
        if initial_state is not None:
            self._initial_state = np.array(initial_state, dtype=np.float64)
            
        super().__init__(config, initial_state)

        if not isinstance(self.libration_point, CollinearPoint):
            msg = f"Expected CollinearPoint, got {type(self.libration_point)}."
            logger.error(msg)
            raise TypeError(msg)

        if isinstance(self.libration_point, L3Point):
            msg = "L3 libration points are not supported for Lyapunov orbits."
            logger.error(msg)
            raise NotImplementedError(msg)

    def _initial_guess(self) -> NDArray[np.float64]:

        if self._initial_state is not None:
            logger.info(f"Using provided initial state: {self._initial_state} for {str(self)}")
            return self._initial_state

        L_i = self.libration_point.position
        mu = self.mu
        x_L_i: float = L_i[0]
        # Note: This mu_bar is often denoted c2 or \omega_p^2 in literature
        mu_bar: float = mu * np.abs(x_L_i - 1 + mu) ** (-3) + (1 - mu) * np.abs(x_L_i + mu) ** (-3)

        if mu_bar < 0:
            msg = f"Error in linearization: mu_bar ({mu_bar}) is negative for {self.libration_point.name}"
            logger.error(msg)
            raise ValueError(msg)

        # alpha_2 relates to the square of the in-plane frequency (lambda^2 in Szebehely)
        alpha_2_complex: complex = (mu_bar - 2 - np.emath.sqrt(9*mu_bar**2 - 8*mu_bar + 0j)) / 2
        
        # Eigenvalue related to planar motion (often denoted lambda or omega_p in literature)
        eig2_complex: complex = np.emath.sqrt(-alpha_2_complex + 0j)
        
        if np.imag(eig2_complex) != 0:
             logger.warning(f"In-plane eigenvalue lambda ({eig2_complex:.4f}) is complex for {self.libration_point.name}. Linear guess might be less accurate.")

        nu_1: float = np.real(eig2_complex) # Planar frequency

        a: float = 2 * mu_bar + 1 # Intermediate calculation constant

        tau: float = - (nu_1 **2 + a) / (2*nu_1) # Relates x and vy components in linear approx

        # Linear approximation eigenvector components (excluding z-components)
        # [delta_x, delta_y, delta_vx, delta_vy]
        u = np.array([1, 0, 0, nu_1 * tau]) 

        displacement = self.Ax * u
        state_4d = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement
        # Construct 6D state [x, y, z, vx, vy, vz]
        state_6d = np.array([state_4d[0], state_4d[1], 0, state_4d[2], state_4d[3], 0], dtype=np.float64)
        logger.debug(f"Generated initial guess for Lyapunov orbit around {self.libration_point} with Ax={self.Ax}: {state_6d}")
        return state_6d

    def differential_correction(self, **kw):
        cfg = correctionConfig(
            residual_indices=(S.VX, S.Z),
            control_indices=(S.VZ, S.VY),
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_find_y_zero_crossing,
        )
        return super().differential_correction(cfg, **kw)

    def eccentricity(self) -> float:
        """Eccentricity is not typically defined for Lyapunov orbits.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Eccentricity is not implemented for Lyapunov orbits.")


class VerticalLyapunovOrbit(PeriodicOrbit):

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        super().__init__(config, initial_state)

    def _initial_guess(self) -> NDArray[np.float64]:
        raise NotImplementedError("Initial guess is not implemented for Vertical Lyapunov orbits.")

    @staticmethod
    def _extra_jacobian(_: np.ndarray, __: np.ndarray) -> np.ndarray:
        pass

    def differential_correction(
        self,
        *,
        tol: float = 1e-10,
        max_attempts: int = 25,
        forward: int = 1,
    ):
        cfg = correctionConfig(
            residual_indices=(S.VX, S.Y),     # Want VX=0 and Y=0
            control_indices=(S.VZ, S.VY),     # Adjust initial VZ and VY
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_find_z_zero_crossing,
        )

        return super().differential_correction(
            cfg, tol=tol, max_attempts=max_attempts, forward=forward
        )

    def eccentricity(self) -> float:
        """Eccentricity is not defined for Vertical Lyapunov orbits."""
        return np.nan