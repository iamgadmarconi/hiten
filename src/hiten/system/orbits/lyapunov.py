r"""
hiten.system.orbits.lyapunov
======================

Periodic Lyapunov orbits of the circular restricted three-body problem.

This module supplies concrete realisations of :pyclass:`hiten.system.orbits.base.PeriodicOrbit`
corresponding to the planar families around the collinear libration points
:math:`L_1` and :math:`L_2`.  Each class provides an analytical
first guess together with a customised differential corrector that exploits the
symmetries of the family.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from hiten.algorithms.dynamics.utils.geometry import _find_y_zero_crossing
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import (CollinearPoint, L1Point, L2Point,
                                              L3Point)
from hiten.system.orbits.base import (PeriodicOrbit, S, _ContinuationConfig,
                                      _CorrectionConfig)
from hiten.utils.log_config import logger


class LyapunovOrbit(PeriodicOrbit):
    r"""
    Planar Lyapunov family around a collinear libration point.

    The orbit lies in the :math:`(x, y)` plane and is symmetric with respect to
    the :math:`x`-axis.  A linear analytical approximation is used to build the
    initial guess which is subsequently refined by a differential corrector.

    Parameters
    ----------
    libration_point : CollinearPoint
        Target :pyclass:`hiten.system.libration.collinear.CollinearPoint` around
        which the orbit is computed.
    amplitude_x : float, optional
        Requested amplitude :math:`A_x` along the :math:`x`-direction. Required if
        *initial_state* is None.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector
        :math:`(x, y, z, \\dot x, \\dot y, \\dot z)` expressed in synodic
        coordinates.  If *None*, an analytical guess is generated.

    Attributes
    ----------
    amplitude_x : float
        Requested amplitude :math:`A_x` along the :math:`x`-direction.
    libration_point : hiten.system.libration.collinear.CollinearPoint
        Equilibrium point about which the orbit is continued.

    Raises
    ------
    TypeError
        If *libration_point* is not an instance of
        :pyclass:`hiten.system.libration.collinear.CollinearPoint`.
    NotImplementedError
        If the selected point corresponds to :math:`L_3`, which is not
        supported for Lyapunov orbits.
    """
    
    _family = "lyapunov"
    
    amplitude_x: float # Amplitude of the Lyapunov orbit

    def __init__(
            self, 
            libration_point: LibrationPoint, 
            amplitude_x: Optional[float] = None,
            initial_state: Optional[Sequence[float]] = None
        ):
        
        # Validate constructor parameters
        if initial_state is not None and amplitude_x is not None:
            raise ValueError("Cannot provide both an initial_state and an analytical parameter (amplitude_x).")

        if not isinstance(libration_point, CollinearPoint):
            msg = f"Lyapunov orbits are only defined for CollinearPoint, but got {type(libration_point)}."
            logger.error(msg)
            raise TypeError(msg)
            
        if initial_state is None:
            if amplitude_x is None:
                err = "Lyapunov orbits require an 'amplitude_x' (x-amplitude) parameter when an initial_state is not provided."
                logger.error(err)
                raise ValueError(err)
            if not isinstance(libration_point, (L1Point, L2Point)):
                err = f"Analytical guess is only available for L1/L2 points. An initial_state must be provided for {libration_point.name}."
                logger.error(err)
                raise ValueError(err)
        
        # Preserve user-supplied amplitude (may be None) for initial guess
        self._amplitude_x = amplitude_x
        
        if isinstance(libration_point, L3Point):
            msg = "L3 libration points are not supported for Lyapunov orbits."
            logger.error(msg)
            raise NotImplementedError(msg)

        # The base class __init__ handles the logic for initial_state vs. _initial_guess
        super().__init__(libration_point, initial_state)

        # Ensure amplitude_x is consistent with the state if it was provided directly.
        if initial_state is not None and self._amplitude_x is None:
            # Infer amplitude from state so _initial_guess works in property logic
            self._amplitude_x = self._initial_state[S.X] - self.libration_point.position[0]

    @property
    def eccentricity(self) -> float:
        """Eccentricity is not a well-defined concept for Lyapunov orbits."""
        return np.nan

    @property
    def amplitude(self) -> float:
        """(Read-only) Current x-amplitude relative to the libration point."""
        if getattr(self, "_initial_state", None) is not None:
            return float(self._initial_state[S.X] - self.libration_point.position[0])
        return float(self._amplitude_x)

    @property
    def _correction_config(self) -> _CorrectionConfig:
        """Provides the differential correction configuration for planar Lyapunov orbits."""
        return _CorrectionConfig(
            residual_indices=(S.VX, S.Z),
            control_indices=(S.VY, S.VZ),
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_find_y_zero_crossing,
        )

    @property
    def _continuation_config(self) -> _ContinuationConfig:
        return _ContinuationConfig(state=S.X, amplitude=True)

    def _initial_guess(self) -> NDArray[np.float64]:
        r"""
        Return an analytical first guess for the planar Lyapunov orbit.

        The guess is derived from the linearised equations of motion around the
        collinear point.  Given the user-supplied amplitude :math:`A_x`, the
        displacement vector is built as

        :math:`\Delta\mathbf x = A_x\,(1, 0, 0, \lambda\tau, 0, 0),`

        where :math:`\lambda` is the in-plane eigenvalue and :math:`\tau` is a
        constant that relates the position and velocity components in the
        linear approximation.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(6,)`` containing the synodic state vector.

        Raises
        ------
        ValueError
            If the auxiliary quantity *mu_bar* computed during the linear
            analysis becomes negative, indicating an invalid parameter regime.
        """
        L_i = self.libration_point.position
        mu = self.mu
        x_L_i: float = L_i[0]
        # Note: This mu_bar is often denoted c2 or \\omega_p^2 in literature
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

        displacement = self._amplitude_x * u
        state_4d = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement
        # Construct 6D state [x, y, z, vx, vy, vz]
        state_6d = np.array([state_4d[0], state_4d[1], 0, state_4d[2], state_4d[3], 0], dtype=np.float64)
        logger.debug(f"Generated initial guess for Lyapunov orbit around {self.libration_point} with amplitude_x={self.amplitude}: {state_6d}")
        return state_6d
