r"""
hiten.system.orbits.halo
==================

Generation and refinement of halo periodic orbits about the collinear
libration points of the Circular Restricted Three-Body Problem (CRTBP).

The module provides the :pyclass:`HaloOrbit` class which

* synthesises an initial state from the third-order analytic expansion of
  Richardson (1980), yielding a fast and robust first guess for the full
  nonlinear halo orbit;
* refines this guess through a differential-correction procedure that
  enforces the periodicity conditions by solving a reduced :math:`2\times2` boundary
  value problem.

References
----------
Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
collinear libration points".
"""

from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import (CollinearPoint, L1Point, L2Point,
                                              L3Point)
from hiten.system.orbits.base import (PeriodicOrbit, S, _ContinuationConfig,
                                      _CorrectionConfig)
from hiten.utils.log_config import logger


class HaloOrbit(PeriodicOrbit):
    r"""
    Halo orbit class.

    Parameters
    ----------
    libration_point : CollinearPoint
        Target :pyclass:`hiten.system.libration.collinear.CollinearPoint` around
        which the halo orbit is computed.
    amplitude_z : float, optional
        :math:`z`-amplitude of the halo orbit in the synodic frame. Required if
        *initial_state* is None.
    zenith : {'northern', 'southern'}, optional
        Indicates the symmetry branch with respect to the :math:`x\,y`-plane.
        Required if *initial_state* is None.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector
        :math:`[x,\,y,\,z,\,\dot{x},\,\dot{y},\,\dot{z}]` in the rotating
        synodic frame. When *None* an analytical initial guess is generated
        from *amplitude_z* and *zenith*.

    Attributes
    ----------
    amplitude_z : float or None
        :math:`z`-amplitude of the halo orbit in the synodic frame.
    zenith : {'northern', 'southern'} or None
        Indicates the symmetry branch with respect to the :math:`x\,y`-plane.

    Raises
    ------
    ValueError
        If the required amplitude or branch is missing and *initial_state*
        is *None*.
    TypeError
        If *libration_point* is not an instance of
        :pyclass:`CollinearPoint`.
    """
    
    _family = "halo"
    
    amplitude_z: Optional[float] # Amplitude of the halo orbit
    zenith: Optional[Literal["northern", "southern"]]

    def __init__(
            self, 
            libration_point: LibrationPoint, 
            amplitude_z: Optional[float] = None,
            zenith: Optional[Literal["northern", "southern"]] = None,
            initial_state: Optional[Sequence[float]] = None
        ):

        # Validate constructor parameters
        if initial_state is not None and (amplitude_z is not None or zenith is not None):
            raise ValueError("Cannot provide both an initial_state and analytical parameters (amplitude_z, zenith).")

        if not isinstance(libration_point, CollinearPoint):
            err = f"Halo orbits are only defined for CollinearPoint, but got {type(libration_point)}."
            logger.error(err)
            raise TypeError(err)
            
        if initial_state is None:
            if amplitude_z is None or zenith is None:
                err = "Halo orbits require an 'amplitude_z' (z-amplitude) and 'zenith' ('northern'/'southern') parameter when an initial_state is not provided."
                logger.error(err)
                raise ValueError(err)
            if not isinstance(libration_point, (L1Point, L2Point)):
                logger.warning(
                    "The analytical guess for L3 Halo orbits is experimental.\n "
                    "Convergence is not guaranteed and may require more iterations."
                )

        # Store user-supplied amplitude; will be replaced after correction
        self._amplitude_z = amplitude_z

        self.zenith = zenith

        super().__init__(libration_point, initial_state)

        if initial_state is not None:
            # Infer missing zenith
            if self.zenith is None:
                self.zenith = "northern" if self._initial_state[S.Z] > 0 else "southern"
            # Infer missing amplitude
            if self._amplitude_z is None:
                self._amplitude_z = self._initial_state[S.Z]

    @property
    def amplitude(self) -> float:
        """(Read-only) Current z-amplitude of the orbit in the synodic frame."""
        if getattr(self, "_initial_state", None) is not None:
            return float(self._initial_state[S.Z])
        return float(self._amplitude_z)

    @property
    def _correction_config(self) -> _CorrectionConfig:
        """Provides the differential correction configuration for halo orbits."""
        return _CorrectionConfig(
            residual_indices=(S.VX, S.VZ),
            control_indices=(S.X, S.VY),
            extra_jacobian=self._halo_quadratic_term
        )

    @property
    def _continuation_config(self) -> _ContinuationConfig:
        return _ContinuationConfig(state=S.Z, amplitude=True)

    def _initial_guess(self) -> NDArray[np.float64]:
        r"""
        Richardson third-order analytical approximation.

        The method evaluates the closed-form expressions published by
        Richardson to obtain an :math:`O(\!\epsilon^{3})` approximation of the halo
        orbit where :math:`\epsilon` is the amplitude ratio.

        Returns
        -------
        numpy.ndarray
            State vector of shape (6,) containing
            :math:`[x,\,y,\,z,\,\dot{x},\,\dot{y},\,\dot{z}]` in the synodic
            frame and normalised CRTBP units.

        Notes
        -----
        The computation follows [Richardson1980]_.

        Examples
        --------
        >>> L1 = L1Point(system)
        >>> orb = HaloOrbit(L1, amplitude_z=0.01, zenith='northern')
        >>> y0 = orb._initial_guess()
        """
        # Determine sign (won) and which "primary" to use

        mu = self.mu
        amplitude_z = self._amplitude_z
        # Get gamma from the libration point instance property
        gamma = self.libration_point.gamma
        
        point_map = {
            L1Point: (+1, 1 - mu),
            L2Point: (-1, 1 - mu),
            L3Point: (+1, -mu)
        }
        
        point_type = type(self.libration_point)
        if point_type in point_map:
            won, primary = point_map[point_type]
        else:
            # This case should ideally not be hit due to __init__ checks, but provides a safeguard.
            raise ValueError(f"Analytical guess for Halo orbits is not supported for {self.libration_point.name} (got {point_type.__name__})")
        
        # Set n for northern/southern family
        n = 1 if self.zenith == "northern" else -1
        
        # Coefficients c(2), c(3), c(4)
        c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
        
        if isinstance(self.libration_point, L3Point):
            for N in [2, 3, 4]:
                c[N] = (1 / gamma**3) * (
                    (1 - mu) + (-primary * gamma**(N + 1)) / ((1 + gamma)**(N + 1))
                )
        else:
            for N in [2, 3, 4]:
                c[N] = (1 / gamma**3) * (
                    (won**N) * mu 
                    + ((-1)**N)
                    * (primary * gamma**(N + 1) / ((1 + (-won) * gamma)**(N + 1)))
                )

        # Solve for lambda (the in-plane frequency)
        polylambda = [
            1,
            0,
            c[2] - 2,
            0,
            - (c[2] - 1) * (1 + 2 * c[2]),
        ]
        lambda_roots = np.roots(polylambda)

        # Pick the appropriate root based on L_i
        if isinstance(self.libration_point, L3Point):
            lam = abs(lambda_roots[2])  # third element in 0-based indexing
        else:
            lam = abs(lambda_roots[0])  # first element in 0-based indexing

        # Calculate parameters
        k = 2 * lam / (lam**2 + 1 - c[2])
        delta = lam**2 - c[2]

        d1 = (3 * lam**2 / k) * (k * (6 * lam**2 - 1) - 2 * lam)
        d2 = (8 * lam**2 / k) * (k * (11 * lam**2 - 1) - 2 * lam)

        a21 = (3 * c[3] * (k**2 - 2)) / (4 * (1 + 2 * c[2]))
        a22 = (3 * c[3]) / (4 * (1 + 2 * c[2]))
        a23 = - (3 * c[3] * lam / (4 * k * d1)) * (
            3 * k**3 * lam - 6 * k * (k - lam) + 4
        )
        a24 = - (3 * c[3] * lam / (4 * k * d1)) * (2 + 3 * k * lam)

        b21 = - (3 * c[3] * lam / (2 * d1)) * (3 * k * lam - 4)
        b22 = (3 * c[3] * lam) / d1

        d21 = - c[3] / (2 * lam**2)

        a31 = (
            - (9 * lam / (4 * d2)) 
            * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2)) 
            + ((9 * lam**2 + 1 - c[2]) / (2 * d2)) 
            * (
                3 * c[3] * (2 * a23 - k * b21) 
                + c[4] * (2 + 3 * k**2)
            )
        )
        a32 = (
            - (1 / d2)
            * (
                (9 * lam / 4) * (4 * c[3] * (k * a24 - b22) + k * c[4]) 
                + 1.5 * (9 * lam**2 + 1 - c[2]) 
                * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
            )
        )

        b31 = (
            0.375 / d2
            * (
                8 * lam 
                * (3 * c[3] * (k * b21 - 2 * a23) - c[4] * (2 + 3 * k**2))
                + (9 * lam**2 + 1 + 2 * c[2])
                * (4 * c[3] * (k * a23 - b21) + k * c[4] * (4 + k**2))
            )
        )
        b32 = (
            (1 / d2)
            * (
                9 * lam 
                * (c[3] * (k * b22 + d21 - 2 * a24) - c[4])
                + 0.375 * (9 * lam**2 + 1 + 2 * c[2])
                * (4 * c[3] * (k * a24 - b22) + k * c[4])
            )
        )

        d31 = (3 / (64 * lam**2)) * (4 * c[3] * a24 + c[4])
        d32 = (3 / (64 * lam**2)) * (4 * c[3] * (a23 - d21) + c[4] * (4 + k**2))

        s1 = (
            1 
            / (2 * lam * (lam * (1 + k**2) - 2 * k))
            * (
                1.5 * c[3] 
                * (
                    2 * a21 * (k**2 - 2) 
                    - a23 * (k**2 + 2) 
                    - 2 * k * b21
                )
                - 0.375 * c[4] * (3 * k**4 - 8 * k**2 + 8)
            )
        )
        s2 = (
            1 
            / (2 * lam * (lam * (1 + k**2) - 2 * k))
            * (
                1.5 * c[3] 
                * (
                    2 * a22 * (k**2 - 2) 
                    + a24 * (k**2 + 2) 
                    + 2 * k * b22 
                    + 5 * d21
                )
                + 0.375 * c[4] * (12 - k**2)
            )
        )

        a1 = -1.5 * c[3] * (2 * a21 + a23 + 5 * d21) - 0.375 * c[4] * (12 - k**2)
        a2 = 1.5 * c[3] * (a24 - 2 * a22) + 1.125 * c[4]

        l1 = a1 + 2 * lam**2 * s1
        l2 = a2 + 2 * lam**2 * s2

        deltan = -n  # matches the original code's sign usage

        # Solve for amplitude_x from the condition ( -del - l2*amplitude_z^2 ) / l1
        amplitude_x = np.sqrt((-delta - l2 * amplitude_z**2) / l1)

        # Evaluate the expansions at tau1 = 0
        tau1 = 0.0
        
        x = (
            a21 * amplitude_x**2 + a22 * amplitude_z**2
            - amplitude_x * np.cos(tau1)
            + (a23 * amplitude_x**2 - a24 * amplitude_z**2) * np.cos(2 * tau1)
            + (a31 * amplitude_x**3 - a32 * amplitude_x * amplitude_z**2) * np.cos(3 * tau1)
        )
        y = (
            k * amplitude_x * np.sin(tau1)
            + (b21 * amplitude_x**2 - b22 * amplitude_z**2) * np.sin(2 * tau1)
            + (b31 * amplitude_x**3 - b32 * amplitude_x * amplitude_z**2) * np.sin(3 * tau1)
        )
        z = (
            deltan * amplitude_z * np.cos(tau1)
            + deltan * d21 * amplitude_x * amplitude_z * (np.cos(2 * tau1) - 3)
            + deltan * (d32 * amplitude_z * amplitude_x**2 - d31 * amplitude_z**3) * np.cos(3 * tau1)
        )

        xdot = (
            lam * amplitude_x * np.sin(tau1)
            - 2 * lam * (a23 * amplitude_x**2 - a24 * amplitude_z**2) * np.sin(2 * tau1)
            - 3 * lam * (a31 * amplitude_x**3 - a32 * amplitude_x * amplitude_z**2) * np.sin(3 * tau1)
        )
        ydot = (
            lam
            * (
                k * amplitude_x * np.cos(tau1)
                + 2 * (b21 * amplitude_x**2 - b22 * amplitude_z**2) * np.cos(2 * tau1)
                + 3 * (b31 * amplitude_x**3 - b32 * amplitude_x * amplitude_z**2) * np.cos(3 * tau1)
            )
        )
        zdot = (
            - lam * deltan * amplitude_z * np.sin(tau1)
            - 2 * lam * deltan * d21 * amplitude_x * amplitude_z * np.sin(2 * tau1)
            - 3 * lam * deltan * (d32 * amplitude_z * amplitude_x**2 - d31 * amplitude_z**3) * np.sin(3 * tau1)
        )

        # Scale back by gamma using original transformation
        rx = primary + gamma * (-won + x)
        ry = -gamma * y
        rz = gamma * z

        vx = gamma * xdot
        vy = gamma * ydot
        vz = gamma * zdot

        # Return the state vector
        logger.debug(f"Generated initial guess for Halo orbit around {self.libration_point} with amplitude_z={self.amplitude}: {np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)}")
        return np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)

    def _halo_quadratic_term(self, X_ev, Phi):
        r"""
        Evaluate the quadratic part of the Jacobian for differential correction.

        Parameters
        ----------
        X_ev : numpy.ndarray, shape (6,)
            State vector at the event time (half-period).
        Phi : numpy.ndarray
            State-transition matrix evaluated at the same event; only the
            row corresponding to :pyattr:`S.Y` and the columns
            :pyattr:`S.X`, :pyattr:`S.VY` are used.

        Returns
        -------
        numpy.ndarray, shape (2, 2)
            Reduced Jacobian matrix employed by the
            :pyfunc:`hiten.system.orbits.base.PeriodicOrbit.correct`
            solver.
        """
        x, y, z, vx, vy, vz = X_ev
        mu2 = 1 - self.mu
        rho_1 = 1/(((x+self.mu)**2 + y**2 + z**2)**1.5)
        rho_2 = 1/(((x-mu2 )**2 + y**2 + z**2)**1.5)
        omega_x  = -(mu2*(x+self.mu)*rho_1) - (self.mu*(x-mu2)*rho_2) + x
        DDx = 2*vy + omega_x
        DDz = -(mu2*z*rho_1) - (self.mu*z*rho_2)

        if abs(vy) < 1e-9:
            logger.warning(f"Denominator 'vy' is very small ({vy:.2e}). Correction step may be inaccurate.")
            vy = np.sign(vy) * 1e-9 if vy != 0 else 1e-9
            
        return np.array([[DDx],[DDz]]) @ Phi[[S.Y],:][:, (S.X,S.VY)] / vy

    def eccentricity(self) -> float:
        """Eccentricity is not a well-defined concept for halo orbits."""
        return np.nan
