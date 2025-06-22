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

from hiten.system.libration.collinear import (CollinearPoint, L1Point, L2Point,
                                              L3Point)
from hiten.system.orbits.base import (PeriodicOrbit, S, correctionConfig,
                                      orbitConfig)
from hiten.utils.log_config import logger


class HaloOrbit(PeriodicOrbit):
    r"""
    Halo orbit class.

    Parameters
    ----------
    config : orbitConfig
        Problem definition comprising the primaries, target
        :pyclass:`hiten.system.libration.collinear.CollinearPoint`, numerical
        tolerances and any extra parameters.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector
        :math:`[x,\,y,\,z,\,\dot{x},\,\dot{y},\,\dot{z}]` in the rotating
        synodic frame. When *None* an analytical initial guess is generated
        from :pyattr:`config.extra_params` (requires ``'Az'`` and
        ``'Zenith'``).

    Attributes
    ----------
    Az : float or None
        :math:`z`-amplitude of the halo orbit in the synodic frame.
    Zenith : {'northern', 'southern'} or None
        Indicates the symmetry branch with respect to the :math:`x\,y`-plane.

    Raises
    ------
    ValueError
        If the required amplitude or branch is missing and *initial_state*
        is *None*.
    TypeError
        If *config.libration_point* is not an instance of
        :pyclass:`hiten.system.libration.collinear.CollinearPoint`.
    """
    Az: Optional[float] # Amplitude of the halo orbit
    Zenith: Optional[Literal["northern", "southern"]]

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.config = config

        self._initial_state = None
        if initial_state is None:
            try:
                self.Az = config.extra_params['Az']
                self.Zenith = config.extra_params['Zenith']
            except KeyError:
                err = "Halo hiten.system.orbits require an 'Az' (z-amplitude) parameter and a 'Zenith' parameter ('northern' or 'southern') OR an initial state."
                logger.error(err)
                raise ValueError(err)
        else:
            self._initial_state = np.array(initial_state, dtype=np.float64)
            self.Az = self._initial_state[2]
            self.Zenith = "northern" if self._initial_state[2] > 0 else "southern"
        if self.Az is None:
            self.Az = self._initial_state[2]
        if self.Zenith is None:
            self.Zenith = "northern" if self._initial_state[2] > 0 else "southern"

        super().__init__(config, initial_state)

        if not isinstance(self.libration_point, CollinearPoint):
            msg = f"Expected CollinearPoint, got {type(self.libration_point)}."
            logger.error(msg)
            raise TypeError(msg)

        if isinstance(self.libration_point, L3Point):
            logger.warning("Must supply initial state for L3 halo hiten.system.orbits.")

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
        >>> cfg = orbitConfig(system, L1Point(system), extra_params={'Az': 0.01, 'Zenith': 'northern'})
        >>> orb = HaloOrbit(cfg)
        >>> y0 = orb._initial_guess()
        """
        # Determine sign (won) and which "primary" to use

        if self._initial_state is not None:
            logger.info(f"Using provided initial state: {self._initial_state} for {str(self)}")
            return self._initial_state

        mu = self.mu
        Az = self.Az
        # Get gamma from the libration point instance property
        gamma = self.libration_point.gamma
        
        if isinstance(self.libration_point, L1Point):
            won = +1
            primary = 1 - mu
        elif isinstance(self.libration_point, L2Point):
            won = -1
            primary = 1 - mu 
        elif isinstance(self.libration_point, L3Point):
            won = +1
            primary = -mu
        else:
            raise ValueError(f"Halo hiten.system.orbits only supported for L1, L2, L3 (got L{self.libration_point})")
        
        # Set n for northern/southern family
        n = 1 if self.Zenith == "northern" else -1
        
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

        # Solve for Ax from the condition ( -del - l2*Az^2 ) / l1
        Ax = np.sqrt((-delta - l2 * Az**2) / l1)

        # Evaluate the expansions at tau1 = 0
        tau1 = 0.0
        
        x = (
            a21 * Ax**2 + a22 * Az**2
            - Ax * np.cos(tau1)
            + (a23 * Ax**2 - a24 * Az**2) * np.cos(2 * tau1)
            + (a31 * Ax**3 - a32 * Ax * Az**2) * np.cos(3 * tau1)
        )
        y = (
            k * Ax * np.sin(tau1)
            + (b21 * Ax**2 - b22 * Az**2) * np.sin(2 * tau1)
            + (b31 * Ax**3 - b32 * Ax * Az**2) * np.sin(3 * tau1)
        )
        z = (
            deltan * Az * np.cos(tau1)
            + deltan * d21 * Ax * Az * (np.cos(2 * tau1) - 3)
            + deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.cos(3 * tau1)
        )

        xdot = (
            lam * Ax * np.sin(tau1)
            - 2 * lam * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau1)
            - 3 * lam * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau1)
        )
        ydot = (
            lam
            * (
                k * Ax * np.cos(tau1)
                + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau1)
                + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau1)
            )
        )
        zdot = (
            - lam * deltan * Az * np.sin(tau1)
            - 2 * lam * deltan * d21 * Ax * Az * np.sin(2 * tau1)
            - 3 * lam * deltan * (d32 * Az * Ax**2 - d31 * Az**3) * np.sin(3 * tau1)
        )

        # Scale back by gamma using original transformation
        rx = primary + gamma * (-won + x)
        ry = -gamma * y
        rz = gamma * z

        vx = gamma * xdot
        vy = gamma * ydot
        vz = gamma * zdot

        # Return the state vector
        logger.debug(f"Generated initial guess for Halo orbit around {self.libration_point} with Az={self.Az}: {np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)}")
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
            :pyfunc:`hiten.system.orbits.base.PeriodicOrbit.differential_correction`
            solver.
        """
        x, y, z, vx, vy, vz = X_ev
        mu2 = 1 - self.mu
        rho_1 = 1/(((x+self.mu)**2 + y**2 + z**2)**1.5)
        rho_2 = 1/(((x-mu2 )**2 + y**2 + z**2)**1.5)
        omega_x  = -(mu2*(x+self.mu)*rho_1) - (self.mu*(x-mu2)*rho_2) + x
        DDx = 2*vy + omega_x
        DDz = -(mu2*z*rho_1) - (self.mu*z*rho_2)
        return np.array([[DDx],[DDz]]) @ Phi[[S.Y],:][:, (S.X,S.VY)] / vy

    def differential_correction(self, **kw):
        cfg = correctionConfig(
            residual_indices=(S.VX, S.VZ),
            control_indices=(S.X,  S.VY),
            extra_jacobian=self._halo_quadratic_term
        )
        return super().differential_correction(cfg, **kw)


    def eccentricity(self) -> float:
        raise NotImplementedError("Eccentricity calculation not implemented for HaloOrbit.")
