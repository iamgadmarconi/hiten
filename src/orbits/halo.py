import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple, Any
from scipy.integrate import solve_ivp


from orbits.base import PeriodicOrbit, orbitConfig
from system.libration import CollinearPoint, L1Point, L2Point, L3Point
from algorithms.geometry import _gamma_L, _find_y_zero_crossing
from algorithms.dynamics import compute_stm
from log_config import logger


class HaloOrbit(PeriodicOrbit):
    Az: float # Amplitude of the halo orbit

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.config = config
        self.Az = config.extra_params['Az']
        self.zenith = config.extra_params['Zenith']
        super().__init__(config, initial_state)

        if not isinstance(self.libration_point, CollinearPoint):
            msg = f"Expected CollinearPoint, got {type(self.libration_point)}."
            logger.error(msg)
            raise TypeError(msg)

        if isinstance(self.libration_point, L3Point):
            raise NotImplementedError("Halo orbits not supported for L3.")

    def _initial_guess(self) -> NDArray[np.float64]:
        """
        Generate initial conditions for a Halo orbit around a libration point.
        
        Returns
        -------
        ndarray
            6D state vector [x, y, z, vx, vy, vz] in the rotating frame
        """
        # Determine sign (won) and which "primary" to use
        mu = self.mu
        Az = self.Az
        gamma = _gamma_L(mu, self.libration_point)
        
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
            raise ValueError(f"Halo orbits only supported for L1, L2, L3 (got L{L_i})")
        
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

    def differential_correction(self, 
                                tol: float = 1e-10, 
                                max_attempts: int = 25,
                                forward: int = 1,
                                # Args for _find_y_zero_crossing's propagator
                                crossing_steps: int = 500, 
                                crossing_rtol: float = 1e-12, 
                                crossing_atol: float = 1e-12, 
                                crossing_method: str = 'DOP853',
                                # Args for STM integration (solve_ivp)
                                stm_rtol: float = 1e-12, 
                                stm_atol: float = 1e-12, 
                                stm_method: str = 'DOP853',
                                **kwargs: Any # Allow unused kwargs
                                ) -> Tuple[NDArray[np.float64], float]:
        """
        Parameters
        ----------
        tol : float, optional
            Convergence tolerance for vx at crossing, by default 1e-10
        max_attempts : int, optional
            Maximum correction iterations, by default 25
        forward : int, optional
            Time integration direction (1 for forward, -1 for backward), by default 1
        crossing_steps : int, optional
            Number of steps hint for _find_y_zero_crossing, by default 500
        crossing_rtol : float, optional
            Relative tolerance for _find_y_zero_crossing's propagator, by default 1e-12
        crossing_atol : float, optional
            Absolute tolerance for _find_y_zero_crossing's propagator, by default 1e-12
        crossing_method : str, optional
            Integration method for _find_y_zero_crossing's propagator, by default 'DOP853'
        stm_rtol : float, optional
            Relative tolerance for STM integration (solve_ivp), by default 1e-12
        stm_atol : float, optional
            Absolute tolerance for STM integration (solve_ivp), by default 1e-12
        stm_method : str, optional
            Integration method for STM integration (solve_ivp), by default 'DOP853'
        **kwargs : Any
            Allows for additional unused keyword arguments.
        """
        mu = self.mu
        X0: NDArray[np.float64] = np.copy(self.initial_state)
        logger.info(f"Starting differential correction for Halo orbit around {self.libration_point} with Az={self.Az}.")
        logger.debug(f"Initial guess: {X0}")
        logger.debug(f"Correction params: tol={tol}, max_attempts={max_attempts}, forward={forward}, crossing_method='{crossing_method}', stm_method='{stm_method}'")

        attempt = 0

        t_cross, X_cross = _find_y_zero_crossing(X0, 
                                                self.mu, 
                                                forward=forward, 
                                                steps=crossing_steps, # Hint for initial propagation
                                                # Pass solver args for internal root-finding propagation
                                                rtol=crossing_rtol, 
                                                atol=crossing_atol, 
                                                method=crossing_method)
        
        vx_cross, vz_cross = X_cross[3], X_cross[5]
        
        if abs(vx_cross) <= tol and abs(vz_cross) <= tol:
            half_period = t_cross
            self._reset()
            self._initial_state = X0
            self.period = 2 * half_period
            logger.info(f"Converged successfully after {attempt} attempts.")
            logger.info(f"Converged Initial State: {np.array2string(self.initial_state, precision=12, suppress_small=True)}")
            logger.info(f"Period: {self.period:.6f} (Half period: {half_period:.6f})")
            return self.initial_state, half_period

        # We will iterate until vx_cross is small enough
        vx_cross = 1.0

        # For convenience:
        mu2 = 1.0 - mu

        while True:
            attempt += 1
            logger.debug(f"Correction attempt {attempt}")
            if attempt > max_attempts:
                msg = f"Failed to converge Halo orbit after {max_attempts} attempts. Last state: {X0}"
                logger.error(msg)
                raise RuntimeError(msg)

            try:
                t_cross, X_cross = _find_y_zero_crossing(X0, 
                                                        self.mu, 
                                                        forward=forward, 
                                                        steps=crossing_steps, # Hint for initial propagation
                                                        # Pass solver args for internal root-finding propagation
                                                        rtol=crossing_rtol, 
                                                        atol=crossing_atol, 
                                                        method=crossing_method)
                logger.debug(f"Found y=0 crossing at t={t_cross:.6f}, state={X_cross}")
            except Exception as e:
                msg = f"Error in _find_y_zero_crossing during attempt {attempt}: {e}. Last state: {X0}"
                logger.error(msg)
                raise RuntimeError(msg) from e

            # Extract relevant components at crossing
            x_cross, y_cross, z_cross, vx_cross, vy_cross, vz_cross = X_cross

            if abs(vx_cross) <= tol:
                half_period = t_cross
                self._reset()
                self._initial_state = X0
                self.period = 2 * half_period
                logger.info(f"Converged successfully after {attempt} attempts.")
                logger.info(f"Converged Initial State: {np.array2string(self.initial_state, precision=12, suppress_small=True)}")
                logger.info(f"Period: {self.period:.6f} (Half period: {half_period:.6f})")
                return self.initial_state, half_period

            try:
                _, _, phi_final, _ = compute_stm(X0, 
                                                self.mu, 
                                                t_cross, 
                                                forward=forward,
                                                # Pass solver kwargs
                                                method=stm_method,
                                                rtol=stm_rtol,
                                                atol=stm_atol,
                                                dense_output=True)
                logger.debug(f"Computed STM at t={t_cross:.6f}")
            except Exception as e:
                msg = f"Error during STM integration (compute_stm) attempt {attempt}: {e}. Last state: {X0}"
                logger.error(msg)
                raise RuntimeError(msg) from e

            # 3) Compute partial derivatives for correction
            #    (these replicate the CR3BP equations used in the Matlab code)
            rho1 = 1.0 / ((x_cross + mu)**2 + y_cross**2 + z_cross**2)**1.5
            rho2 = 1.0 / ((x_cross - mu2)**2 + y_cross**2 + z_cross**2)**1.5

            # second-derivatives
            omgx1 = -(mu2 * (x_cross + mu) * rho1) - (mu * (x_cross - mu2) * rho2) + x_cross
            DDz1  = -(mu2 * z_cross * rho1) - (mu * z_cross * rho2)
            DDx1  = 2.0 * vy_cross + omgx1

            # 4) CASE=1 Correction: fix z0
            #    We want to kill Dx1 and Dz1 by adjusting x0 and Dy0.
            #    In the Matlab code:
            #
            #    C1 = [phi(4,1) phi(4,5);
            #          phi(6,1) phi(6,5)];
            #    C2 = C1 - (1/Dy1)*[DDx1 DDz1]'*[phi(2,1) phi(2,5)];
            #    C3 = inv(C2)*[-Dx1 -Dz1]';
            #    dx0  = C3(1);
            #    dDy0 = C3(2);

            C1 = np.array([[phi_final[3, 0], phi_final[3, 4]],
                        [phi_final[5, 0], phi_final[5, 4]]])

            # Vector for partial derivative in the (Dx, Dz) direction
            # [DDx1, DDz1]^T (2x1) times [phi(2,1), phi(2,5)] (1x2)
            dd_vec = np.array([[DDx1], [DDz1]])  # Shape (2,1)
            phi_2 = np.array([[phi_final[1, 0], phi_final[1, 4]]])  # Shape (1,2)
            partial = dd_vec @ phi_2  # Result is (2,2)

            # Subtract the partial derivative term, scaled by 1/Dy1
            C2 = C1 - (1/vy_cross) * partial
            
            # Add regularization if matrix is nearly singular
            if np.linalg.det(C2) < 1e-10:
                C2 += np.eye(2) * 1e-10

            # Compute the correction
            C3 = np.linalg.solve(C2, np.array([[-vx_cross], [-vz_cross]]))

            # Apply the correction
            dx0 = C3[0, 0]
            dDy0 = C3[1, 0]

            # Update the initial guess
            X0[0] += dx0
            X0[4] += dDy0


    def eccentricity(self) -> float:
        """
        Calculate the eccentricity of the halo orbit.
        """
        raise NotImplementedError("Eccentricity calculation not implemented for HaloOrbit.")
