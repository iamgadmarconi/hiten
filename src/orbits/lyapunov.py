import numpy as np
from numpy.typing import NDArray
from typing import Optional, Sequence, Tuple, Any
from scipy.integrate import solve_ivp

from system.libration import CollinearPoint, L3Point
from algorithms.geometry import _find_y_zero_crossing
from algorithms.dynamics import variational_equations, compute_stm
from orbits.base import PeriodicOrbit, orbitConfig

from utils.log_config import logger


class LyapunovOrbit(PeriodicOrbit):
    Ax: float # Amplitude of the Lyapunov orbit

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.Ax = config.extra_params['Ax']
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
        Performs single-shooting differential correction for a planar Lyapunov orbit.

        Adjusts the initial vy component to achieve vx = 0 at the x-z plane crossing.

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

        Returns
        -------
        Tuple[NDArray[np.float64], float]
            Converged initial state vector and half-period.
        
        Raises
        ------
        RuntimeError
            If convergence is not achieved within max_attempts.
        """
        X0: NDArray[np.float64] = np.copy(self.initial_state)
        logger.info(f"Starting differential correction for Lyapunov orbit around {self.libration_point} with Ax={self.Ax}.")
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
        vx_cross = X_cross[3]

        if abs(vx_cross) < tol:
            half_period = t_cross
            self._reset()
            self._initial_state = X0
            self.period = 2 * half_period
            logger.info(f"Converged successfully after {attempt} attempts.")
            logger.info(f"Converged Initial State: {np.array2string(self.initial_state, precision=12, suppress_small=True)}")
            logger.info(f"Period: {self.period:.6f} (Half period: {half_period:.6f})")
            return self.initial_state, half_period

        while True:
            attempt += 1
            logger.debug(f"Correction attempt {attempt}")
            if attempt > max_attempts:
                msg = f"Failed to converge Lyapunov orbit after {max_attempts} attempts. Last state: {X0}"
                logger.error(msg)
                raise RuntimeError(msg)
            
            # 1. Find the time and state of the next x-z plane crossing (y=0)
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
            # x_cross, y_cross, z_cross, vx_cross, vy_cross, vz_cross = X_cross
            vx_cross = X_cross[3] # Indexing for performance

            # 2. Check convergence: Is vx sufficiently close to zero?
            if abs(vx_cross) < tol:
                half_period = t_cross
                self._reset() # Clear any intermediate data if needed
                self._initial_state = X0
                self.period = 2 * half_period
                logger.info(f"Converged successfully after {attempt} attempts.")
                logger.info(f"Converged Initial State: {np.array2string(self.initial_state, precision=12, suppress_small=True)}")
                logger.info(f"Period: {self.period:.6f} (Half period: {half_period:.6f})")
                return self.initial_state, half_period
            
            logger.debug(f"Attempt {attempt}: vx_cross={vx_cross:.3e} (target=0), tolerance={tol}. Applying correction.")

            # 3. Integrate variational equations (state + STM) from t=0 to t_cross
            try:
                # Call compute_stm with solver parameters similar to _find_y_zero_crossing
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

            # Extract final STM and state (though state isn't used for correction itself)
            # phi_final is already the 6x6 matrix from compute_stm

            # 4. Calculate the required correction in initial vy (dvy0)
            # We need the partial derivative d(vx_cross) / d(vy0)
            # From the STM: phi_final[row, col] = d(state_row(t_cross)) / d(state_col(0))
            # So we need phi_final[3, 4] = d(vx(t_cross)) / d(vy(0))
            dvx_dvy0: float = phi_final[3, 4]
            logger.debug(f"STM element phi[3,4] (dvx_cross/dvy0) = {dvx_dvy0:.4e}")
            
            if abs(dvx_dvy0) < 1e-12: # Avoid division by zero/very small number
                msg = f"STM element dvx/dvy0 ({dvx_dvy0:.3e}) is too small; correction unstable. Attempt {attempt}. Last state: {X0}"
                logger.error(msg)
                raise RuntimeError(msg)

            # Linear correction: target_vx - current_vx = (dvx/dvy0) * dvy0
            # 0 - vx_cross = dvx_dvy0 * dvy0
            dvy0: float = -vx_cross / dvx_dvy0
            logger.debug(f"Calculated correction dvy0 = {dvy0:.3e}")

            # 5. Apply the correction to the initial state guess
            X0[4] += dvy0 # Update initial vy
            logger.debug(f"Updated initial state guess for next attempt: {X0}")


    def eccentricity(self) -> float:
        """Eccentricity is not typically defined for Lyapunov orbits.
        
        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError("Eccentricity is not implemented for Lyapunov orbits.")
