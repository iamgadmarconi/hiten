from typing import Callable, List, Sequence

import numpy as np

from hiten.algorithms.bifurcation.analysis import (_eigenvalues_flow,
                                                   _equilibrium_with_energy)
from hiten.utils.log_config import logger


def _sigma_indicator(eigvals: np.ndarray) -> float:
    return float(np.max(eigvals.real))


def _nu_indicator(eigvals: np.ndarray) -> float:
    return float(np.min(np.abs(eigvals.imag)))


class _BifurcationEngine:
    def __init__(
        self,
        *,
        coeffs_list,
        clmoF,
        I_theta_seed: np.ndarray,
        energy_seed: float,
        energy_target: Sequence[float] | tuple[float, float],
        step: float = 1e-4,
        indicator: Callable[[np.ndarray], float] = _sigma_indicator,
        tol: float = 1e-8,
        max_points: int = 2048,
    ) -> None:
        I_theta_seed = np.asarray(I_theta_seed, dtype=float)
        if I_theta_seed.shape != (6,):
            raise ValueError("I_theta_seed must be a 6-component vector (I1,I2,I3,theta1,theta2,theta3)")

        energy_target = tuple(map(float, energy_target))
        if len(energy_target) != 2:
            raise ValueError("energy_target must be a 2-tuple (E_min,E_max)")

        self._coeffs_list = coeffs_list
        self._clmoF = clmoF
        self._tol = float(tol)
        self._max_points = int(max_points)
        self._indicator_func = indicator

        step = float(step)
        E_min, E_max = min(energy_target), max(energy_target)
        if not (E_min <= energy_seed <= E_max):
            raise ValueError("energy_seed must lie inside the target interval")
        if (energy_seed == E_min and step < 0) or (energy_seed == E_max and step > 0):
            step = -step  # flip sign so that we march *into* the interval
        self._step = step

        self._E_min = E_min
        self._E_max = E_max

        self._energies: List[float] = []
        self._indicators: List[float] = []
        self._states: List[np.ndarray] = []  # (I,theta) pairs
        self._events: List[dict] = []  # detected bifurcation events

        vec_guess = np.concatenate((I_theta_seed, [0.0]))  # Lagrange multiplier guess 0
        I, th, converged, _ = _equilibrium_with_energy(
            coeffs_list,
            clmoF,
            vec_guess,
            target_energy=energy_seed,
            tol=self._tol,
        )
        if not converged:
            raise RuntimeError("Seed equilibrium did not converge - aborting scan.")

        eig0 = _eigenvalues_flow(coeffs_list, I, th, clmoF, sort=True)
        ind0 = indicator(eig0)

        self._energies.append(float(energy_seed))
        self._indicators.append(ind0)
        self._states.append(np.concatenate((I, th)))

        logger.info(
            "Bifurcation scan initialised: E=%.6e, indicator=%.3e, step=%.1e, target=[%.6e, %.6e]",
            energy_seed,
            ind0,
            self._step,
            self._E_min,
            self._E_max,
        )

    @property
    def energies(self) -> Sequence[float]:
        return tuple(self._energies)

    @property
    def indicator_values(self) -> Sequence[float]:
        return tuple(self._indicators)

    @property
    def states(self) -> Sequence[np.ndarray]:
        return tuple(self._states)

    @property
    def events(self) -> Sequence[dict]:
        return tuple(self._events)

    def run(self) -> List[dict]:
        logger.info("Starting bifurcation scan ...")
        attempts_at_current_step = 0
        while not self._stop_condition():
            if len(self._energies) >= self._max_points:
                logger.warning("Reached max_points=%d, terminating scan.", self._max_points)
                break

            E_trial = self._energies[-1] + self._step
            if (self._step > 0 and E_trial > self._E_max) or (self._step < 0 and E_trial < self._E_min):
                break

            try:
                state_guess = self._states[-1]
                vec_guess = np.concatenate((state_guess, [0.0]))
                I, th, converged, _ = _equilibrium_with_energy(
                    self._coeffs_list,
                    self._clmoF,
                    vec_guess,
                    target_energy=E_trial,
                    tol=self._tol,
                )
                if not converged:
                    raise RuntimeError("Newton solver did not converge")
            except Exception as exc:
                logger.debug(
                    "Equilibrium correction failed at E=%.6e (attempt %d): %s",
                    E_trial,
                    attempts_at_current_step + 1,
                    exc,
                    exc_info=exc,
                )
                self._step = self._update_step(self._step, success=False)
                attempts_at_current_step += 1
                if attempts_at_current_step > 10:
                    logger.error("Too many failed attempts at current step; aborting scan.")
                    break
                continue

            attempts_at_current_step = 0
            eig = _eigenvalues_flow(self._coeffs_list, I, th, self._clmoF, sort=True)
            ind_val = self._indicator_func(eig)

            self._energies.append(E_trial)
            self._indicators.append(ind_val)
            self._states.append(np.concatenate((I, th)))

            logger.debug(
                "Accepted point #%d, E=%.6e, indicator=%.3e",
                len(self._energies) - 1,
                E_trial,
                ind_val,
            )

            prev_ind = self._indicators[-2]
            if self._is_event(prev_ind, ind_val):
                event = {
                    "energy": E_trial,
                    "indicator": ind_val,
                    "index": len(self._energies) - 1,
                }
                self._events.append(event)
                logger.info("Possible bifurcation detected near E=%.6e (indicator changed sign).", E_trial)

            self._step = self._update_step(self._step, success=True)

        logger.info("Bifurcation scan finished analysed %d points, detected %d events.", len(self._energies), len(self._events))
        return self._events

    def _update_step(self, current_step: float, *, success: bool) -> float:
        factor = 2.0 if success else 0.5
        new_step = current_step * factor
        return float(np.clip(new_step, 1e-10, 1.0))

    def _stop_condition(self) -> bool:
        E = self._energies[-1]
        return E < self._E_min or E > self._E_max

    def _is_event(self, ind_prev: float, ind_curr: float) -> bool:
        return (abs(ind_prev) > self._tol or abs(ind_curr) > self._tol) and (
            np.sign(ind_prev) != np.sign(ind_curr)
        )
