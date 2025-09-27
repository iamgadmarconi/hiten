"""Adapters backing manifold propagation, stability, and persistence services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Literal, Tuple

import numpy as np
from tqdm import tqdm

from hiten.algorithms.common.energy import _max_rel_energy_error
from hiten.algorithms.types.states import Trajectory
from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.linalg.base import StabilityProperties
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.types import _ProblemType, _SystemType
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.utils.io.manifold import load_manifold, save_manifold
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.libration import LibrationPoint
    from hiten.system.manifold import Manifold
    from hiten.system.orbits import PeriodicOrbit


class _ManifoldPersistenceService(_PersistenceServiceBase):
    """Persistence helpers for manifold objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda manifold, path, **kw: save_manifold(manifold, Path(path), **kw),
            load_fn=lambda path, **kw: load_manifold(Path(path), **kw),
        )


class _ManifoldDynamicsService(_DynamicsServiceBase):
    """Manage STM computation and manifold trajectory generation."""

    def __init__(self, manifold: "Manifold") -> None:
        super().__init__(manifold)

        self._stable = 1 if self.domain_obj.stable else -1
        self._direction = 1 if self.domain_obj.direction == "positive" else -1
        self._forward = - self._stable
        self._manifold_result = None

    @property
    def orbit(self) -> "PeriodicOrbit":
        return self.domain_obj.generating_orbit

    @property
    def period(self) -> float:
        return self.orbit.period

    @property
    def stable(self) -> bool:
        return self._stable

    @property
    def direction(self) -> bool:
        return self._direction

    @property
    def forward(self) -> bool:
        return self._forward

    @property
    def libration_point(self) -> "LibrationPoint":
        return self.orbit.libration_point

    @property
    def system(self) -> "System":
        return self.libration_point.system

    @property
    def mu(self) -> float:
        return self.system.mu

    @property
    def dynsys(self) -> _DynamicalSystem:
        return self.system.dynsys
    
    @property
    def var_dynsys(self) -> _DynamicalSystem:
        return self.system.var_dynsys
    
    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        return self.system.jacobian_dynsys

    @property
    def stm(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.compute_stm(steps=2000)

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.stability.eigenvalues

    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.stability.eigenvectors

    @property
    def sn(self) -> np.ndarray:
        return self.stability.stable

    @property
    def un(self) -> np.ndarray:
        return self.stability.unstable

    @property
    def cn(self) -> np.ndarray:
        return self.stability.center
    
    @property
    def wsn(self) -> np.ndarray:
        return self.stability.Ws

    @property
    def wun(self) -> np.ndarray:
        return self.stability.Wu

    @property
    def wcn(self) -> np.ndarray:
        return self.stability.Wc        

    @property
    def stability(self) -> StabilityProperties:
        return self.compute_stability()

    @property
    def manifold_result(self) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]:
        return self._manifold_result

    @property
    def trajectories(self) -> List[Trajectory]:
        times_list = self._manifold_result[2]
        states_list = self._manifold_result[3]
        return [Trajectory(times, states) for times, states in zip(times_list, states_list)]

    @property
    def compute_stm(
        self,
        *,
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = self.make_key(id(self.orbit), steps, self.forward)
        
        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return _compute_stm(
                self.var_dynsys,
                self.orbit.initial_state,
                self.period,
                steps=steps,
                forward=self.forward,
            )
        
        return self.get_or_create(cache_key, _factory)

    def compute_manifold(
        self,
        *,
        step: float,
        integration_fraction: float,
        NN: int,
        displacement: float,
        method: str,
        order: int,
        dt: float,
        energy_tol: float,
        safe_distance: float,
        show_progress: bool,
    ) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]:
        cache_key = self.make_key(
            id(self.orbit),
            self.stable,
            self.direction,
            step,
            integration_fraction,
            NN,
            displacement,
            method,
            order,
            dt,
            energy_tol,
            safe_distance,
        )

        def _factory() -> Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]:
            self._manifold_result = self._run_compute(
                step=step,
                integration_fraction=integration_fraction,
                NN=NN,
                displacement=displacement,
                method=method,
                order=order,
                dt=dt,
                energy_tol=energy_tol,
                safe_distance=safe_distance,
                show_progress=show_progress,
            )
            return self._manifold_result

        return self.get_or_create(cache_key, _factory)

    def _run_compute(
        self,
        *,
        step: float,
        integration_fraction: float,
        NN: int,
        displacement: float,
        method: str,
        order: int,
        dt: float,
        energy_tol: float,
        safe_distance: float,
        show_progress: bool,
    ) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]:
        orbit = self.orbit
        mu = self.mu
        forward = self.forward

        dist_m = self.system.distance * 1e3
        pr_nd = self.system.primary.radius / dist_m
        sr_nd = self.system.secondary.radius / dist_m
        safe_r1 = safe_distance * pr_nd
        safe_r2 = safe_distance * sr_nd

        sn, un, _ = self.eigenvalues
        Ws, Wu, _ = self.eigenvectors

        _, snreal_vecs = self.stability.get_real_eigenvectors(Ws, sn)
        _, unreal_vecs = self.stability.get_real_eigenvectors(Wu, un)  

        col_idx = NN - 1
        if self.stable == 1:
            if snreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(
                    f"Requested stable eigenvector {NN} not available. "
                    f"Only {snreal_vecs.shape[1]} real stable eigenvectors found."
                )
            eigvec = snreal_vecs[:, col_idx]
        else:
            if unreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(
                    f"Requested unstable eigenvector {NN} not available. "
                    f"Only {unreal_vecs.shape[1]} real unstable eigenvectors found."
                )
            eigvec = unreal_vecs[:, col_idx]

        fractions: Iterable[float] = tuple(np.arange(0.0, 1.0, step))
        iterator = (
            tqdm(fractions, desc="Computing manifold") if show_progress else fractions
        )

        ysos: list[float] = []
        dysos: list[float] = []
        states_list = []
        times_list = []
        successes = 0
        attempts = 0

        for fraction in iterator:
            attempts += 1
            try:
                # Get cached STM data
                xx, tt, _, PHI = self.compute_stm(steps=2000)
                
                x0W = self._compute_manifold_section(
                    period=orbit.period,
                    fraction=fraction,
                    displacement=displacement,
                    xx=xx,
                    tt=tt,
                    PHI=PHI,
                    eigvec=eigvec,
                ).astype(np.float64)
                tf = integration_fraction * 2 * np.pi
                steps = max(int(abs(tf) / dt) + 1, 100)

                sol = _propagate_dynsys(
                    dynsys=self.dynsys,
                    state0=x0W,
                    t0=0.0,
                    tf=tf,
                    forward=forward,
                    steps=steps,
                    method=method,
                    order=order,
                    flip_indices=slice(0, 6),
                )
                states, times = sol.states, sol.times

                x = states[:, 0]
                y = states[:, 1]
                z = states[:, 2]

                r1 = np.sqrt((x + mu) ** 2 + y ** 2 + z ** 2)
                r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2 + z ** 2)

                if (r1.min() < safe_r1) or (r2.min() < safe_r2):
                    logger.debug(
                        f"Fraction {fraction:.3f}: Trajectory discarded due to body-radius proximity "
                        f"(min(r1)={r1.min():.2e}, min(r2)={r2.min():.2e})"
                    )
                    continue

                max_energy_err = _max_rel_energy_error(states, mu)
                if max_energy_err > energy_tol:
                    logger.warning(
                        f"Fraction {fraction:.3f}: Trajectory discarded due to energy drift "
                        f"(|C(t)|/|C(0)|={max_energy_err:.2e} > {energy_tol:.1e})"
                    )
                    continue

                states_list.append(states)
                times_list.append(times)
                successes += 1

            except Exception as exc:
                logger.error(f"Error computing manifold: {exc}")
                continue

        return (ysos, dysos, states_list, times_list, successes, attempts)

    def compute_stability(self) -> StabilityProperties:
        key = self.make_key(id(self.domain_obj))
        
        def _factory() -> StabilityProperties:
            config = _EigenDecompositionConfig(
                problem_type=_ProblemType.ALL,
                system_type=_SystemType.DISCRETE,
            )
            stability = StabilityProperties.with_default_engine()
            _, _, phi_T, _ = self.compute_stm(steps=2000)
            stability.compute(
                matrix=phi_T,
                system_type=config.system_type,
                problem_type=config.problem_type,
            )
            return stability
        
        return self.get_or_create(key, _factory)

    def _compute_manifold_section(
        self,
        *,
        period: float,
        fraction: float,
        displacement: float,
        xx: np.ndarray,
        tt: np.ndarray,
        PHI: np.ndarray,
        eigvec: np.ndarray,
    ) -> np.ndarray:
        mfrac = self._totime(tt, fraction * period)

        if np.isscalar(mfrac):
            mfrac_idx = mfrac
        else:
            mfrac_idx = mfrac[0]

        phi_frac_flat = PHI[mfrac_idx, :36]
        phi_frac = phi_frac_flat.reshape((6, 6))

        MAN = self.direction * (phi_frac @ eigvec)

        disp_magnitude = np.linalg.norm(MAN[0:3])

        if disp_magnitude < 1e-14:
            logger.warning(
                "Very small displacement magnitude: %.2e, setting to 1.0",
                disp_magnitude,
            )
            disp_magnitude = 1.0
        d = displacement / disp_magnitude

        fracH = xx[mfrac_idx, :].copy()

        x0W = fracH + d * MAN.real
        x0W = x0W.flatten()

        if abs(x0W[2]) < 1.0e-15:
            x0W[2] = 0.0
        if abs(x0W[5]) < 1.0e-15:
            x0W[5] = 0.0

        return x0W

    def _totime(self, t, tf):
        """Find indices of closest time values in array.

        Searches time array for indices where values are closest to specified
        target times. Useful for extracting trajectory points at specific times.

        Parameters
        ----------
        t : array_like
            Time array to search.
        tf : float or array_like
            Target time value(s) to locate.

        Returns
        -------
        ndarray
            Indices where t values are closest to corresponding tf values.

        Notes
        -----
        - Uses absolute time values, so signs are ignored
        - Particularly useful for periodic orbit analysis
        - Returns single index for scalar tf, array of indices for array tf
        
        Examples
        --------
        >>> import numpy as np
        >>> from hiten.algorithms.common.mani import _totime
        >>> t = np.linspace(0, 10, 101)  # Time array
        >>> tf = [2.5, 7.1]  # Target times
        >>> indices = _totime(t, tf)
        >>> t[indices]  # Closest actual times
        array([2.5, 7.1])
        """
        # Convert to absolute values and ensure tf is array
        t = np.abs(t)
        tf = np.atleast_1d(tf)
        
        # Find closest indices
        I = np.empty(tf.shape, dtype=int)
        for k, target in enumerate(tf):
            diff = np.abs(target - t)
            I[k] = np.argmin(diff)
        
        return I


class _ManifoldServices(_ServiceBundleBase):
    
    def __init__(self, manifold: "Manifold", persistence: _ManifoldPersistenceService, dynamics: _ManifoldDynamicsService) -> None:
        super().__init__(manifold)
        self.dynamics = dynamics
        self.persistence = persistence

 
    @classmethod
    def default(cls, manifold: "Manifold") -> "_ManifoldServices":
        return cls(
            domain_obj=manifold,
            dynamics=_ManifoldDynamicsService(manifold),
            persistence=_ManifoldPersistenceService()
        )
    
    @classmethod
    def with_shared_dynamics(cls, dynamics: _ManifoldDynamicsService) -> "_ManifoldServices":
        return cls(
            domain_obj=dynamics.domain_obj,
            dynamics=dynamics,
            persistence=_ManifoldPersistenceService()
        )
    
    @classmethod
    def for_loading(cls, manifold: "Manifold") -> "_ManifoldServices":
        """Create services for loading operations that don't need dynamics adapter."""
        return cls(
            domain_obj=manifold,
            dynamics=None,
            persistence=_ManifoldPersistenceService()
        )
