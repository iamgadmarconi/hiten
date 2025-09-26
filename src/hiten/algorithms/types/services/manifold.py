"""Adapters backing manifold propagation, stability, and persistence services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hiten.algorithms.common.energy import _max_rel_energy_error
from hiten.algorithms.common.mani import _totime
from hiten.algorithms.dynamics.base import _propagate_dynsys
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
    from hiten.system.manifold import Manifold


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
        self._domain_obj = manifold
        self._orbit = self._domain_obj.generating_orbit
        self._system = self._orbit.system
        self._forward = - self._domain_obj.stable

    def compute_stm(
        self,
        *,
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_key = self.make_key(id(self._orbit), steps, self._forward)
        
        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            return _compute_stm(
                self._system.var_dynsys,
                self._orbit.initial_state,
                self._orbit.period,
                steps=steps,
                forward=self._forward,
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
            id(self._orbit),
            self._domain_obj.stable,
            self._domain_obj.direction,
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
            return self._run_compute(
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

        return self.get_or_create(cache_key, _factory)

    def reset(self) -> None:
        super().reset()

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
        orbit = self._orbit
        mu = self._system.mu
        forward = self._forward

        dist_m = orbit.system.distance * 1e3
        pr_nd = orbit.system.primary.radius / dist_m
        sr_nd = orbit.system.secondary.radius / dist_m
        safe_r1 = safe_distance * pr_nd
        safe_r2 = safe_distance * sr_nd

        stability = self.compute_stability()
        sn, un, _ = stability.eigenvalues
        Ws, Wu, _ = stability.eigenvectors

        _, snreal_vecs = stability.get_real_eigenvectors(Ws, sn)
        _, unreal_vecs = stability.get_real_eigenvectors(Wu, un)  

        col_idx = NN - 1
        if self._domain_obj.stable == 1:
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
                    dynsys=self._system.dynsys,
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
        key = self.make_key(id(self._domain_obj))
        
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
        mfrac = _totime(tt, fraction * period)

        if np.isscalar(mfrac):
            mfrac_idx = mfrac
        else:
            mfrac_idx = mfrac[0]

        phi_frac_flat = PHI[mfrac_idx, :36]
        phi_frac = phi_frac_flat.reshape((6, 6))

        MAN = self._domain_obj.direction * (phi_frac @ eigvec)

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


@dataclass
class _ManifoldServices(_ServiceBundleBase):
    domain_obj: "Manifold"
    dynamics: _ManifoldDynamicsService
    persistence: _ManifoldPersistenceService

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
            domain_obj=dynamics._domain_obj,
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
