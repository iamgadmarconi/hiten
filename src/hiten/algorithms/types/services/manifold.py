"""Adapters backing manifold propagation, stability, and persistence services."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from hiten.algorithms.common.energy import _max_rel_energy_error
from hiten.algorithms.connections.types import ConnectionDomainPayload
from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.linalg.base import StabilityPipeline
from hiten.algorithms.linalg.config import EigenDecompositionConfig
from hiten.algorithms.linalg.types import _ProblemType, _SystemType
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import Trajectory
from hiten.utils.io.manifold import load_manifold, save_manifold
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.algorithms.linalg.options import EigenDecompositionOptions
    from hiten.system.base import System
    from hiten.system.libration import LibrationPoint
    from hiten.system.manifold import Manifold
    from hiten.system.orbits import PeriodicOrbit


class _ManifoldPersistenceService(_PersistenceServiceBase):
    """Persistence helpers for manifold objects.
    
    Parameters
    ----------
    save_fn : Callable[[Manifold, Path, Any], None]
        The function to save the manifold.
    load_fn : Callable[[Path, Any], Manifold]
        The function to load the manifold.
    """

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda manifold, path, **kw: save_manifold(manifold, Path(path), **kw),
            load_fn=lambda path, **kw: load_manifold(Path(path), **kw),
        )


class _ManifoldDynamicsService(_DynamicsServiceBase):
    """Manage STM computation and manifold trajectory generation.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.manifold.Manifold`
        The manifold.
    
    Attributes
    ----------
    stable : int
        The stability of the manifold.
    direction : int
        The direction of the manifold.
    forward : int
        The forward direction of the manifold.
    manifold_result : Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]
        The manifold result.
    generator : :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
        The stability pipeline.
    eigendecomposition_config : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
        Compile-time configuration for eigenvalue decomposition.
    eigendecomposition_options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
        Runtime options for eigenvalue decomposition.
    """

    def __init__(self, manifold: "Manifold") -> None:
        super().__init__(manifold)

        self._stable = 1 if self.domain_obj._stable else -1
        self._direction = 1 if self.domain_obj._direction == "positive" else -1
        self._forward = - self._stable
        self._manifold_result = None

        self._generator = None
        self._eigendecomposition_config = None
        self._eigendecomposition_options = None

    @property
    def generator(self) -> StabilityPipeline:
        """The stability pipeline."""
        if self._generator is None:
            self._generator = StabilityPipeline.with_default_engine(config=self.eigendecomposition_config)
        return self._generator

    @property
    def orbit(self) -> "PeriodicOrbit":
        """The generatingorbit of the manifold."""
        return self.domain_obj._generating_orbit

    @property
    def period(self) -> float:
        """The period of the orbit."""
        return self.orbit.period

    @property
    def stable(self) -> bool:
        """The stability of the manifold."""
        return self._stable

    @property
    def direction(self) -> bool:
        """The direction of the manifold."""
        return self._direction

    @property
    def forward(self) -> bool:
        """The forward direction of the manifold."""
        return self._forward

    @property
    def libration_point(self) -> "LibrationPoint":
        """The libration point of the manifold."""
        return self.orbit.libration_point

    @property
    def system(self) -> "System":
        """The system of the manifold."""
        return self.libration_point.system

    @property
    def mu(self) -> float:
        """The mu of the system."""
        return self.system.mu

    @property
    def dynsys(self) -> _DynamicalSystem:
        """The dynsys of the system."""
        return self.system.dynsys
    
    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """The var_dynsys of the system."""
        return self.system.var_dynsys
    
    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """The jacobian_dynsys of the system."""
        return self.system.jacobian_dynsys

    @property
    def stm(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The stm of the manifold."""
        return self.compute_stm(steps=2000)

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The eigenvalues of the system."""
        return self.stability.eigenvalues

    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The eigenvectors of the system."""
        return self.stability.eigenvectors

    @property
    def sn(self) -> np.ndarray:
        """The stable eigenvectors of the system."""
        return self.stability.stable

    @property
    def un(self) -> np.ndarray:
        """The unstable eigenvectors of the system."""
        return self.stability.unstable

    @property
    def cn(self) -> np.ndarray:
        """The center eigenvectors of the system."""
        return self.stability.center
    
    @property
    def wsn(self) -> np.ndarray:
        """The real stable eigenvectors of the system."""
        return self.stability.Ws

    @property
    def wun(self) -> np.ndarray:
        """The real unstable eigenvectors of the system."""
        return self.stability.Wu

    @property
    def wcn(self) -> np.ndarray:
        """The complex center eigenvectors of the system."""
        return self.stability.Wc        

    @property
    def stability(self) -> StabilityPipeline:
        """The stability of the manifold."""
        return self.compute_stability()

    @property
    def manifold_result(self) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]:
        """The manifold result."""
        return self._manifold_result

    @property
    def trajectories(self) -> List[Trajectory]:
        """The trajectories of the manifold."""
        if self._manifold_result is None:
            return None
        states_list = self._manifold_result[2]
        times_list = self._manifold_result[3]
        return [Trajectory(times, states) for times, states in zip(times_list, states_list)]

    def compute_stm(
        self,
        *,
        steps: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The stm of the manifold.
        
        This function computes the state transition matrix (STM) of the manifold.

        Parameters
        ----------
        steps : int
            The number of steps to take.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The stm of the manifold.
        """
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
        """The main kernel to compute the manifold.
        
        Parameters
        ----------
        step : float
            The step size.
        integration_fraction : float
            The integration fraction.
        NN : int
            The index of the real eigenvector to follow.
        displacement : float
            The displacement magnitude in the eigenvector direction.
        method : str
            The integration method to use.
        order : int
            The order of the integration method to use.
        dt : float
            The time step.
        energy_tol : float
            The energy deviation tolerance for which to discard a trajectory.
        safe_distance : float
            The safe distance from the nearest primary for which to discard a trajectory.
        show_progress : bool
            Whether to show a progress bar.

        Returns
        -------
        Tuple[float, float, List[np.ndarray], List[np.ndarray], int, int]
            The manifold result.
        """
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
                times, states = sol.times, sol.states

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

    def compute_stability(self, options: "EigenDecompositionOptions" = None) -> StabilityPipeline:
        """Compute the stability of the manifold.
        
        Parameters
        ----------
        options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`, optional
            Runtime options for eigenvalue decomposition. If None, uses self.eigendecomposition_options.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
            The stability pipeline with computed eigenvalue decomposition.
        """
        # Use self.eigendecomposition_options if options not provided
        if options is None:
            options = self.eigendecomposition_options
            
        key = self.make_key(id(self.domain_obj), tuple(sorted(options.to_dict().items())))
        
        def _factory() -> StabilityPipeline:
            _, _, phi_T, _ = self.compute_stm(steps=2000)
            self.generator.compute(domain_obj=phi_T, options=options)
            return self.generator
        
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
        """Compute the manifold section.
        
        Parameters
        ----------
        period : float
            The period of the orbit.
        fraction : float
            The fraction of the period to compute the manifold section.
        displacement : float
            The displacement magnitude in the eigenvector direction.
        xx : np.ndarray
            The stm.
        tt : np.ndarray
            The times.
        PHI : np.ndarray
            The stm.
        eigvec : np.ndarray
            The eigenvector.

        Returns
        -------
        np.ndarray
            The manifold section.
        """
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

    @property
    def eigendecomposition_config(self) -> EigenDecompositionConfig:
        """The eigen decomposition configuration.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            The eigendecomposition configuration with reasonable defaults.
        """
        if self._eigendecomposition_config is None:
            self._eigendecomposition_config = EigenDecompositionConfig(
                system_type=_SystemType.DISCRETE,
                problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
            )
        return self._eigendecomposition_config
    
    @eigendecomposition_config.setter
    def eigendecomposition_config(self, value: EigenDecompositionConfig) -> None:
        """Set the eigen decomposition configuration.
        
        Invalidates the generator cache to trigger recreation with the new config.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            New eigendecomposition configuration.
        """
        self._eigendecomposition_config = value
        self._generator = None  # Invalidate cache to trigger recreation
    
    @property
    def eigendecomposition_options(self) -> "EigenDecompositionOptions":
        """Runtime options for eigenvalue decomposition.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            The eigendecomposition options with reasonable defaults.
        """
        if self._eigendecomposition_options is None:
            from hiten.algorithms.linalg.options import \
                EigenDecompositionOptions
            self._eigendecomposition_options = EigenDecompositionOptions(
                delta=1e-6,
                tol=1e-6,
            )
        return self._eigendecomposition_options
    
    @eigendecomposition_options.setter
    def eigendecomposition_options(self, value: "EigenDecompositionOptions") -> None:
        """Set runtime options for eigenvalue decomposition.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            New eigendecomposition options.
        """
        self._eigendecomposition_options = value

    def apply_connections(self, payload: ConnectionDomainPayload) -> None:
        """Apply connection results payload to the manifold domain object."""
        self.domain_obj._connection_results = tuple(payload.connections)


class _ManifoldServices(_ServiceBundleBase):
    """Bundle all manifold services together.
    
    Parameters
    ----------
    manifold : :class:`~hiten.system.manifold.Manifold`
        The manifold.
    persistence : :class:`~hiten.algorithms.types.services.manifold._ManifoldPersistenceService`
        The persistence service.
    dynamics : :class:`~hiten.algorithms.types.services.manifold._ManifoldDynamicsService`
        The dynamics service.
    """
    
    def __init__(self, manifold: "Manifold", persistence: _ManifoldPersistenceService, dynamics: _ManifoldDynamicsService) -> None:
        super().__init__(manifold)
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, manifold: "Manifold") -> "_ManifoldServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        manifold : :class:`~hiten.system.manifold.Manifold`
            The manifold.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.manifold._ManifoldServices`
            The service bundle.
        """
        return cls(
            manifold,
            _ManifoldPersistenceService(),
            _ManifoldDynamicsService(manifold)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _ManifoldDynamicsService) -> "_ManifoldServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.manifold._ManifoldDynamicsService`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.manifold._ManifoldServices`
            The service bundle.
        """
        return cls(
            dynamics.domain_obj,
            _ManifoldPersistenceService(),
            dynamics
        )