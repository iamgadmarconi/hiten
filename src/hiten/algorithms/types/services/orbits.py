"""Adapter helpers orchestrating periodic-orbit numerics and persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.common.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.continuation.base import StateParameter
from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.algorithms.corrector.base import Corrector
from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_monodromy, _compute_stm
from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicStateVector,
                                           Trajectory)
from hiten.system.manifold import Manifold
from hiten.utils.io.orbits import (load_periodic_orbit,
                                   load_periodic_orbit_inplace,
                                   save_periodic_orbit)

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.orbits.base import PeriodicOrbit, GenericOrbit


class _OrbitPersistenceService(_PersistenceServiceBase):
    """Thin wrapper around orbit persistence helpers."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda orbit, path, **kw: save_periodic_orbit(orbit, Path(path), **kw),
            load_fn=lambda path, **kw: load_periodic_orbit(Path(path), **kw),
            load_inplace_fn=lambda orbit, path, **kw: load_periodic_orbit_inplace(orbit, path, **kw),
        )


class _OrbitCorrectionService(_DynamicsServiceBase):
    """Drive Newton-based differential correction for periodic orbits."""

    def __init__(self, domain_obj: "PeriodicOrbit") -> None:
        super().__init__(domain_obj)
        self._corrector = Corrector.with_default_engine(config=self.correction_config)

    def correct(self, *, overrides: Dict[str, Any] | None = None) -> Tuple[np.ndarray, float]:
        """Differential correction wrapper."""
        cache_key = self.make_key("correct", overrides)

        def _factory() -> Tuple[np.ndarray, float]:
            if overrides:
                override = True
            results = self.corrector.correct(self._domain_obj, override=override, **overrides)

            return results.corrected_state, 2 * results.half_period

        return self.get_or_create(cache_key, _factory)

    def update_correction(self, **kwargs) -> None:
        """Update algorithm-level correction parameters for this orbit.

        Allowed keys: tol, max_attempts, max_delta, line_search_config,
        finite_difference, forward.
        """
        self.corrector.update_config(**kwargs)

    @property
    def corrector(self) -> Corrector:
        return self._corrector

    @property
    @abstractmethod
    def correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for this orbit family."""
        pass


class _OrbitContinuationService(_DynamicsServiceBase):
    """Drive continuation for periodic orbits."""

    def __init__(self, domain_obj: "PeriodicOrbit") -> None:
        super().__init__(domain_obj)
        self._generator = StateParameter.with_default_engine(config=self.continuation_config)

    @property
    def generator(self) -> StateParameter:
        return self._generator

    def generate(self, *, overrides: Dict[str, Any] | None = None) -> Tuple[np.ndarray, float]:
        """Generate a family of periodic orbits."""
        cache_key = self.make_key("generate", overrides)

        def _factory() -> Tuple[np.ndarray, float]:
            if overrides:
                override = True
            results = self.generator.generate(self._domain_obj, override=override, **overrides)

            return results

        return self.get_or_create(cache_key, _factory)

    def update_continuation(self, **kwargs) -> None:
        """Update algorithm-level continuation parameters for this orbit.

        Allowed keys: tol, max_attempts, max_delta, line_search_config,
        finite_difference, forward.
        """
        self.generator.update_config(**kwargs)

    @property
    @abstractmethod
    def continuation_config(self) -> "_OrbitContinuationConfig":
        """Default parameter for family continuation (must be overridden)."""
        pass


class _OrbitDynamicsService(ABC, _DynamicsServiceBase):
    """Integrate periodic orbits using the system dynamics."""

    def __init__(self, orbit: "PeriodicOrbit", *, initial_state: np.ndarray | None = None) -> None:
        super().__init__(orbit)

        self._initial_state = initial_state
    
        if self._initial_state is not None:
            self._initial_state = np.asarray(initial_state, dtype=np.float64)
        else:
            self._initial_state = self._initial_guess()

        self._period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
        self._correction_overrides: dict[str, object] = {}

    @property
    def orbit(self) -> PeriodicOrbit:
        return self._domain_obj

    @property
    def libration_point(self) -> LibrationPoint:
        return self.orbit.libration_point

    @property
    def system(self) -> System:
        return self.orbit.system

    @property
    def mu(self) -> float:
        return self.system.mu

    @property
    def is_stable(self) -> bool:
        """
        Check if the orbit is linearly stable.
        
        Returns
        -------
        bool
            True if all stability indices have magnitude <= 1, False otherwise.
        """
        if self._stability_info is None:
            self.compute_stability()
        
        indices = self.stability_indices
        
        # An orbit is stable if all stability indices have magnitude <= 1
        return np.all(np.abs(indices) <= 1.0)

    @property
    def stability_indices(self) -> Optional[Tuple]:
        if self._stability_info is None:
            self.compute_stability()
        return self._stability_info[0]
    
    @property
    def eigenvalues(self) -> Optional[Tuple]:
        if self._stability_info is None:
            self.compute_stability()
        return self._stability_info[1]
    
    @property
    def eigenvectors(self) -> Optional[Tuple]:
        if self._stability_info is None:
            self.compute_stability()
        return self._stability_info[2]

    @property
    def energy(self) -> float:
        """
        Compute the energy of the orbit at the initial state.
        
        Returns
        -------
        float
            The energy value in nondimensional units.
        """
        energy_val = crtbp_energy(self.initial_state, self.mu)
        return energy_val
    
    @property
    def jacobi_constant(self) -> float:
        """
        Compute the Jacobi constant of the orbit.
        
        Returns
        -------
        float
            The Jacobi constant value (dimensionless).
        """
        return energy_to_jacobi(self.energy)

    @property
    def dynsys(self) -> _DynamicalSystem:
        """Underlying vector field instance.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying vector field instance.
        """
        return self.system.dynsys

    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """Underlying variational equations system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying variational equations system.
        """
        return self.system.var_dynsys

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """Underlying Jacobian evaluation system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying Jacobian evaluation system.
        """
        return self.system.jacobian_dynsys

    @property
    def initial_state(self) -> np.ndarray:
        return self._initial_state

    @property
    def period(self) -> float:
        return self._period

    @period.setter
    def period(self, value: Optional[float]):
        """Set the orbit period and invalidate cached data.

        Setting the period manually allows users (or serialization logic)
        to override the value obtained via differential correction. Any time
        the period changes we must invalidate cached trajectory, time array
        and stability information so they can be recomputed consistently.

        Parameters
        ----------
        value : float or None
            The orbit period in nondimensional units, or None to clear.

        Raises
        ------
        ValueError
            If value is not positive.
        """

        if value is not None and value <= 0:
            raise ValueError("period must be a positive number or None.")

        if value != self.period:

            self._period = value


            # Also invalidate service attributes and caches that depend on period
            self._trajectory = None
            self._stability_info = None
            self.reset()

    @property
    def trajectory(self) -> Optional[Trajectory]:
        """
        Get the computed trajectory points.

        Returns
        -------
        Trajectory or None
            Array of shape (steps, 6) containing state vectors at each time step,
            or None if the trajectory hasn't been computed yet.
        """
        if self._trajectory is None:
            raise ValueError("Trajectory not computed. Call propagate() first.")
        return self._trajectory

    @property
    def monodromy(self):
        if self.initial_state is None:
            raise ValueError("Initial state must be provided")

        if self.period is None:
            raise ValueError("Period must be set before computing monodromy")

        cache_key = self.make_key("monodromy")

        def _factory() -> np.ndarray:
            return _compute_monodromy(self.var_dynsys, self.initial_state, self.period)

        return self.get_or_create(cache_key, _factory)

    def propagate(self, *, steps: int, method: str, order: int) -> Trajectory:
        if self._initial_state is None:
            raise ValueError("Initial state must be provided")

        if self._period is None:
            raise ValueError("Period must be set before propagation")

        cache_key = self.make_key("propagate", steps, method, order)

        def _factory() -> Trajectory:
            sol = _propagate_dynsys(
                dynsys=self.system.dynsys,
                state0=self.initial_state,
                t0=0.0,
                tf=self._period,
                forward=1,
                steps=steps,
                method=method,
                order=order,
            )

            traj = Trajectory.from_solution(
                sol,
                state_vector_cls=SynodicStateVector,
                frame=ReferenceFrame.ROTATING,
            )
            self._trajectory = traj
            return traj

        return self.get_or_create(cache_key, _factory)

    def manifold(self, stable: bool = True, direction: Literal["positive", "negative"] = "positive") -> "Manifold":
        return Manifold(self.orbit, stable=stable, direction=direction)

    def compute_stability(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.initial_state is None:
            raise ValueError("Initial state must be provided")

        if self.period is None:
            raise ValueError("Period must be set before computing stability")

        cache_key = self.make_key("stability")

        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            _, _, Phi, _ = _compute_stm(self.var_dynsys, self.initial_state, self.period)
            backend = _LinalgBackend()
            indices, eigvals, eigvecs = backend.stability_indices(Phi)
            self._stability_info = (indices, eigvals, eigvecs)
            return indices, eigvals, eigvecs

        return self.get_or_create(cache_key, _factory)

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit."""
        pass

    @amplitude.setter
    def amplitude(self, value: float):
        """Set the orbit amplitude."""
        self._amplitude = value

    @abstractmethod
    def _initial_guess(self) -> np.ndarray:
        pass


class _GenericOrbitCorrectionService(_OrbitCorrectionService):
    """Drive Newton-based differential correction for generic orbits."""

    def __init__(self, orbit: "GenericOrbit") -> None:
        super().__init__(orbit)
        self._corrector = Corrector.with_default_engine(config=self.correction_config)

    @property
    def correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for generic orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The correction configuration.
            
        Raises
        ------
        NotImplementedError
            If correction_config is not set on the orbit.
        """
        if self.orbit.correction_config is not None:
            return self.orbit.correction_config
        raise NotImplementedError(
            "Differential correction is not defined for a GenericOrbit unless the "
            "`correction_config` property is set with a valid :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`."
        )

    @correction_config.setter
    def correction_config(self, value: "_OrbitCorrectionConfig"):
        """Set the correction configuration."""
        from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
        if value is not None and not isinstance(value, _OrbitCorrectionConfig):
            raise TypeError("correction_config must be an instance of _OrbitCorrectionConfig or None.")
        self.orbit.correction_config = value


class _GenericOrbitContinuationService(_OrbitContinuationService):
    """Drive continuation for generic orbits."""

    def __init__(self, orbit: "GenericOrbit") -> None:
        super().__init__(orbit)
        self._generator = StateParameter.with_default_engine(config=self.continuation_config)

    @property
    def continuation_config(self) -> "_OrbitContinuationConfig":
        """Provides the continuation configuration for generic orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The continuation configuration.
            
        Raises
        ------
        NotImplementedError
            If continuation_config is not set on the orbit.
        """
        if self.orbit.continuation_config is not None:
            return self.orbit.continuation_config
        raise NotImplementedError(
            "GenericOrbit requires 'continuation_config' to be set before using continuation engines."
        )

    @continuation_config.setter
    def continuation_config(self, value: "_OrbitContinuationConfig"):
        """Set the continuation configuration."""
        from hiten.algorithms.continuation.config import _OrbitContinuationConfig
        if value is not None and not isinstance(value, _OrbitContinuationConfig):
            raise TypeError("continuation_config must be an instance of _OrbitContinuationConfig or None.")
        self.orbit.continuation_config = value


class _GenericOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for generic orbits with custom amplitude handling."""

    def __init__(self, orbit: "GenericOrbit", *, initial_state: np.ndarray | None = None) -> None:
        super().__init__(orbit, initial_state=initial_state)
        self._amplitude = None

    @property
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit.
        
        Returns
        -------
        float or None
            The orbit amplitude in nondimensional units, or None if not set.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        """Set the orbit amplitude.
        
        Parameters
        ----------
        value : float
            The orbit amplitude in nondimensional units.
        """
        self._amplitude = value

    def _initial_guess(self) -> np.ndarray:
        """Generate initial guess for GenericOrbit.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments (unused).
            
        Returns
        -------
        numpy.ndarray, shape (6,)
            The initial state vector in nondimensional units.
            
        Raises
        ------
        ValueError
            If no initial state is provided.
        """
        # Check if the orbit has an initial state set
        if hasattr(self.orbit, '_initial_state') and self.orbit._initial_state is not None:
            return np.asarray(self.orbit._initial_state, dtype=np.float64)
        raise ValueError("No initial state provided for GenericOrbit.")



@dataclass
class _OrbitServices(_ServiceBundleBase):
    domain_obj: "PeriodicOrbit"
    correction: _OrbitCorrectionService
    continuation: _OrbitContinuationService
    dynamics: _OrbitDynamicsService
    persistence: _OrbitPersistenceService

    @classmethod
    def default(cls, orbit: "PeriodicOrbit", *, initial_state = Optional[np.ndarray] = None) -> "_OrbitServices":
        return cls(
            domain_obj=orbit,
            correction=_OrbitCorrectionService(orbit),
            continuation=_OrbitContinuationService(orbit),
            dynamics=_OrbitDynamicsService(orbit, initial_state=initial_state),
            persistence=_OrbitPersistenceService()
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _OrbitDynamicsService) -> "_OrbitServices":
        return cls(
            domain_obj=dynamics._domain_obj,
            correction=_OrbitCorrectionService(dynamics._domain_obj),
            continuation=_OrbitContinuationService(dynamics._domain_obj),
            dynamics=dynamics,
            persistence=_OrbitPersistenceService()
        )
