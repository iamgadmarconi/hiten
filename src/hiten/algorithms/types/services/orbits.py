"""Adapter helpers orchestrating periodic-orbit numerics and persistence."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.common.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.continuation.base import ContinuationPipeline
from hiten.algorithms.continuation.types import (ContinuationDomainPayload,
                                                 ContinuationResult)
from hiten.algorithms.corrector.base import CorrectorPipeline
from hiten.algorithms.corrector.types import (CorrectionResult,
                                              OrbitCorrectionDomainPayload)
from hiten.algorithms.dynamics.base import _DynamicalSystem, _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_monodromy, _compute_stm
from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.poincare.singlehit.backend import (_y_plane_crossing,
                                                         _z_plane_crossing)
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicState,
                                           SynodicStateVector, Trajectory)
from hiten.system.manifold import Manifold
from hiten.utils.io.orbits import (load_periodic_orbit,
                                   load_periodic_orbit_inplace,
                                   save_periodic_orbit)

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import OrbitContinuationConfig
    from hiten.algorithms.continuation.options import OrbitContinuationOptions
    from hiten.algorithms.continuation.types import ContinuationResult
    from hiten.algorithms.corrector.config import OrbitCorrectionConfig
    from hiten.algorithms.corrector.options import OrbitCorrectionOptions
    from hiten.system.base import System
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.orbits.base import GenericOrbit, PeriodicOrbit
    from hiten.system.orbits.halo import HaloOrbit
    from hiten.system.orbits.lissajous import LissajousOrbit
    from hiten.system.orbits.lyapunov import LyapunovOrbit
    from hiten.system.orbits.vertical import VerticalOrbit

class _OrbitPersistenceService(_PersistenceServiceBase):
    """Thin wrapper around orbit persistence helpers.
    
    Parameters
    ----------
    save_fn : Callable[..., Any]
        The function to save the object.
    load_fn : Callable[..., Any]
        The function to load the object.
    load_inplace_fn : Callable[..., Any]
        The function to load the object in place.
    """

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda orbit, path, **kw: save_periodic_orbit(orbit, Path(path), **kw),
            load_fn=lambda path, **kw: load_periodic_orbit(Path(path), **kw),
            load_inplace_fn=lambda orbit, path, **kw: load_periodic_orbit_inplace(orbit, path, **kw),
        )


class _OrbitCorrectionService(_DynamicsServiceBase):
    """Drive Newton-based differential correction for periodic orbits.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The domain object.

    Attributes
    ----------
    corrector : :class:`~hiten.algorithms.corrector.base.CorrectorPipeline`
        The corrector.
    """

    def __init__(self, domain_obj: "PeriodicOrbit") -> None:
        super().__init__(domain_obj)
        self._corrector = None
        self._correction_config = None
        self._correction_options = None

    @property
    def corrector(self) -> CorrectorPipeline:
        """The corrector.
        
        Lazily creates a corrector pipeline with the appropriate interface
        and backend based on the config type (single shooting vs multiple shooting).
        """
        if self._corrector is None:
            config = self.correction_config

            from hiten.algorithms.corrector.backends.newton import \
                _NewtonBackend
            from hiten.algorithms.corrector.interfaces import \
                _OrbitCorrectionInterface
            from hiten.algorithms.corrector.stepping import make_armijo_stepper
            
            interface = _OrbitCorrectionInterface()
            backend = _NewtonBackend(stepper_factory=make_armijo_stepper())
        
            self._corrector = CorrectorPipeline.with_default_engine(
                config=config,
                interface=interface,
                backend=backend
            )
        return self._corrector

    def correct(
        self, *, options: "OrbitCorrectionOptions" = None
    ) -> tuple[np.ndarray, float, "CorrectionResult"]:
        """Differential correction wrapper.
        
        Parameters
        ----------
        options : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`, optional
            Runtime options for correction. If None, uses default options from
            the orbit's `correction_options` property.

        Returns
        -------
        tuple[np.ndarray, float, :class:`~hiten.algorithms.corrector.types.CorrectionResult`]
            Corrected state, full period, and the full correction result object.
        """
        # Use provided options or get defaults
        if options is None:
            options = self.correction_options
        
        # Cache key based on options
        cache_key = self.make_key("correct", tuple(sorted(options.to_dict().items())))

        def _factory() -> tuple[np.ndarray, float, OrbitCorrectionDomainPayload, "CorrectionResult"]:
            result = self.corrector.correct(self.domain_obj, options=options)
            payload = OrbitCorrectionDomainPayload._from_mapping(
                {
                    "x_full": result.x_corrected,
                    "half_period": result.half_period,
                    "iterations": result.iterations,
                    "residual_norm": result.residual_norm,
                }
            )
            self.apply_correction(payload)
            return result.x_corrected, 2 * result.half_period, payload, result

        state, period, payload, result = self.get_or_create(cache_key, _factory)
        return state, period, result

    def apply_correction(self, update: OrbitCorrectionDomainPayload) -> OrbitCorrectionDomainPayload:
        """Apply a correction update to the orbit dynamics and return the payload."""

        payload = update
        x_full = np.asarray(payload.x_full, dtype=float)
        half_period = float(payload.half_period)

        self.domain_obj.dynamics.reset()
        self.domain_obj.dynamics._initial_state = x_full
        self.domain_obj.dynamics.period = 2.0 * half_period

        return payload

    @property
    def correction_options(self):
        """Get the correction options for this orbit.
        
        Returns the stored options, lazily initializing with defaults if needed.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            The correction options.
        """
        if self._correction_options is None:
            self._correction_options = self._default_correction_options()
        return self._correction_options

    @abstractmethod
    def _default_correction_options(self):
        """Create default correction options for this orbit family.
        
        Concrete implementations should override this to provide family-specific defaults.
        """
        pass

    @correction_options.setter
    def correction_options(self, value: "OrbitCorrectionOptions"):
        """Set the correction options.
        
        Stores the options and invalidates the corrector cache.
        """
        self._correction_options = value
        self._corrector = None  # Invalidate cache

    @property
    def correction_config(self) -> "OrbitCorrectionConfig":
        """Provides the differential correction configuration for this orbit family.
        
        Returns the stored config, lazily initializing with defaults if needed.
        """
        if self._correction_config is None:
            self._correction_config = self._default_correction_config()
        return self._correction_config

    @abstractmethod
    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Create the default correction configuration for this orbit family.
        
        Concrete implementations should override this to provide family-specific defaults.
        """
        pass

    @correction_config.setter
    def correction_config(self, value: "OrbitCorrectionConfig"):
        """Set the correction configuration.
        
        Stores the config and invalidates the corrector cache to trigger 
        recreation with the appropriate interface.
        """
        self._correction_config = value
        self._corrector = None  # Invalidate cache to trigger recreation


class _OrbitContinuationService(_DynamicsServiceBase):
    """Drive continuation for periodic orbits.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The domain object.

    Attributes
    ----------
    generator : :class:`~hiten.algorithms.continuation.base.ContinuationPipeline`
        The generator.
    """

    def __init__(self, domain_obj: "PeriodicOrbit") -> None:
        super().__init__(domain_obj)
        self._generator = None
        self._continuation_config = None
        self._continuation_options = None

    @property
    def initial_state(self) -> np.ndarray:
        """The initial state from the dynamics service."""
        return self.domain_obj.dynamics.initial_state

    @property
    def generator(self) -> ContinuationPipeline:
        """The continuation pipeline."""
        if self._generator is None:
            self._generator = ContinuationPipeline.with_default_engine(config=self.continuation_config)
        return self._generator

    def generate(self, *, options: "OrbitContinuationOptions" = None) -> ContinuationResult:
        """Generate a family of periodic orbits.
        
        Parameters
        ----------
        options : :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`, optional
            Runtime options for continuation. If None, uses default options from
            the orbit's `continuation_options` property.

        Returns
        -------
        ContinuationResult
            The continuation result.
        """
        # Use provided options or get defaults
        if options is None:
            options = self.continuation_options
        
        # Cache key based on options
        cache_key = self.make_key("generate", tuple(sorted(options.to_dict().items())))

        def _factory() -> ContinuationDomainPayload:
            result = self.generator.generate(self.domain_obj, options)
            payload = ContinuationDomainPayload._from_mapping(
                {
                    "family": result.family,
                    "family_repr": tuple(
                        getattr(member, "initial_state", None)
                        for member in result.family
                    ),
                    "accepted_count": result.accepted_count,
                    "rejected_count": result.rejected_count,
                    "iterations": result.iterations,
                    "success_rate": result.success_rate,
                    "parameter_values": result.parameter_values,
                    "info": {},
                }
            )
            self.apply_continuation(payload)
            return payload

        payload = self.get_or_create(cache_key, _factory)
        return ContinuationResult(
            accepted_count=payload.accepted_count,
            rejected_count=payload.rejected_count,
            success_rate=payload.success_rate,
            family=payload.family,
            parameter_values=payload.parameter_values,
            iterations=payload.iterations,
        )

    def apply_continuation(self, payload: ContinuationDomainPayload) -> ContinuationDomainPayload:
        representatives = payload.family_repr
        new_family = [self.domain_obj]
        for repr_vec in representatives[1:]:
            new_orbit = self.domain_obj.__class__(
                libration_point=self.domain_obj.libration_point,
                initial_state=np.asarray(repr_vec, dtype=float),
            )
            new_orbit.period = getattr(self.domain_obj, "period", None)
            new_family.append(new_orbit)
        self._continuation_overrides = {
            "family": tuple(new_family),
            "parameter_values": payload.parameter_values,
        }
        return payload

    @property
    def continuation_config(self) -> "OrbitContinuationConfig":
        """Get the continuation configuration for this orbit family.
        
        Returns the stored config, lazily initializing with defaults if needed.
        """
        if self._continuation_config is None:
            self._continuation_config = self._default_continuation_config()
        return self._continuation_config

    @abstractmethod
    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Create the default continuation configuration for this orbit family.
        
        Concrete implementations should override this to provide family-specific defaults.
        """
        pass

    @continuation_config.setter
    def continuation_config(self, value: "OrbitContinuationConfig"):
        """Set the continuation configuration.
        
        Stores the config and invalidates the generator cache.
        """
        self._continuation_config = value
        self._generator = None  # Invalidate cache to trigger recreation

    @property
    def continuation_options(self) -> "OrbitContinuationOptions":
        """Get the continuation options for this orbit family.
        
        Returns the stored options, lazily initializing with defaults if needed.
        """
        if self._continuation_options is None:
            self._continuation_options = self._default_continuation_options()
        return self._continuation_options

    @abstractmethod
    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create default continuation options for this orbit family.
        
        Concrete implementations should override this to provide family-specific defaults.
        """
        pass

    @continuation_options.setter
    def continuation_options(self, value: "OrbitContinuationOptions"):
        """Set the continuation options.
        
        Stores the options and invalidates the generator cache.
        """
        self._continuation_options = value
        self._generator = None  # Invalidate cache


class _OrbitDynamicsService(_DynamicsServiceBase):
    """Integrate periodic orbits using the system dynamics.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The orbit.
    """

    def __init__(self, orbit: "PeriodicOrbit") -> None:
        super().__init__(orbit)

        self._initial_state = self.domain_obj._initial_state
        self._libration_point = self.domain_obj._libration_point

        if self._initial_state is not None:
            self._initial_state = np.asarray(self._initial_state, dtype=np.float64)
        else:
            self._initial_state = self.initial_guess()

        self._period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
        self._correction_overrides: dict[str, object] = {}

    @property
    def orbit(self) -> PeriodicOrbit:
        """The orbit."""
        return self.domain_obj

    @property
    def libration_point(self) -> LibrationPoint:
        """The libration point."""
        return self._libration_point

    @property
    def system(self) -> System:
        """The system."""
        return self.libration_point.system

    @property
    def mu(self) -> float:
        """The mass ratio."""
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
        """The stability indices."""
        if self._stability_info is None:
            self.compute_stability()
        return self._stability_info[0]
    
    @property
    def eigenvalues(self) -> Optional[Tuple]:
        """The eigenvalues."""
        if self._stability_info is None:
            self.compute_stability()
        return self._stability_info[1]
    
    @property
    def eigenvectors(self) -> Optional[Tuple]:
        """The eigenvectors."""
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
        """The initial state."""
        return self._initial_state

    @property
    def period(self) -> float:
        """The period."""
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
    def trajectories(self) -> List[Trajectory]:
        """Compatibility helper for SynodicMap - returns list with single trajectory."""
        if self._trajectory is None:
            raise ValueError("Trajectory not computed. Call propagate() first.")
        return [self._trajectory]

    @property
    def monodromy(self):
        """The monodromy."""
        if self.initial_state is None:
            raise ValueError("Initial state must be provided")

        if self.period is None:
            raise ValueError("Period must be set before computing monodromy")

        cache_key = self.make_key("monodromy")

        def _factory() -> np.ndarray:
            return _compute_monodromy(self.var_dynsys, self.initial_state, self.period)

        return self.get_or_create(cache_key, _factory)

    def propagate(self, *, steps: int, method: str, order: int) -> Trajectory:
        """Propagate the orbit.
        
        Parameters
        ----------
        steps : int
            The number of steps to take.
        method : str
            The method to use for propagation.
        order : int
            The order of the method to use for propagation.
        """
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
        """Create a manifold for the orbit.

        Parameters
        ----------
        stable : bool
            Whether to create a stable manifold.
        direction : Literal["positive", "negative"]
            The direction of the manifold.

        Returns
        -------
        :class:`~hiten.system.manifold.Manifold`
            The manifold.
        """
        return Manifold(self.orbit, stable=stable, direction=direction)

    def compute_stability(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the stability of the orbit.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The stability of the orbit.
        """
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
    def amplitude(self) -> float:
        """The amplitude of the orbit."""
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, value: float):
        """Set the amplitude of the orbit.
        
        Parameters
        ----------
        value : float
            The amplitude of the orbit.
        """
        self._amplitude = value

    @abstractmethod
    def initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the orbit.
        
        Returns
        -------
        np.ndarray
            The initial guess for the orbit.
        """
        pass


class _GenericOrbitCorrectionService(_OrbitCorrectionService):
    """Drive Newton-based differential correction for generic orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.GenericOrbit`
        The orbit.
    """

    def __init__(self, orbit: "GenericOrbit") -> None:
        super().__init__(orbit)

    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Raises error since GenericOrbit has no default config.
        
        Raises
        ------
        NotImplementedError
            GenericOrbit requires explicit config.
        """
        raise NotImplementedError(
            "Differential correction is not defined for a GenericOrbit unless the "
            "`correction_config` property is set with a valid :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`."
        )
    
    def _default_correction_options(self) -> "OrbitCorrectionOptions":
        """Raises error since GenericOrbit has no default options.
        
        Raises
        ------
        NotImplementedError
            GenericOrbit requires explicit options.
        """
        raise NotImplementedError(
            "Correction options are not defined for a GenericOrbit unless explicitly set."
        )


class _GenericOrbitContinuationService(_OrbitContinuationService):
    """Drive continuation for generic orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.GenericOrbit`
        The orbit.
    """

    def __init__(self, orbit: "GenericOrbit") -> None:
        super().__init__(orbit)

    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Provides the continuation configuration for generic orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`
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

    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create default continuation options for generic orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            Default continuation options.
        """
        if self.orbit._continuation_config is not None:
            return self.orbit._continuation_config
        raise NotImplementedError(
            "Continuation is not defined for a GenericOrbit unless the "
            "`continuation_config` property is set with a valid :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`."
        )


class _GenericOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for generic orbits with custom amplitude handling.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.GenericOrbit`
        The orbit.
    """

    def __init__(self, orbit: "GenericOrbit") -> None:
        super().__init__(orbit)

    def initial_guess(self) -> np.ndarray:
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
        if self.orbit._initial_state is not None:
            return np.asarray(self.orbit._initial_state, dtype=np.float64)
        raise ValueError("No initial state provided for GenericOrbit.")



class _HaloOrbitCorrectionService(_OrbitCorrectionService):
    """Drive Newton-based differential correction for halo orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.HaloOrbit`
        The orbit.
    """

    def __init__(self, orbit: "HaloOrbit") -> None:
        super().__init__(orbit)

    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Create the default correction configuration for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The default correction configuration for halo orbits.
        """
        from hiten.algorithms.corrector.config import OrbitCorrectionConfig
        from hiten.algorithms.types.configs import (IntegrationConfig,
                                                    NumericalConfig)
        return OrbitCorrectionConfig(
            event_func=_y_plane_crossing,
            residual_indices=(SynodicState.VX, SynodicState.VZ),
            control_indices=(SynodicState.X, SynodicState.VY),
            target=(0.0, 0.0),
            extra_jacobian=self._halo_quadratic_term,
            integration=IntegrationConfig(
                method="adaptive", 
                forward=1, 
                flip_indices=None
            ),
            numerical=NumericalConfig(
                finite_difference=False,
                line_search_enabled=True,
            ),
        )

    def _default_correction_options(self) -> "OrbitCorrectionOptions":
        """Create the default correction options for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            The default correction options for halo orbits.
        """
        from hiten.algorithms.corrector.options import OrbitCorrectionOptions
        from hiten.algorithms.types.options import (ConvergenceOptions,
                                                    CorrectionOptions,
                                                    IntegrationOptions,
                                                    NumericalOptions)
        
        return OrbitCorrectionOptions(
            base=CorrectionOptions(
                convergence=ConvergenceOptions(
                    max_attempts=50,
                    tol=1e-12,
                    max_delta=1e-2,
                ),
                integration=IntegrationOptions(
                    dt=1e-2,
                    order=8,
                    max_steps=2000,
                    c_omega_heuristic=20.0,
                    steps=500,
                ),
                numerical=NumericalOptions(
                    fd_step=1e-8,
                    line_search_alpha_reduction=0.5,
                    line_search_min_alpha=1e-4,
                    line_search_armijo_c=0.1,
                ),
            ),
            forward=1,
        )

    def _halo_quadratic_term(self, X_ev, Phi):
        """
        Evaluate the quadratic part of the Jacobian for differential correction.

        Parameters
        ----------
        X_ev : numpy.ndarray, shape (6,)
            State vector at the event time (half-period) in nondimensional units.
        Phi : numpy.ndarray
            State-transition matrix evaluated at the same event.
            
        Returns
        -------
        numpy.ndarray, shape (2, 2)
            Reduced Jacobian matrix employed by the
            :meth:`~hiten.system.orbits.base.PeriodicOrbit.correct`
            solver.
        """
        x, y, z, vx, vy, vz = X_ev
        mu = self.domain_obj.mu
        mu2 = 1 - mu
        rho_1 = 1/(((x+mu)**2 + y**2 + z**2)**1.5)
        rho_2 = 1/(((x-mu2 )**2 + y**2 + z**2)**1.5)
        omega_x  = -(mu2*(x+mu)*rho_1) - (mu*(x-mu2)*rho_2) + x
        DDx = 2*vy + omega_x
        DDz = -(mu2*z*rho_1) - (mu*z*rho_2)

        if abs(vy) < 1e-9:
            vy = np.sign(vy) * 1e-9 if vy != 0 else 1e-9
            
        return np.array([[DDx],[DDz]]) @ Phi[[SynodicState.Y],:][:, (SynodicState.X,SynodicState.VY)] / vy



class _HaloOrbitContinuationService(_OrbitContinuationService):

    """Drive continuation for halo orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.HaloOrbit`
        The orbit.
    """

    def __init__(self, orbit: "HaloOrbit") -> None:
        super().__init__(orbit)

    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Create the default continuation configuration for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`
            The default continuation configuration for halo orbits.
        """
        from hiten.algorithms.continuation.config import \
            OrbitContinuationConfig
        return OrbitContinuationConfig(
            state=(SynodicState.Z,),
            stepper="secant",
        )

    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create the default continuation options for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            The default continuation options for halo orbits.
        """
        from hiten.algorithms.continuation.options import \
            OrbitContinuationOptions
        return OrbitContinuationOptions(
            target=([self.initial_state[SynodicState.Z]], [self.initial_state[SynodicState.Z] + 1.0]),
            step=((1 - self.initial_state[SynodicState.Z]) / (100 - 1),),
            max_members=100,
            max_retries_per_step=50,
            step_min=1e-10,
            step_max=1.0,
            shrink_policy=None,
        )


class _HaloOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for halo orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.HaloOrbit`
        The orbit.
    """

    def __init__(self, orbit: "HaloOrbit") -> None:
        self._amplitude = orbit._amplitude_z
        self._zenith = orbit._zenith
        
        if orbit._initial_state is not None and (self._amplitude is not None or self._zenith is not None):
            self._amplitude = None
            self._zenith = None
        
        super().__init__(orbit)

        from hiten.system.libration.collinear import (CollinearPoint, L1Point,
                                                      L2Point)
        if not isinstance(self._libration_point, CollinearPoint):
            raise TypeError(f"Halo orbits are only defined for CollinearPoint, but got {type(self._libration_point)}.")
        if self._initial_state is None:
            if self._amplitude is None or self._zenith is None:
                raise ValueError("Halo orbits require an 'amplitude_z' (z-amplitude) and 'zenith' ('northern'/'southern') parameter when an initial_state is not provided.")
            if not isinstance(self._libration_point, (L1Point, L2Point)):
                raise ValueError("The analytical guess for L3 Halo orbits is experimental.\n Convergence is not guaranteed and may require more iterations.")

            self._initial_state = self.initial_guess()

        if self._initial_state is not None:
            if self._zenith is None:
                self._zenith = "northern" if self._initial_state[SynodicState.Z] > 0 else "southern"
            # Infer missing amplitude
            if self._amplitude is None:
                self._amplitude = self._initial_state[SynodicState.Z]

    @property
    def zenith(self) -> Literal["northern", "southern"]:
        """(Read-only) Current zenith of the orbit.
        
        Returns
        -------
        Literal["northern", "southern"]
            The orbit zenith.
        """
        return self._zenith

    @property
    def n(self) -> int:
        """(Read-only) Current n value of the orbit.
        
        Returns
        -------
        int
            The orbit n value.
        """
        return 1 if self.zenith == "northern" else -1

    def initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the orbit. using Richardson's third-order analytical approximation.
        
        Returns
        -------
        np.ndarray
            The initial guess for the orbit.

        References
        ----------
        .. [Richardson1980] Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
        collinear libration points".
        """
        amplitude_z = self.amplitude
        gamma = self.libration_point.dynamics.gamma
        won, primary = self.libration_point.dynamics.won
        
        c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
        for N in [2, 3, 4]:
            c[N] = self.libration_point.dynamics.cn(N)

        _, lambda2, _ = self.libration_point.dynamics.linear_modes
        lam = lambda2

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

        deltan = - self.n

        amplitude_x = np.sqrt((-delta - l2 * amplitude_z**2) / l1)
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

        rx = primary + gamma * (-won + x)
        ry = -gamma * y
        rz = gamma * z

        vx = gamma * xdot
        vy = gamma * ydot
        vz = gamma * zdot
        return np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)


class _LyapunovOrbitCorrectionService(_OrbitCorrectionService):
    """Dynamics service for Lyapunov orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.LyapunovOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LyapunovOrbit") -> None:
        super().__init__(orbit)

    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Create the default correction configuration for Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The default correction configuration for Lyapunov orbits.
        """
        from hiten.algorithms.corrector.config import OrbitCorrectionConfig
        from hiten.algorithms.types.configs import (IntegrationConfig,
                                                    NumericalConfig)
        return OrbitCorrectionConfig(
            residual_indices=(SynodicState.VX, SynodicState.Z),
            control_indices=(SynodicState.VY, SynodicState.VZ),
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_y_plane_crossing,
            integration=IntegrationConfig(
                method="adaptive", 
                forward=1, 
                flip_indices=None
            ),
            numerical=NumericalConfig(
                finite_difference=False,
                line_search_enabled=True,
            ),
        )

    def _default_correction_options(self) -> "OrbitCorrectionOptions":
        """Create the default correction options for Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            The default correction options for Lyapunov orbits.
        """
        from hiten.algorithms.corrector.options import OrbitCorrectionOptions
        from hiten.algorithms.types.options import (ConvergenceOptions,
                                                    CorrectionOptions,
                                                    IntegrationOptions,
                                                    NumericalOptions)
        
        return OrbitCorrectionOptions(
            base=CorrectionOptions(
                convergence=ConvergenceOptions(
                    max_attempts=50,
                    tol=1e-12,
                    max_delta=1e-2,
                ),
                integration=IntegrationOptions(
                    dt=1e-2,
                    order=8,
                    max_steps=2000,
                    c_omega_heuristic=20.0,
                    steps=500,
                ),
                numerical=NumericalOptions(
                    fd_step=1e-8,
                    line_search_alpha_reduction=0.5,
                    line_search_min_alpha=1e-4,
                    line_search_armijo_c=0.1,
                ),
            ),
            forward=1,
        )


class _LyapunovOrbitContinuationService(_OrbitContinuationService):
    """Dynamics service for Lyapunov orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.LyapunovOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LyapunovOrbit") -> None:
        super().__init__(orbit)

    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Provides the continuation configuration for Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`
            The continuation configuration.
        """
        from hiten.algorithms.continuation.config import \
            OrbitContinuationConfig
        return OrbitContinuationConfig(
            state=(SynodicState.X, SynodicState.Y),
            stepper="secant",
        )

    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create default continuation options for Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            Default continuation options.
        """
        from hiten.algorithms.continuation.options import \
            OrbitContinuationOptions
        return OrbitContinuationOptions(
            target=(
                [self.initial_state[SynodicState.X], self.initial_state[SynodicState.Y]],
                [self.initial_state[SynodicState.X] + 1.0, self.initial_state[SynodicState.Y] + 1.0]),
            step=(
                (1 - self.initial_state[SynodicState.X]) / (100 - 1),
                (1 - self.initial_state[SynodicState.Y]) / (100 - 1),
            ),
            max_members=100,
            max_retries_per_step=50,
            step_min=1e-10,
            step_max=1.0,
            shrink_policy=None,
        )

class _LyapunovOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for Lyapunov orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.LyapunovOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LyapunovOrbit") -> None:
        self._amplitude_x = orbit._amplitude_x
        
        if orbit._initial_state is not None and self._amplitude_x is not None:
            self._amplitude_x = None
        
        super().__init__(orbit)

        from hiten.system.libration.collinear import (CollinearPoint, L1Point,
                                                      L2Point)
        if not isinstance(self._libration_point, CollinearPoint):
            raise TypeError(f"Lyapunov orbits are only defined for CollinearPoint, but got {type(self.libration_point)}.")
        if self._initial_state is None:
            if self._amplitude_x is None:
                raise ValueError("Lyapunov orbits require an 'amplitude_x' (x-amplitude) parameter when an initial_state is not provided.")
            if not isinstance(self._libration_point, (L1Point, L2Point)):
                raise ValueError(f"Analytical guess is only available for L1/L2 points. An initial_state must be provided for {self._libration_point.name}.")
            self._initial_state = self.initial_guess()

        if self._initial_state is not None and self._amplitude_x is None:
            self._amplitude_x = self._initial_state[SynodicState.X] - self._libration_point.position[0]

        self._amplitude = self._amplitude_x

    def initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the orbit using the analytical approximation.
        
        Returns
        -------
        np.ndarray
            The initial guess for the orbit.

        References
        ----------
        .. [Richardson1980] Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
        collinear libration points".
        """
        L_i = self.libration_point.position
        x_L_i: float = L_i[0]
        c2 = self.libration_point.dynamics.cn(2)
        nu_1 = self.libration_point.dynamics.linear_modes[1]
        a = 2 * c2 + 1
        tau = - (nu_1 **2 + a) / (2*nu_1)
        u = np.array([1, 0, 0, nu_1 * tau]) 

        displacement = self._amplitude_x * u
        state_4d = np.array([x_L_i, 0, 0, 0], dtype=np.float64) + displacement
        state_6d = np.array([state_4d[0], state_4d[1], 0, state_4d[2], state_4d[3], 0], dtype=np.float64)
        return state_6d


class _VerticalOrbitCorrectionService(_OrbitCorrectionService):
    """Dynamics service for Vertical orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.VerticalOrbit`
        The orbit.
    """

    def __init__(self, orbit: "VerticalOrbit") -> None:
        super().__init__(orbit)

    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Create the default correction configuration for Vertical orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The default correction configuration for Vertical orbits.
        """
        from hiten.algorithms.corrector.config import OrbitCorrectionConfig
        from hiten.algorithms.types.configs import (IntegrationConfig,
                                                    NumericalConfig)
        return OrbitCorrectionConfig(
            residual_indices=(SynodicState.VX, SynodicState.Y),     # Want VX=0 and Y=0
            control_indices=(SynodicState.VZ, SynodicState.VY),     # Adjust initial VZ and VY
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_z_plane_crossing,
            integration=IntegrationConfig(
                method="adaptive", 
                forward=1, 
                flip_indices=None
            ),
            numerical=NumericalConfig(
                finite_difference=True,
                line_search_enabled=True,
            ),
        )

    def _default_correction_options(self) -> "OrbitCorrectionOptions":
        """Create the default correction options for Vertical orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            The default correction options for Vertical orbits.
        """
        from hiten.algorithms.corrector.options import OrbitCorrectionOptions
        from hiten.algorithms.types.options import (ConvergenceOptions,
                                                    CorrectionOptions,
                                                    IntegrationOptions,
                                                    NumericalOptions)
        
        return OrbitCorrectionOptions(
            base=CorrectionOptions(
                convergence=ConvergenceOptions(
                    max_attempts=50,
                    tol=1e-12,
                    max_delta=1e-2,
                ),
                integration=IntegrationOptions(
                    dt=1e-2,
                    order=8,
                    max_steps=2000,
                    c_omega_heuristic=20.0,
                    steps=500,
                ),
                numerical=NumericalOptions(
                    fd_step=1e-8,
                    line_search_alpha_reduction=0.5,
                    line_search_min_alpha=1e-4,
                    line_search_armijo_c=0.1,
                ),
            ),
            forward=1,
        )


class _VerticalOrbitContinuationService(_OrbitContinuationService):
    """Dynamics service for Vertical orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.VerticalOrbit`
        The orbit.
    """

    def __init__(self, orbit: "VerticalOrbit") -> None:
        super().__init__(orbit)

    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Provides the continuation configuration for Vertical orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`
            The continuation configuration for Vertical orbits.
        """
        from hiten.algorithms.continuation.config import \
            OrbitContinuationConfig
        return OrbitContinuationConfig(
            state=(SynodicState.X, SynodicState.Y, SynodicState.Z),
            stepper="secant",
        )

    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create default continuation options for Vertical orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            Default continuation options.
        """
        from hiten.algorithms.continuation.options import \
            OrbitContinuationOptions
        return OrbitContinuationOptions(
            target=(
                [self.initial_state[SynodicState.X], self.initial_state[SynodicState.Y], self.initial_state[SynodicState.Z]],
                [self.initial_state[SynodicState.X] + 1.0, self.initial_state[SynodicState.Y] + 1.0, self.initial_state[SynodicState.Z] + 1.0]),
            step=(
                (1 - self.initial_state[SynodicState.X]) / (100 - 1),
                (1 - self.initial_state[SynodicState.Y]) / (100 - 1),
                (1 - self.initial_state[SynodicState.Z]) / (100 - 1),
            ),
            max_members=100,
            max_retries_per_step=50,
            step_min=1e-10,
            step_max=1.0,
            shrink_policy=None,
        )


class _VerticalOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for Vertical orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.VerticalOrbit`
        The orbit.
    """

    def __init__(self, orbit: "VerticalOrbit") -> None:
        self._amplitude_z = orbit._amplitude_z
        
        if orbit._initial_state is not None and self._amplitude_z is not None:
            self._amplitude_z = None
        
        super().__init__(orbit)

        from hiten.system.libration.collinear import (CollinearPoint, L1Point,
                                                      L2Point)
        if not isinstance(self._libration_point, CollinearPoint):
            raise TypeError(f"Vertical orbits are only defined for CollinearPoint, but got {type(self._libration_point)}.")
        if self._initial_state is None:
            if self._amplitude_z is None:
                raise ValueError("Vertical orbits require an 'amplitude_z' (z-amplitude) parameter when an initial_state is not provided.")
            if not isinstance(self._libration_point, (L1Point, L2Point)):
                raise ValueError("The analytical guess for L3 Vertical orbits is experimental.\n Convergence is not guaranteed and may require more iterations.")

            self._initial_state = self.initial_guess()

        if self._initial_state is not None and self._amplitude_z is None:
            self._amplitude_z = abs(self._initial_state[SynodicState.Z])
    
        self._amplitude = self._amplitude_z

    def initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the orbit using Richardson's third-order analytical approximation.
        
        For vertical Lyapunov orbits, the x-amplitude (Ax) is set to zero, meaning the orbit
        oscillates primarily in the z-direction with minimal motion in the x-y plane.
        
        Returns
        -------
        np.ndarray
            The initial guess for the orbit.

        References
        ----------
        .. [Richardson1980] Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
        collinear libration points". Celestial Mechanics 22 (3):241-253.
        """
        amplitude_z = self.amplitude
        gamma = self.libration_point.dynamics.gamma
        won, primary = self.libration_point.dynamics.won
        
        c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
        for N in [2, 3, 4]:
            c[N] = self.libration_point.dynamics.cn(N)

        _, lambda2, _ = self.libration_point.dynamics.linear_modes
        lam = lambda2

        k = 2 * lam / (lam**2 + 1 - c[2])

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

        # For vertical Lyapunov orbits: Ax = 0
        amplitude_x = 0.0
        
        # Normalized amplitude_z
        amplitude_z_norm = amplitude_z / gamma
        
        # Phase angle (initial time)
        phi = 0.0
        
        # Frequency correction
        omega1 = 0.0
        omega2 = s1 * amplitude_x**2 + s2 * amplitude_z_norm**2
        omega = 1 + omega1 + omega2
        
        # Initial phase
        tau1 = omega * 0.0 + phi
        sigma = - won
        
        # Third-order approximation for position
        x = (
            a21 * amplitude_x**2 + a22 * amplitude_z_norm**2
            - amplitude_x * np.cos(tau1)
            + (a23 * amplitude_x**2 - a24 * amplitude_z_norm**2) * np.cos(2 * tau1)
            + (a31 * amplitude_x**3 - a32 * amplitude_x * amplitude_z_norm**2) * np.cos(3 * tau1)
        )
        y = (
            k * amplitude_x * np.sin(tau1)
            + (b21 * amplitude_x**2 - b22 * amplitude_z_norm**2) * np.sin(2 * tau1)
            + (b31 * amplitude_x**3 - b32 * amplitude_x * amplitude_z_norm**2) * np.sin(3 * tau1)
        )
        z = (
            sigma * (
                amplitude_z_norm * np.cos(tau1)
                + d21 * amplitude_x * amplitude_z_norm * (np.cos(2 * tau1) - 3)
                + (d32 * amplitude_x**2 * amplitude_z_norm - d31 * amplitude_z_norm**3) * np.cos(3 * tau1)
            )
        )
        
        # Third-order approximation for velocity
        xdot = (
            lam * amplitude_x * np.sin(tau1)
            - 2 * lam * (a23 * amplitude_x**2 - a24 * amplitude_z_norm**2) * np.sin(2 * tau1)
            - 3 * lam * (a31 * amplitude_x**3 - a32 * amplitude_x * amplitude_z_norm**2) * np.sin(3 * tau1)
        ) * omega
        ydot = (
            lam
            * (
                k * amplitude_x * np.cos(tau1)
                + 2 * (b21 * amplitude_x**2 - b22 * amplitude_z_norm**2) * np.cos(2 * tau1)
                + 3 * (b31 * amplitude_x**3 - b32 * amplitude_x * amplitude_z_norm**2) * np.cos(3 * tau1)
            )
        ) * omega
        zdot = (
            sigma * (
                - lam * amplitude_z_norm * np.sin(tau1)
                - 2 * lam * d21 * amplitude_x * amplitude_z_norm * np.sin(2 * tau1)
                - 3 * lam * (d32 * amplitude_x**2 * amplitude_z_norm - d31 * amplitude_z_norm**3) * np.sin(3 * tau1)
            )
        ) * omega

        # Convert from normalized to synodic frame
        rx = primary + gamma * (-won + x)
        ry = -gamma * y
        rz = gamma * z

        vx = gamma * xdot
        vy = gamma * ydot
        vz = gamma * zdot
        
        return np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)


class _LissajousOrbitCorrectionService(_OrbitCorrectionService):
    """Drive Newton-based differential correction for Lissajous orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.lissajous.LissajousOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LissajousOrbit") -> None:
        super().__init__(orbit)

    def _default_correction_config(self) -> "OrbitCorrectionConfig":
        """Create the default correction configuration for Lissajous orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The default correction configuration for Lissajous orbits.
        """
        raise NotImplementedError("Lissajous orbits are not supported for correction.")

    def _default_correction_options(self) -> "OrbitCorrectionOptions":
        """Create the default correction options for Lissajous orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions`
            The default correction options for Lissajous orbits.
        """
        raise NotImplementedError("Lissajous orbits are not supported for correction.")

class _LissajousOrbitContinuationService(_OrbitContinuationService):
    """Drive continuation for Lissajous orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.lissajous.LissajousOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LissajousOrbit") -> None:
        super().__init__(orbit)

    def _default_continuation_config(self) -> "OrbitContinuationConfig":
        """Create the default continuation configuration for Lissajous orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig`
            The default continuation configuration for Lissajous orbits.
        """
        raise NotImplementedError("Lissajous orbits are not supported for continuation.")

    def _default_continuation_options(self) -> "OrbitContinuationOptions":
        """Create the default continuation options for Lissajous orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions`
            The default continuation options for Lissajous orbits.
        """
        raise NotImplementedError("Lissajous orbits are not supported for continuation.")


class _LissajousOrbitDynamicsService(_OrbitDynamicsService):
    """Dynamics service for Lissajous orbits.
    
    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.lissajous.LissajousOrbit`
        The orbit.
    """

    def __init__(self, orbit: "LissajousOrbit") -> None:
        self._amplitude_y = orbit._amplitude_y
        self._amplitude_z = orbit._amplitude_z
        self._phi = orbit._phi
        self._psi = orbit._psi
        
        if orbit._initial_state is not None and (self._amplitude_y is not None or self._amplitude_z is not None):
            self._amplitude_y = None
            self._amplitude_z = None
        
        super().__init__(orbit)

        from hiten.system.libration.collinear import (CollinearPoint, L1Point,
                                                      L2Point)
        if not isinstance(self._libration_point, CollinearPoint):
            raise TypeError(f"Lissajous orbits are only defined for CollinearPoint, but got {type(self._libration_point)}.")
        if self._initial_state is None:
            if self._amplitude_y is None or self._amplitude_z is None:
                raise ValueError("Lissajous orbits require 'amplitude_y' and 'amplitude_z' parameters when an initial_state is not provided.")
            if not isinstance(self._libration_point, (L1Point, L2Point)):
                raise ValueError("The analytical guess for L3 Lissajous orbits is experimental.\n Convergence is not guaranteed and may require more iterations.")

            self._initial_state = self.initial_guess()

        if self._initial_state is not None:
            # Infer missing amplitudes if not provided
            if self._amplitude_y is None:
                self._amplitude_y = self._initial_state[SynodicState.Y]
            if self._amplitude_z is None:
                self._amplitude_z = self._initial_state[SynodicState.Z]

        # Set a default amplitude for the base class
        self._amplitude = self._amplitude_y if self._amplitude_y is not None else self._amplitude_z

    @property
    def phi(self) -> float:
        """(Read-only) Phase angle for in-plane motion.
        
        Returns
        -------
        float
            Phase angle in radians.
        """
        return self._phi

    @property
    def psi(self) -> float:
        """(Read-only) Phase angle for out-of-plane motion.
        
        Returns
        -------
        float
            Phase angle in radians.
        """
        return self._psi

    @property
    def amplitude_y(self) -> float:
        """(Read-only) y-amplitude of the orbit.
        
        Returns
        -------
        float
            The y-amplitude in nondimensional units.
        """
        return self._amplitude_y

    @property
    def amplitude_z(self) -> float:
        """(Read-only) z-amplitude of the orbit.
        
        Returns
        -------
        float
            The z-amplitude in nondimensional units.
        """
        return self._amplitude_z

    def initial_guess(self) -> np.ndarray:
        """Generate an initial guess for the Lissajous orbit using Richardson's third-order analytical approximation.
        
        This method implements the third-order approximation for Lissajous orbits
        based on Richardson (1980). The analytical solution combines in-plane and
        out-of-plane oscillations with independent frequencies.
        
        Returns
        -------
        np.ndarray
            The initial guess for the orbit as a 6-element state vector.

        References
        ----------
        .. [Richardson1980] Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
        collinear libration points". Celestial Mechanics 22 (3): 241-253.
        """
        amplitude_y = self._amplitude_y
        amplitude_z = self._amplitude_z
        phi = self._phi
        psi = self._psi
        
        gamma = self.libration_point.dynamics.gamma
        won, primary = self.libration_point.dynamics.won
        
        # Normalize amplitudes by gamma
        Ay = amplitude_y / gamma
        Az = amplitude_z / gamma
        
        c = [0.0, 0.0, 0.0, 0.0, 0.0]  # just to keep 5 slots: c[2], c[3], c[4]
        for N in [2, 3, 4]:
            c[N] = self.libration_point.dynamics.cn(N)

        # Get linear frequencies
        _, lambda2, _ = self.libration_point.dynamics.linear_modes
        lam = lambda2
        nu = np.sqrt(c[2])  # Out-of-plane frequency

        k = 2 * lam / (lam**2 + 1 - c[2])
        delta = lam**2 - c[2]

        d1 = (3 * lam**2 / k) * (k * (6 * lam**2 - 1) - 2 * lam)
        d2 = (8 * lam**2 / k) * (k * (11 * lam**2 - 1) - 2 * lam)

        # Second-order coefficients
        a21 = (3 * c[3] * (k**2 - 2)) / (4 * (1 + 2 * c[2]))
        a22 = (3 * c[3]) / (4 * (1 + 2 * c[2]))
        a23 = - (3 * c[3] * lam / (4 * k * d1)) * (
            3 * k**3 * lam - 6 * k * (k - lam) + 4
        )
        a24 = - (3 * c[3] * lam / (4 * k * d1)) * (2 + 3 * k * lam)

        b21 = - (3 * c[3] * lam / (2 * d1)) * (3 * k * lam - 4)
        b22 = (3 * c[3] * lam) / d1

        d21 = - c[3] / (2 * lam**2)

        # Third-order coefficients
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

        # Calculate x-amplitude from y-amplitude
        Ax = Ay / k

        # Phase angles (tau1 for in-plane, tau2 for out-of-plane)
        # At t=0, we use the provided phase angles
        tau1 = phi
        tau2 = psi
        
        deltan = won

        # Third-order position approximation
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
            deltan * (
                Az * np.cos(tau2)
                + d21 * Ax * Az * (np.cos(2 * tau2) - 3)
                + (d32 * Ax**2 * Az - d31 * Az**3) * np.cos(3 * tau2)
            )
        )

        # Third-order velocity approximation
        xdot = (
            Ax * np.sin(tau1)
            - 2 * (a23 * Ax**2 - a24 * Az**2) * np.sin(2 * tau1)
            - 3 * (a31 * Ax**3 - a32 * Ax * Az**2) * np.sin(3 * tau1)
        ) * lam
        
        ydot = (
            k * Ax * np.cos(tau1)
            + 2 * (b21 * Ax**2 - b22 * Az**2) * np.cos(2 * tau1)
            + 3 * (b31 * Ax**3 - b32 * Ax * Az**2) * np.cos(3 * tau1)
        ) * lam
        
        zdot = deltan * (
            -Az * np.sin(tau2)
            - 2 * d21 * Ax * Az * np.sin(2 * tau2)
            - 3 * (d32 * Ax**2 * Az - d31 * Az**3) * np.sin(3 * tau2)
        ) * nu

        # Transform back to dimensional coordinates
        rx = primary + gamma * (-won + x)
        ry = -gamma * y
        rz = gamma * z

        vx = gamma * xdot
        vy = gamma * ydot
        vz = gamma * zdot

        return np.array([rx, ry, rz, vx, vy, vz], dtype=np.float64)


class _OrbitServices(_ServiceBundleBase):
    """Bundle all orbit services together.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The domain object.
    
    correction : :class:`~hiten.algorithms.types.services.orbits._OrbitCorrectionService`
        The correction service.
    continuation : :class:`~hiten.algorithms.types.services.orbits._OrbitContinuationService`
        The continuation service.
    dynamics : :class:`~hiten.algorithms.types.services.orbits._OrbitDynamicsService`
        The dynamics service.
    persistence : :class:`~hiten.algorithms.types.services.orbits._OrbitPersistenceService`
        The persistence service.
    """
    
    def __init__(self, domain_obj: "PeriodicOrbit", correction: _OrbitCorrectionService, continuation: _OrbitContinuationService, dynamics: _OrbitDynamicsService, persistence: _OrbitPersistenceService) -> None:
        super().__init__(domain_obj)
        self.correction = correction
        self.continuation = continuation
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, domain_obj: "PeriodicOrbit") -> "_OrbitServices":
        """Create a default service bundle.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The domain object.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.orbits._OrbitServices`
            The service bundle.
        """
        correction, continuation, dynamics = cls._check_orbit_type(domain_obj)
        
        return cls(
            domain_obj=domain_obj,
            correction=correction(domain_obj),
            continuation=continuation(domain_obj),
            dynamics=dynamics(domain_obj),
            persistence=_OrbitPersistenceService()
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _OrbitDynamicsService) -> "_OrbitServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.orbits._OrbitDynamicsService`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.orbits._OrbitServices`
            The service bundle.
        """
        return cls(
            domain_obj=dynamics.domain_obj,
            correction=_OrbitCorrectionService(dynamics.domain_obj),
            continuation=_OrbitContinuationService(dynamics.domain_obj),
            dynamics=dynamics,
            persistence=_OrbitPersistenceService()
        )

    @staticmethod
    def _check_orbit_type(orbit: "PeriodicOrbit") -> tuple:
        """Check the type of the orbit and return the corresponding correction, continuation, and dynamics services.
        
        Parameters
        ----------
        orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The orbit.

        Returns
        -------
        tuple
            The correction, continuation, and dynamics services.
        """
        from hiten.system.orbits.base import GenericOrbit
        from hiten.system.orbits.halo import HaloOrbit
        from hiten.system.orbits.lissajous import LissajousOrbit
        from hiten.system.orbits.lyapunov import LyapunovOrbit
        from hiten.system.orbits.vertical import VerticalOrbit

        mapping = {
            GenericOrbit: (_GenericOrbitCorrectionService, _GenericOrbitContinuationService, _GenericOrbitDynamicsService),
            HaloOrbit: (_HaloOrbitCorrectionService, _HaloOrbitContinuationService, _HaloOrbitDynamicsService),
            LissajousOrbit: (_LissajousOrbitCorrectionService, _LissajousOrbitContinuationService, _LissajousOrbitDynamicsService),
            LyapunovOrbit: (_LyapunovOrbitCorrectionService, _LyapunovOrbitContinuationService, _LyapunovOrbitDynamicsService),
            VerticalOrbit: (_VerticalOrbitCorrectionService, _VerticalOrbitContinuationService, _VerticalOrbitDynamicsService),
        }

        return mapping[type(orbit)]