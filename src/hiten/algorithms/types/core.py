"""Abstract base class for Hiten classes.

This module provides the abstract base class for all Hiten classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, Union

import pandas as pd

from hiten.algorithms.types.exceptions import EngineError
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)

DomainT = TypeVar("DomainT")

ConfigT = TypeVar("ConfigT", bound=Union["_HitenBaseConfig", None])

ProblemT = TypeVar("ProblemT", bound="_HitenBaseProblem")

ResultT = TypeVar("ResultT", bound="_HitenBaseResults")

BackendT = TypeVar("BackendT", bound="_HitenBaseBackend")

OutputsT = TypeVar("OutputsT")

InterfaceT = TypeVar("InterfaceT", bound="_HitenBaseInterface[ConfigT, ProblemT, ResultT, OutputsT]")

EngineT = TypeVar("EngineT", bound="_HitenBaseEngine[ProblemT, ResultT, OutputsT]")

FacadeT = TypeVar("FacadeT", bound="_HitenBaseFacade")


@dataclass(frozen=True)
class _BackendCall:
    """Describe a backend call with positional and keyword arguments."""

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


class _HitenBaseProblem(ABC):
    """Marker base class for problem payloads produced by interfaces."""

    __slots__ = ()


class _HitenBaseResults(ABC):
    """Marker base class for user-facing results returned by engines."""

    __slots__ = ()


class _HitenBaseConfig(ABC):
    """Marker base class for configuration payloads produced by interfaces."""

    __slots__ = ()


class _HitenBaseBackend(ABC):
    ...


class _HitenBaseInterface(Generic[ConfigT, ProblemT, ResultT, OutputsT], ABC):
    """Shared contract for translating between domain objects and backends."""

    def __init__(self) -> None:
        self._config: ConfigT | None = None

    @property
    def current_config(self) -> ConfigT | None:
        return self._config

    @abstractmethod
    def create_problem(self, config: ConfigT | None = None, *args) -> ProblemT:
        """Compose an immutable problem payload for the backend."""

    @abstractmethod
    def to_backend_inputs(self, problem: ProblemT) -> _BackendCall:
        """Translate a problem into backend invocation arguments."""

    def from_domain(self, *, config: ConfigT | None = None, **kwargs) -> _BackendCall:
        """Convenience helper to build backend inputs directly from config."""
        problem = self.create_problem(config=config, **kwargs)
        return self.to_backend_inputs(problem)

    def to_domain(self, outputs: OutputsT, *, problem: ProblemT) -> Any:
        """Optional hook to mutate or derive domain artefacts from outputs."""
        return None

    @abstractmethod
    def to_results(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any = None) -> ResultT:
        """Package backend outputs into user-facing result objects."""

    def bind_backend(self, backend: _HitenBaseBackend) -> None:
        self._backend = backend

    def on_start(self, problem: ProblemT) -> None:
        return None

    def on_success(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any = None) -> None:
        return None

    def on_failure(self, exc: Exception, *, problem: ProblemT) -> None:
        return None


class _HitenBaseEngine(Generic[ProblemT, ResultT, OutputsT], ABC):
    """Template providing the canonical engine flow."""

    def __init__(
        self,
        *,
        backend: _HitenBaseBackend[ProblemT, ResultT, OutputsT],
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT] | None = None,
    ) -> None:
        self._backend = backend
        self._interface = interface

    @property
    def backend(self) -> _HitenBaseBackend[ProblemT, ResultT, OutputsT]:
        return self._backend

    def with_interface(
        self,
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT],
    ) -> "_HitenBaseEngine[ProblemT, ResultT, OutputsT]":
        self._interface = interface
        return self

    def solve(self, problem: ProblemT) -> ResultT:
        """Execute the standard engine orchestration for ``problem``."""

        interface = self._get_interface(problem)
        interface.bind_backend(self._backend)
        call = interface.to_backend_inputs(problem)
        interface.on_start(problem)
        self._before_backend(problem, call, interface)

        try:
            outputs = self._invoke_backend(call)

        except Exception as exc:
            interface.on_failure(exc, problem=problem)
            self._handle_backend_failure(exc, problem=problem, call=call, interface=interface)

        domain_payload = interface.to_domain(outputs, problem=problem)
        interface.on_success(outputs, problem=problem, domain_payload=domain_payload)
        self._after_backend_success(outputs, problem=problem, domain_payload=domain_payload, interface=interface)
        return interface.to_results(outputs, problem=problem, domain_payload=domain_payload)

    def _get_interface(
        self,
        problem: ProblemT,
    ) -> _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]:
        if self._interface is None:
            raise EngineError(
                f"{self.__class__.__name__} must be configured with an interface before solving."
            )
        return self._interface

    def set_interface(
        self,
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT],
    ) -> None:
        self._interface = interface

    def with_interface(
        self,
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT],
    ) -> "_HitenBaseEngine[ProblemT, ResultT, OutputsT]":
        self.set_interface(interface)
        return self

    def _before_backend(self, problem: ProblemT, call: _BackendCall, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        return None

    def _after_backend_success(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        return None

    def _handle_backend_failure(self, exc: Exception, *, problem: ProblemT, call: _BackendCall, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        raise EngineError(exc) from exc

    def _invoke_backend(self, call: _BackendCall) -> OutputsT:
        backend_callable = getattr(self._backend, "run")
        return backend_callable(*call.args, **call.kwargs)


class _HitenBaseBackend(Generic[ProblemT, ResultT, OutputsT]):
    """Abstract base class for all backend implementations in the Hiten framework.
    
    This class defines the common interface and lifecycle hooks that all backend
    implementations should follow. Backends are responsible for the core numerical
    computations and algorithms, while engines handle orchestration and interfaces
    manage data translation.
    
    Backend implementations should inherit from this class and implement the
    appropriate abstract methods for their specific domain (correction, continuation,
    integration, etc.).
    
    Notes
    -----
    This base class provides common lifecycle hooks that backends can override:
    - on_iteration: Called after each iteration of the main algorithm
    - on_accept: Called when the backend detects convergence/success
    - on_failure: Called when the backend completes without converging
    - on_success: Called by the engine after final acceptance
    
    Subclasses should document their specific solve/compute methods and any
    additional parameters they accept.
    
    Examples
    --------
    >>> class MyBackend(_HitenBaseBackend):
    ...     def run(self, **kwargs):
    ...         # Implementation here
    ...         pass
    >>> 
    >>> backend = MyBackend()
    >>> result = backend.run(input_data)
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> OutputsT:
        """Run the backend.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the run method.
        """
        ...

    def on_iteration(self, k: int, x: Any, r_norm: float) -> None:
        """Called after each iteration of the main algorithm.
        
        Parameters
        ----------
        k : int
            Current iteration number (0-based).
        x : Any
            Current solution estimate or state.
        r_norm : float
            Current residual norm or convergence metric.
        """
        return

    def on_accept(self, x: Any, *, iterations: int, residual_norm: float) -> None:
        """Called when the backend detects convergence or successful completion.
        
        Parameters
        ----------
        x : Any
            Final solution or result.
        iterations : int
            Total number of iterations performed.
        residual_norm : float
            Final residual norm or convergence metric.
        """
        return

    def on_failure(self, x: Any, *, iterations: int, residual_norm: float) -> None:
        """Called when the backend completes without converging.
        
        Parameters
        ----------
        x : Any
            Final solution estimate (may not be converged).
        iterations : int
            Total number of iterations performed.
        residual_norm : float
            Final residual norm or convergence metric.
        """
        return

    def on_success(self, x: Any, *, iterations: int, residual_norm: float) -> None:
        """Called by the engine after final acceptance.
        
        This hook is called after the engine has accepted the backend's result
        and is typically used for final cleanup or logging.
        
        Parameters
        ----------
        x : Any
            Final accepted solution.
        iterations : int
            Total number of iterations performed.
        residual_norm : float
            Final residual norm or convergence metric.
        """
        return


class _HitenBaseFacade(Generic[ConfigT, ProblemT, ResultT]):
    """Abstract base class for user-facing facades in the Hiten framework.
    
    This class provides a common pattern for building facades that orchestrate
    the entire pipeline: facade → engine → interface → backend. Facades serve
    as the main entry point for users and handle dependency injection, configuration
    management, and result processing.
    
    The facade pattern provides:
    - Clean user-facing APIs that hide implementation complexity
    - Consistent dependency injection patterns
    - Factory methods for easy construction with default components
    - Configuration management and validation
    - Result processing and caching
    
    Notes
    -----
    Facades should follow these patterns:
    1. Accept engines via constructor (dependency injection)
    2. Provide `with_default_engine()` class methods for easy construction
    3. Delegate computation to engines while handling configuration
    4. Provide domain-specific methods like plotting, caching, etc.
    5. Handle result processing and user-friendly error messages
    
    Examples
    --------
    >>> class MyFacade(_HitenBaseFacade):
    ...     def __init__(self, config, engine=None):
    ...         super().__init__()
    ...         self.config = config
    ...         self._engine = engine
    ...     
    ...     @classmethod
    ...     def with_default_engine(cls, config):
    ...         backend = MyBackend()
    ...         interface = MyInterface()
    ...         engine = MyEngine(backend=backend, interface=interface)
    ...         return cls(config, engine=engine)
    ...     
    ...     def solve(self, **kwargs):
    ...         problem = self._create_problem(**kwargs)
    ...         return self._engine.solve(problem)
    """

    def __init__(self, config, interface, engine) -> None:
        """Initialize the facade."""
        self._make_pipeline(config, interface, engine)  

    @classmethod
    @abstractmethod
    def with_default_engine(cls, config, interface) -> "_HitenBaseFacade[ConfigT, ProblemT, ResultT]":
        pass
    
    @abstractmethod
    def solve(self, **kwargs) -> ResultT:
        """Solve the problem using the configured engine.
        
        This method should be implemented by concrete facades to define
        the main entry point for the algorithm. It typically:
        1. Creates a problem from the input parameters
        2. Delegates to the engine for computation
        3. Processes and returns the results
        
        Parameters
        ----------
        **kwargs
            Problem-specific parameters that vary by facade implementation.
            Common parameters may include:
            - Input data (orbits, manifolds, etc.)
            - Configuration overrides
            - Algorithm-specific options
            
        Returns
        -------
        Any
            Domain-specific results, typically containing:
            - Computed data structures
            - Convergence information
            - Metadata and diagnostics
        """
        ...

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.
        
        Parameters
        ----------
        **kwargs
            Configuration parameters to update.
        
        Raises
        ------
        ValueError
            If the configuration parameter is not valid.
        """
        # Filter out None values
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        if not filtered_kwargs:
            return
            
        # Get all fields from the current config
        config_dict = {}
        for field in self._config.__dataclass_fields__:
            config_dict[field] = getattr(self._config, field)
        
        # Apply overrides
        for key, value in filtered_kwargs.items():
            if hasattr(self._config, key):
                config_dict[key] = value
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
        
        # Create new instance and update internal state
        self._config = type(self._config)(**config_dict)

    def _set_engine(self, engine: _HitenBaseEngine[ProblemT, ResultT, OutputsT]) -> None:
        self._engine = engine

    def _set_interface(self, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        self._interface = interface

    def _set_backend(self, backend: _HitenBaseBackend[ProblemT, ResultT, OutputsT]) -> None:
        self._backend = backend

    def _set_config(self, config: ConfigT) -> None:
        self._config = config

    def _get_engine(self) -> _HitenBaseEngine[ProblemT, ResultT, OutputsT]:
        return self._engine

    def _get_interface(self) -> _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]:
        return self._interface

    def _get_backend(self) -> _HitenBaseBackend[ProblemT, ResultT, OutputsT]:
        return self._backend

    def _get_config(self) -> ConfigT:
        return self._config

    def _create_problem(self, domain_obj: DomainT, override: bool = False, *args, **kwargs) -> ProblemT:
        """Create a problem object from input parameters.
        
        This method can be overridden by concrete facades to handle
        problem creation logic specific to their domain.
        
        Parameters
        ----------
        domain_obj : DomainT
            The domain object to create a problem for.
        override : bool, default=False
            Whether to override configuration with provided kwargs.
        *args
            Additional positional arguments to pass to interface.create_problem.
        **kwargs
            Configuration parameters to update if override=True
            
        Returns
        -------
        Any
            Problem object suitable for the engine.
        """
        if override and kwargs:
            self.update_config(**kwargs)
        
        interface = self._get_interface()
        config = self._get_config()
        return interface.create_problem(config=config, domain_obj=domain_obj, *args)


    def _make_pipeline(self, config, interface, engine):
        self._config: ConfigT = config
        self._interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT] = interface
        self._engine: _HitenBaseEngine[ProblemT, ResultT, OutputsT] = engine
        self._engine.set_interface(interface)
        self._backend: _HitenBaseBackend[ProblemT, ResultT, OutputsT] = engine.backend
        interface.bind_backend(self._backend)

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.
        """
        pass

    @property
    def results(self) -> ResultT:
        return self._results


class _HitenBase(ABC):
    """Abstract base class for public Hiten classes.
    """

    def __init__(self, services: _ServiceBundleBase):
        self._services = services
        self._unpack_services()

    @property
    def services(self) -> _ServiceBundleBase:
        return self._services

    @property
    def persistence(self) -> _PersistenceServiceBase:
        """Get the persistence service if available."""
        if hasattr(self, '_persistence'):
            return self._persistence
        raise AttributeError("No persistence service available in this service bundle")
    
    @property
    def dynamics(self) -> _DynamicsServiceBase:
        """Get the dynamics service if available."""
        if hasattr(self, '_dynamics'):
            return self._dynamics
        raise AttributeError("No dynamics service available in this service bundle")
    

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        """Get state for pickling.
        
        Excludes services from pickling as they often contain
        non-serializable objects like numba-compiled functions.
        """
        state = self.__dict__.copy()
        state.pop("_services", None)
        # Only remove service-related attributes, not all private attributes
        service_attrs = [name for name in dir(self) if name.startswith('_') and not name.startswith('__')]
        for attr in service_attrs:
            if hasattr(self, attr) and not callable(getattr(self, attr)):
                # Only remove attributes that are clearly service-related
                if attr in ['_cache', '_persistence', '_dynamics', '_correction', '_continuation', 'pipeline', '_conversion'] or attr.endswith('_service'):
                    state.pop(attr, None)
        return state
    
    def __setstate__(self, state):
        """Set state after unpickling.
        """
        self.__dict__.update(state)
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = {}

    def _unpack_services(self) -> None:
        """Unpack services from the service bundle into individual attributes.
        
        This method extracts services from the service bundle and creates
        individual attributes on this instance for easy access.
        """
        if not hasattr(self, "_services") or self._services is None:
            return
            
        # Get all attributes from the service bundle that are not private
        service_attrs = {
            name: getattr(self._services, name) 
            for name in dir(self._services) 
            if not name.startswith('_') and not callable(getattr(self._services, name))
        }
        
        # Create individual service attributes on this instance
        for service_name, service_instance in service_attrs.items():
            setattr(self, f"_{service_name}", service_instance)

    def _bind_services(self) -> None:
        """Bind individual service properties from the service bundle.
        
        This method should be called by child classes after setting up _services
        in their __setstate__ or load methods to ensure the parent class properties
        work correctly.
        """
        if hasattr(self, "_services") and self._services is not None:
            self._unpack_services()

    def _setup_services(self, services: _ServiceBundleBase) -> None:
        """Complete service setup including binding and cache reset.
        
        This method handles the full service setup pattern:
        1. Sets the service bundle
        2. Binds individual service properties
        3. Resets the dynamics cache if available
        
        Parameters
        ----------
        services : _ServiceBundleBase
            The service bundle to set up
        """
        self._services = services
        self._bind_services()
        # Reset dynamics cache if dynamics service is available
        if hasattr(self, '_dynamics') and self._dynamics is not None:
            self._dynamics.reset()

    @classmethod
    def _load_with_services(cls, filepath: str | Path, persistence_service, services_factory, **kwargs) -> "_HitenBase":
        """Generic load method that handles the common pattern.
        
        This method abstracts the common load pattern:
        1. Load object from file using persistence service
        2. Create services using the factory
        3. Initialize the object with services
        4. Return the loaded object
        
        Parameters
        ----------
        filepath : str or Path
            Path to the file to load from
        persistence_service : _PersistenceServiceBase
            The persistence service to use for loading
        services_factory : callable
            Function that takes the loaded object and returns services
        **kwargs
            Additional arguments passed to the load method
            
        Returns
        -------
        _HitenBase
            The loaded object with services properly initialized
        """
        obj = persistence_service.load(filepath, **kwargs)
        services = services_factory(obj)
        super(cls, obj).__init__(services)
        return obj

    def save(self, filepath: str | Path, **kwargs) -> None:
        """Save the object to a file.

        Parameters
        ----------
        filepath : str or Path
            The path to the file to save the object to.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        self.persistence.save(self, filepath, **kwargs)

    def load_inplace(self, filepath: str | Path, **kwargs) -> "_HitenBase":
        """Load data into this object from a file (in place).

        Parameters
        ----------
        filepath : str or Path
            The path to the file to load the object from.
        **kwargs
            Additional keyword arguments passed to the load method.
            
        Returns
        -------
        :class:`~hiten.algorithms.types.core._HitenBase`
            The object with loaded data (self).
        """
        self.persistence.load_inplace(self, filepath, **kwargs)
        return self

    @classmethod
    @abstractmethod
    def load(cls, filepath: str | Path, **kwargs) -> "_HitenBase":
        """Load the object from a file.
        
        Parameters
        ----------
        filepath : str or Path
            The path to the file to load the object from.
        **kwargs
            Additional keyword arguments passed to the load method.
            
        Returns
        -------
        :class:`~hiten.system.base._HitenBase`
            The loaded object.
        """
        ...

    def to_csv(self, filepath: str | Path, **kwargs) -> None:
        """Save the object to a CSV file.

        Parameters
        ----------
        filepath : str or Path
            The path to the file to save the object to.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        ...

    def to_df(cls, **kwargs) -> pd.DataFrame:
        """Convert the object to a pandas DataFrame.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the to_df method.
            
        Returns
        -------
        pandas.DataFrame
            The converted object.
        """
        ...