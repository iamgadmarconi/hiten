"""Abstract base class for Hiten classes.

This module provides the abstract base class for all Hiten classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Union

import pandas as pd

from hiten.algorithms.types.exceptions import EngineError
from hiten.algorithms.types.serialization import _SerializeBase
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)

if TYPE_CHECKING:
    from hiten.algorithms.types.configs import _HitenBaseConfig

# Type variables for the Hiten base classes
DomainT = TypeVar("DomainT")

# Type variable for the configuration type
ConfigT = TypeVar("ConfigT", bound=Union["_HitenBaseConfig", None])

# Type variable for the problem type
ProblemT = TypeVar("ProblemT", bound="_HitenBaseProblem")

# Type variable for the result type
ResultT = TypeVar("ResultT", bound="_HitenBaseResults")

# Type variable for the backend type
BackendT = TypeVar("BackendT", bound="_HitenBaseBackend")

# Type variable for the outputs type
OutputsT = TypeVar("OutputsT")

# Type variable for the interface type
InterfaceT = TypeVar("InterfaceT", bound="_HitenBaseInterface[ConfigT, ProblemT, ResultT, OutputsT]")

# Type variable for the engine type
EngineT = TypeVar("EngineT", bound="_HitenBaseEngine[ProblemT, ResultT, OutputsT]")

# Type variable for the facade type
FacadeT = TypeVar("FacadeT", bound="_HitenBasePipeline")


@dataclass(frozen=True)
class _BackendCall:
    """Describe a backend call to execute against a backend."""

    request: Any | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _DomainPayload(ABC):
    """Immutable base container for domain side-effects produced by interfaces.

    Subclasses should provide a concrete ``_from_mapping`` implementation that
    returns an instance of their own type. The base class offers convenience
    helpers for interacting with the underlying mapping while keeping it
    read-only.
    """

    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    @classmethod
    @abstractmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> "_DomainPayload":
        """Instantiate the payload from a plain mapping."""

    @classmethod
    def empty(cls) -> "_DomainPayload":
        """Return an empty payload instance for this subclass."""

        return cls._from_mapping({})

    def get(self, key: str, default: Any | None = None) -> Any | None:
        """Fetch an optional entry without raising if the key is missing."""

        return self.data.get(key, default)

    def require(self, key: str) -> Any:
        """Fetch a required entry, raising ``KeyError`` if absent."""

        if key not in self.data:
            raise KeyError(f"Domain payload missing required key '{key}'")
        return self.data[key]

    def as_dict(self) -> dict[str, Any]:
        """Return a mutable copy of the payload contents."""

        return dict(self.data)

    def merge(self, **updates: Any) -> "_DomainPayload":
        """Return a new payload with ``updates`` applied."""

        merged = {**self.data, **updates}
        return self.__class__._from_mapping(merged)

    def merge_from_mapping(self, mapping: Mapping[str, Any]) -> "_DomainPayload":
        """Return a new payload including keys from ``mapping``."""

        merged = {**self.data, **dict(mapping)}
        return self.__class__._from_mapping(merged)

    def keys(self):
        """Expose mapping-style key iteration."""

        return self.data.keys()

    def items(self):
        """Expose mapping-style item iteration."""

        return self.data.items()

    def values(self):
        """Expose mapping-style value iteration."""

        return self.data.values()


class _HitenBaseProblem(ABC):
    """Marker base class for problem payloads produced by interfaces."""

    __slots__ = ()


class _HitenBaseResults(ABC):
    """Marker base class for user-facing results returned by engines."""

    __slots__ = ()


class _HitenBaseResults(ABC):
    """Marker base class for user-facing results returned by engines."""

    __slots__ = ()


class _HitenBaseEvent(ABC):
    """Marker base class for event payloads produced by interfaces."""

    __slots__ = ()


class _HitenBaseBackend(ABC):
    """Abstract base class for all backend implementations in the Hiten framework."""

    ...


class _HitenBaseInterface(Generic[ConfigT, ProblemT, ResultT, OutputsT], ABC):
    """Shared contract for translating between domain objects and backends.
    
    Parameters
    ----------
    :class:`~hiten.algorithms.types.core.ConfigT` : TypeVar
        The configuration type.
    :class:`~hiten.algorithms.types.core.ProblemT` : TypeVar
        The problem type.
    :class:`~hiten.algorithms.types.core.ResultT` : TypeVar
        The result type.
    :class:`~hiten.algorithms.types.core.OutputsT` : TypeVar
        The outputs type.
    """

    def __init__(self) -> None:
        self._config: ConfigT | None = None

    @property
    def current_config(self) -> ConfigT | None:
        """Get the current configuration."""
        return self._config

    @abstractmethod
    def create_problem(self, config: ConfigT | None = None, options: Any = None, *args) -> ProblemT:
        """Compose an immutable problem payload for the backend.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT` | None, optional
            The configuration to use for the problem
        options : Any, optional
            The options to use for the problem
        *args : Any
            Additional arguments to pass to the problem.

        Returns
        -------
        :class:`~hiten.algorithms.types.core.ProblemT`
            The problem payload.
        """

    @abstractmethod
    def to_backend_inputs(self, problem: ProblemT) -> _BackendCall:
        """Translate a problem into backend invocation arguments.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to translate into backend invocation arguments.

        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend invocation arguments.
        """

    def from_domain(self, *, config: ConfigT | None = None, **kwargs) -> _BackendCall:
        """Convenience helper to build backend inputs directly from config.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT` | None, optional
            The configuration to use for the problem
        **kwargs : Any
            Additional arguments to pass to the problem.

        Returns
        -------
        :class:`~hiten.algorithms.types.core._BackendCall`
            The backend invocation arguments.
        """
        problem = self.create_problem(config=config, **kwargs)
        return self.to_backend_inputs(problem)

    def to_domain(self, outputs: OutputsT, *, problem: ProblemT) -> Any:
        """Optional hook to mutate or derive domain artefacts from outputs.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.types.core.OutputsT`
            The outputs to mutate or derive domain artefacts from.
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to mutate or derive domain artefacts from.
        
        Returns
        -------
        Any
            The domain artefacts.
        """
        return None

    @abstractmethod
    def to_results(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any = None) -> ResultT:
        """Package backend outputs into user-facing result objects.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.types.core.OutputsT`
            The outputs to package into user-facing result objects.
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to package into user-facing result objects.
        domain_payload : Any, optional
            The domain payload to package into user-facing result objects.

        Returns
        -------
        :class:`~hiten.algorithms.types.core.ResultT`
            The user-facing result objects.
        """

    def bind_backend(self, backend: _HitenBaseBackend) -> None:
        """Bind the backend to the interface.
        
        Parameters
        ----------
        backend : :class:`~hiten.algorithms.types.core._HitenBaseBackend`
            The backend to bind to the interface.
        """
        self._backend = backend


class _HitenBaseEngine(Generic[ProblemT, ResultT, OutputsT], ABC):
    """Template providing the canonical engine flow.
    
    Parameters
    ----------
    :class:`~hiten.algorithms.types.core.ProblemT` : TypeVar
        The problem type.
    :class:`~hiten.algorithms.types.core.ResultT` : TypeVar
        The result type.
    :class:`~hiten.algorithms.types.core.OutputsT` : TypeVar
        The outputs type.
    """

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
        """Get the backend."""
        return self._backend

    def solve(self, problem: ProblemT) -> ResultT:
        """Execute the standard engine orchestration for ``problem``.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to solve.

        Returns
        -------
        :class:`~hiten.algorithms.types.core.ResultT`
            The result of the engine.
        """

        interface = self._get_interface(problem)
        interface.bind_backend(self._backend)
        call = interface.to_backend_inputs(problem)
        self._before_backend(problem, call, interface)

        try:
            outputs = self._invoke_backend(call)

        except Exception as exc:
            self._handle_backend_failure(exc, problem=problem, call=call, interface=interface)

        domain_payload = interface.to_domain(outputs, problem=problem)
        self._after_backend_success(outputs, problem=problem, domain_payload=domain_payload, interface=interface)
        return interface.to_results(outputs, problem=problem, domain_payload=domain_payload)

    def _get_interface(
        self,
        problem: ProblemT,
    ) -> _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]:
        """Get the interface."""
        if self._interface is None:
            raise EngineError(
                f"{self.__class__.__name__} must be configured with an interface before solving."
            )
        return self._interface

    def set_interface(
        self,
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT],
    ) -> None:
        """Set the interface.
        
        Parameters
        ----------
        interface : :class:`~hiten.algorithms.types.core._HitenBaseInterface`
            The interface to set.
        """
        self._interface = interface

    def with_interface(
        self,
        interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT],
    ) -> "_HitenBaseEngine[ProblemT, ResultT, OutputsT]":
        """Set the interface.
        
        Parameters
        ----------
        interface : :class:`~hiten.algorithms.types.core._HitenBaseInterface`
            The interface to set.
        """
        self.set_interface(interface)
        return self

    def _before_backend(self, problem: ProblemT, call: _BackendCall, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        """Called before the backend is invoked.
        
        Parameters
        ----------
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to solve.
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.
        interface : :class:`~hiten.algorithms.types.core._HitenBaseInterface`
            The interface.
        """
        return None

    def _after_backend_success(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        """Called after the backend is invoked.
        
        Parameters
        ----------
        outputs : :class:`~hiten.algorithms.types.core.OutputsT`
            The outputs to succeed.

        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to succeed.
        domain_payload : Any
            The domain payload to succeed.
        interface : :class:`~hiten.algorithms.types.core._HitenBaseInterface`
            The interface.
        """
        return None

    def _handle_backend_failure(self, exc: Exception, *, problem: ProblemT, call: _BackendCall, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        """Handle backend failure.
        
        Parameters
        ----------
        exc : Exception
            The exception to fail.
        problem : :class:`~hiten.algorithms.types.core.ProblemT`
            The problem to fail.
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.
        interface : :class:`~hiten.algorithms.types.core._HitenBaseInterface`
            The interface.
        """
        raise EngineError(exc) from exc

    def _invoke_backend(self, call: _BackendCall) -> OutputsT:
        """Invoke the backend.
        
        Parameters
        ----------
        call : :class:`~hiten.algorithms.types.core._BackendCall`
            The call to the backend.

        Returns
        -------
        :class:`~hiten.algorithms.types.core.OutputsT`
            The outputs of the backend.
        """
        request = call.request
        if request is None:
            return self._backend.run(**call.kwargs)
        return self._backend.run(request=request, **call.kwargs)


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

        Returns
        -------
        :class:`~hiten.algorithms.types.core.OutputsT`
            The outputs of the backend.
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

        Returns
        -------
        None
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

        Returns
        -------
        None
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

        Returns
        -------
        None
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

        Returns
        -------
        None
        """
        return


class _HitenBasePipeline(Generic[ConfigT, ProblemT, ResultT]):
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
    >>> class MyFacade(_HitenBasePipeline):
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

    def __init__(self, config, engine, interface=None, backend=None) -> None:
        """Initialize the facade.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            The configuration to use for the facade.
        engine : :class:`~hiten.algorithms.types.core.EngineT`
            The engine to use for the facade. The engine should already be
            configured with its backend and interface.
        interface : :class:`~hiten.algorithms.types.core.InterfaceT`, optional
            The interface to use. If not provided, uses engine's interface.
            If provided, must match engine's interface.
        backend : :class:`~hiten.algorithms.types.core.BackendT`, optional
            The backend to use. If not provided, uses engine's backend.
            If provided, must match engine's backend.
            
        Raises
        ------
        ValueError
            If provided interface/backend differs from engine's interface/backend.
        
        Notes
        -----
        **Recommended usage**: Only pass `config` and `engine`:
        
        >>> backend = MyBackend()
        >>> interface = MyInterface()
        >>> engine = MyEngine(backend=backend, interface=interface)
        >>> pipeline = MyPipeline(config=config, engine=engine)  # Clean!
        
        The interface and backend parameters are optional and mainly exist
        for backwards compatibility. Always prefer extracting them from the
        engine to avoid duplication and potential mismatches.
        """
        self._make_pipeline(config, interface, engine, backend)  

    @classmethod
    @abstractmethod
    def with_default_engine(cls, config, interface, backend) -> "_HitenBasePipeline[ConfigT, ProblemT, ResultT]":
        """Create a facade instance with a default engine (factory).
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            The configuration to use for the facade.
        interface : :class:`~hiten.algorithms.types.core.InterfaceT`
            The interface to use for the facade.
        backend : :class:`~hiten.algorithms.types.core.BackendT`
            The backend to use for the facade.
        """
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
        """Set the engine."""
        self._engine = engine

    def _set_interface(self, interface: _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]) -> None:
        """Set the interface."""
        self._interface = interface

    def _set_backend(self, backend: _HitenBaseBackend[ProblemT, ResultT, OutputsT]) -> None:
        """Set the backend."""
        self._backend = backend

    def _set_config(self, config: ConfigT) -> None:
        """Set the config."""
        self._config = config

    def _get_engine(self) -> _HitenBaseEngine[ProblemT, ResultT, OutputsT]:
        """Get the engine."""
        return self._engine

    def _get_interface(self) -> _HitenBaseInterface[Any, ProblemT, ResultT, OutputsT]:
        """Get the interface."""
        return self._interface

    def _get_backend(self) -> _HitenBaseBackend[ProblemT, ResultT, OutputsT]:
        """Get the backend."""
        return self._backend

    def _get_config(self) -> ConfigT:
        """Get the config."""
        return self._config

    def _create_problem(
        self,
        domain_obj: DomainT,
        *args,
        options: Any = None,
        **kwargs,
    ) -> ProblemT:
        """Create a problem object from input parameters.

        This base implementation delegates directly to the configured
        interface. Facades are responsible for updating configuration and
        constructing options objects before calling this helper.

        Parameters
        ----------
        domain_obj : DomainT
            The domain object to create a problem for.
        options : Any, optional
            Options object to forward to ``interface.create_problem``.
        *args, **kwargs
            Additional parameters forwarded unchanged to the interface.

        Returns
        -------
        ProblemT
            Problem payload suitable for the engine.
        """
        interface = self._get_interface()
        config = self._get_config()
        return interface.create_problem(
            domain_obj=domain_obj,
            config=config,
            options=options,
            *args,
            **kwargs,
        )


    def _make_pipeline(self, config, interface, engine, backend):
        """Make the pipeline.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            The configuration to use for the pipeline.
        interface : :class:`~hiten.algorithms.types.core.InterfaceT` or None
            The interface to use for the pipeline. If None, extracts from engine.
        engine : :class:`~hiten.algorithms.types.core.EngineT`
            The engine to use for the pipeline.
        backend : :class:`~hiten.algorithms.types.core.BackendT` or None
            The backend to use for the pipeline. If None, extracts from engine.
            
        Raises
        ------
        ValueError
            If provided interface/backend differs from engine's interface/backend.
        """
        self._config: ConfigT = config
        self._engine: _HitenBaseEngine[ProblemT, ResultT, OutputsT] = engine
        
        # Extract interface from engine if not provided, otherwise validate consistency
        if interface is None:
            self._interface = self._engine._interface
            if self._interface is None:
                raise ValueError(
                    "Engine has no interface configured. Either provide an interface "
                    "parameter or configure the engine with an interface first."
                )
        else:
            # Validate that provided interface matches engine's interface
            if self._engine._interface is not None and interface is not self._engine._interface:
                raise ValueError(
                    f"Interface mismatch: the provided interface must be the same instance as "
                    f"engine._interface. Got interface={interface} but engine._interface={self._engine._interface}. "
                    f"To avoid this error, either omit the interface parameter (recommended) or ensure "
                    f"interface is engine._interface."
                )
            self._interface = interface
            self._engine.set_interface(interface)
        
        # Extract backend from engine if not provided, otherwise validate consistency
        if backend is None:
            self._backend = self._engine.backend
        else:
            # Validate that provided backend matches engine's backend
            if backend is not self._engine.backend:
                raise ValueError(
                    f"Backend mismatch: the provided backend must be the same instance as "
                    f"engine.backend. Got backend={backend} but engine.backend={self._engine.backend}. "
                    f"To avoid this error, either omit the backend parameter (recommended) or ensure "
                    f"backend is engine.backend."
                )
            self._backend = backend
        
        # Ensure interface is bound to the correct backend
        self._interface.bind_backend(self._backend)

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.

        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            The configuration to validate.
        """
        pass

    @property
    def results(self) -> ResultT:
        """Get the results."""
        return self._results


class _HitenBase(_SerializeBase, ABC):
    """Abstract base class for public Hiten classes.

    Parameters
    ----------
    services : :class:`~hiten.algorithms.types.core._ServiceBundleBase`
        The services to use for the base class.
    """

    def __init__(self, services: _ServiceBundleBase):
        """Initialize the service bundle."""
        self._services = services
        self._unpack_services()

    @property
    def services(self) -> _ServiceBundleBase:
        """Get the services."""
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

    def _get_computed_properties_source(self):
        """Get the source object for computed properties.
        
        By default, uses the dynamics service if available. Subclasses can override to use
        a different source object or multiple sources.
        
        Returns
        -------
        object or None
            The object containing the computed properties, or None if no source available.
        """
        return getattr(self, 'dynamics', None) if hasattr(self, 'dynamics') else None
    
    def _set_computed_properties_target(self):
        """Get the target object for restoring computed properties.
        
        By default, uses the dynamics service if available. Subclasses can override to use
        a different target object.
        
        Returns
        -------
        object or None
            The object where computed properties should be restored, or None if no target available.
        """
        return getattr(self, 'dynamics', None) if hasattr(self, 'dynamics') else None
    
    def _is_computed_property(self, attr_name, value):
        """Determine if an attribute represents a computed property that should be preserved.
        
        This method can be overridden by subclasses to define custom logic for
        identifying computed properties. By default, preserves attributes that:
        1. Start with underscore (private attributes)
        2. Are not None
        3. Are not callable (methods)
        4. Are not basic types like strings, ints, bools (computed data)
        5. Are not abstract base class internals
        6. Are not properties with no setters
        
        Parameters
        ----------
        attr_name : str
            The name of the attribute
        value : any
            The value of the attribute
            
        Returns
        -------
        bool
            True if this attribute should be preserved during serialization
        """
        # Skip if None or callable
        if value is None or callable(value):
            return False
            
        # Skip service-related attributes
        if attr_name in ['_services', '_cache', '_domain_obj']:
            return False
            
        # Skip abstract base class internals
        if attr_name.startswith('_abc_'):
            return False
            
        # Skip module references
        if hasattr(value, '__module__') and value.__module__ == 'abc':
            return False
            
        # Skip known problematic types
        if isinstance(value, (type, type(None))):
            return False
            
        # Preserve private attributes that contain computed data
        if attr_name.startswith('_'):
            return True
            
        return False

    def _get_service_related_attrs(self):
        """Get list of service-related attribute names that should be excluded from serialization."""
        return ['_services', '_cache', '_persistence', '_dynamics', '_correction', 
                '_continuation', '_conversion', 'pipeline', '_ham_conversion', 
                '_ham_dynamics', '_ham_pipeline', '_lgf_persistence']

    def _is_service_related_attr(self, attr_name):
        """Check if an attribute name is service-related and should be excluded from serialization."""
        service_attrs = self._get_service_related_attrs()
        return (attr_name in service_attrs or 
                (attr_name.startswith('_') and not attr_name.startswith('__') and attr_name.endswith('_service')))

    def _clean_for_serialization(self, obj):
        """Helper method to clean an object for serialization by removing service-related attributes."""
        from copy import copy
        clean_obj = copy(obj)
        
        # Remove service-related attributes
        for attr_name in dir(clean_obj):
            if self._is_service_related_attr(attr_name) and hasattr(clean_obj, attr_name):
                delattr(clean_obj, attr_name)
        
        return clean_obj

    def __getstate__(self):
        """Get state for pickling, preserving computed properties."""
        state = self.__dict__.copy()
        state.pop("_services", None)

        source = self._get_computed_properties_source()
        if source is not None:
            # Automatically detect computed properties
            for attr_name in dir(source):
                if not attr_name.startswith('__'):  # Skip magic methods
                    try:
                        value = getattr(source, attr_name)
                        if self._is_computed_property(attr_name, value):
                            # Convert value to serializable format if needed
                            state[attr_name] = self._make_serializable(value)
                    except (AttributeError, TypeError, ValueError):
                        # Skip attributes that can't be accessed or raise errors
                        continue
        
        # Remove other service-related attributes using the shared logic
        for attr_name in list(state.keys()):
            if self._is_service_related_attr(attr_name):
                state.pop(attr_name, None)
        
        # Convert all remaining values to serializable format
        for key, value in state.items():
            state[key] = self._make_serializable(value)
        
        return state

    def __setstate__(self, state):
        """Set state after unpickling, restoring computed properties."""
        # Process state to handle serialization placeholders
        processed_state = {}
        for attr_name, value in state.items():
            if isinstance(value, dict) and value.get('_serialization_placeholder', False):
                # Skip serialization placeholders - these objects will be reconstructed
                continue
            processed_state[attr_name] = value
            
        self.__dict__.update(processed_state)
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = {}
        
        # Store computed properties temporarily for later restoration
        self._computed_properties_to_restore = {}
        for attr_name, value in state.items():
            if self._is_computed_property(attr_name, value) and not (isinstance(value, dict) and value.get('_serialization_placeholder', False)):
                self._computed_properties_to_restore[attr_name] = value
        
        # Mark that this object needs reconstruction of excluded objects
        self._needs_reconstruction = True

    def _reconstruct_excluded_objects(self):
        """Reconstruct objects that were excluded during serialization.
        
        This method is called after services are set up to reconstruct objects
        that couldn't be serialized (like Numba-compiled functions, generators, etc.).
        Subclasses can override this method to implement specific reconstruction logic.
        """
        if not hasattr(self, '_needs_reconstruction') or not self._needs_reconstruction:
            return
            
        # Reconstruct common excluded objects
        self._reconstruct_generators()

    def _reconstruct_generators(self):
        """Reconstruct generator objects that were excluded during serialization.
        
        This method resets generator attributes to None so they get lazily reconstructed
        when accessed. This is a common pattern for objects containing Numba-compiled functions.
        """
        # Reset generator in dynamics service if it exists
        if hasattr(self, 'dynamics') and self.dynamics is not None:
            if hasattr(self.dynamics, '_generator'):
                self.dynamics._generator = None
                
        # Reset generator in other services if they exist
        for service_name in ['_correction', '_continuation', '_conversion', '_ham_conversion', 
                            '_ham_dynamics', '_ham_pipeline', '_lgf_persistence']:
            if hasattr(self, service_name):
                service = getattr(self, service_name)
                if service is not None and hasattr(service, '_generator'):
                    service._generator = None

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
        3. Resets the dynamics cache if available (but preserves computed properties)
        4. Restores computed properties from serialization if available
        
        Parameters
        ----------
        services : _ServiceBundleBase
            The service bundle to set up
        """
        self._services = services
        self._bind_services()
        
        # Restore computed properties if they were stored during deserialization
        if hasattr(self, '_computed_properties_to_restore'):
            target = self._set_computed_properties_target()
            if target is not None:
                for attr_name, value in self._computed_properties_to_restore.items():
                    if hasattr(target, attr_name):
                        try:
                            setattr(target, attr_name, value)
                        except (AttributeError, TypeError):
                            # Skip properties that can't be set (e.g., read-only properties)
                            continue
            # Clean up the temporary storage
            delattr(self, '_computed_properties_to_restore')
        
        # Reset dynamics cache if dynamics service is available, but preserve computed properties
        if hasattr(self, '_dynamics') and self._dynamics is not None:
            # Check if dynamics service has computed properties that should be preserved
            has_computed_properties = any(
                hasattr(self._dynamics, attr) and getattr(self._dynamics, attr) is not None
                for attr in ['_period', '_trajectory', '_times', '_stability_info']
            )
            
            if not has_computed_properties:
                self._dynamics.reset()
        
        # Reconstruct excluded objects if needed
        self._reconstruct_excluded_objects()
        
        # Clear the reconstruction flag
        if hasattr(self, '_needs_reconstruction'):
            delattr(self, '_needs_reconstruction')

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
        if not hasattr(obj, '_services') or obj._services is None:
            services = services_factory(obj)
            obj._setup_services(services)
        else:
            obj._unpack_services()
        
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
