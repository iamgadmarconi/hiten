"""Abstract base class for Hiten classes.

This module provides the abstract base class for all Hiten classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

import pandas as pd

from hiten.algorithms.types.exceptions import EngineError


class _HitenBase(ABC):
    """Abstract base class for public Hiten classes.
    """

    def __init__(self):
        self._cache = {}

    def cache_get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache.

        Parameters
        ----------
        key : Any
            The cache key.
        default : Any, optional
            The default value to return if the key is not found.
            
        Returns
        -------
        Any
            The cached value or the default value if the key is not found.
        """
        return self._cache.get(key, default)
    
    def cache_set(self, key: Any, value: Any) -> Any:
        """Set item in cache.

        Parameters
        ----------
        key : Any
            The cache key.
        value : Any
            The value to cache.
            
        Returns
        -------
        Any
            The cached value.
        """
        self._cache[key] = value
        return value
    
    def cache_clear(self) -> None:
        """Clear cache.

        This method resets all cached properties to None, forcing them to be
        recomputed on next access.
        """
        self._cache.clear()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        """Get state for pickling.
        """
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Set state after unpickling.
        """
        self.__dict__.update(state)
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = {}

    @abstractmethod
    def save(self, file_path: str | Path, **kwargs) -> None:
        """Save the object to a file.

        Parameters
        ----------
        file_path : str or Path
            The path to the file to save the object to.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, file_path: str | Path, **kwargs) -> "_HitenBase":
        """Load the object from a file.
        
        Parameters
        ----------
        file_path : str or Path
            The path to the file to load the object from.
        **kwargs
            Additional keyword arguments passed to the load method.
            
        Returns
        -------
        :class:`~hiten.system.base._HitenBase`
            The loaded object.
        """
        ...

    def to_csv(self, file_path: str | Path, **kwargs) -> None:
        """Save the object to a CSV file.

        Parameters
        ----------
        file_path : str or Path
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


DomainT = TypeVar("DomainT")

ConfigT = TypeVar("ConfigT", bound=Union["_HitenBaseConfig", None])

ProblemT = TypeVar("ProblemT", bound="_HitenBaseProblem")

ResultT = TypeVar("ResultT", bound="_HitenBaseResults")

BackendT = TypeVar("BackendT", bound="_HitenBaseBackend")

OutputsT = TypeVar("OutputsT")

InterfaceT = TypeVar("InterfaceT", bound="_HitenBaseInterface[Any, ConfigT, ProblemT, ResultT, OutputsT]")

EngineT = TypeVar("EngineT", bound="_HitenBaseEngine[ProblemT, ResultT, OutputsT]")


@dataclass(frozen=True)
class BackendCall:
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


class _HitenBaseInterface(Generic[DomainT, ConfigT, ProblemT, ResultT, OutputsT], ABC):
    """Shared contract for translating between domain objects and backends."""

    def __init__(self) -> None:
        self._config: ConfigT | None = None

    @property
    def current_config(self) -> ConfigT | None:
        return self._config

    @abstractmethod
    def create_problem(self, *, config: ConfigT | None = None, **kwargs) -> ProblemT:
        """Compose an immutable problem payload for the backend."""

    @abstractmethod
    def to_backend_inputs(self, problem: ProblemT) -> BackendCall:
        """Translate a problem into backend invocation arguments."""

    def from_domain(self, *, config: ConfigT | None = None, **kwargs) -> BackendCall:
        """Convenience helper to build backend inputs directly from config."""
        problem = self.create_problem(config=config, **kwargs)
        return self.to_backend_inputs(problem)

    def to_domain(self, outputs: OutputsT, *, problem: ProblemT) -> Any:
        """Optional hook to mutate or derive domain artefacts from outputs."""
        return None

    @abstractmethod
    def to_results(self, outputs: OutputsT, *, problem: ProblemT) -> ResultT:
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
        backend: Any,
        interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT] | None = None,
        backend_method: str = "solve",
    ) -> None:
        self._backend = backend
        self._backend_method = backend_method
        self._interface = interface

    @property
    def backend(self) -> Any:
        return self._backend

    def with_interface(
        self,
        interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT],
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
        return interface.to_results(outputs, problem=problem)

    def _get_interface(
        self,
        problem: ProblemT,
    ) -> _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT]:
        if self._interface is None:
            raise EngineError(
                f"{self.__class__.__name__} must be configured with an interface before solving."
            )
        return self._interface

    def set_interface(
        self,
        interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT],
    ) -> None:
        self._interface = interface

    def with_interface(
        self,
        interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT],
    ) -> "_HitenBaseEngine[ProblemT, ResultT, OutputsT]":
        self.set_interface(interface)
        return self

    def _before_backend(self, problem: ProblemT, call: BackendCall, interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT]) -> None:
        return None

    def _after_backend_success(self, outputs: OutputsT, *, problem: ProblemT, domain_payload: Any, interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT]) -> None:
        return None

    def _handle_backend_failure(self, exc: Exception, *, problem: ProblemT, call: BackendCall, interface: _HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT]) -> None:
        raise EngineError(exc) from exc

    def _invoke_backend(self, call: BackendCall) -> OutputsT:
        backend_callable = getattr(self._backend, self._backend_method)
        return backend_callable(*call.args, **call.kwargs)


class _HitenBaseBackend(ABC):
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
    ...     def solve(self, **kwargs):
    ...         # Implementation here
    ...         pass
    >>> 
    >>> backend = MyBackend()
    >>> result = backend.solve(input_data)
    """

    def __init__(self) -> None:
        """Initialize the backend."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
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
