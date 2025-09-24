"""Base adapter abstractions bridging system facades and algorithm engines.

These helpers standardise the way higher-level code constructs backends,
interfaces, and engines without exposing algorithm internals to the user
facing layers. Concrete adapters should inherit from the provided mixins
and specialise the factory methods for their specific domain objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .core import (
    _HitenBaseConfig,
    _HitenBaseEngine,
    _HitenBaseInterface,
    _HitenBaseProblem,
    _HitenBaseResults,
)

BackendT = TypeVar("BackendT")
ConfigT = TypeVar("ConfigT", bound=_HitenBaseConfig | None)
ProblemT = TypeVar("ProblemT", bound=_HitenBaseProblem)
ResultT = TypeVar("ResultT", bound=_HitenBaseResults)
OutputsT = TypeVar("OutputsT")
InterfaceT = TypeVar(
    "InterfaceT",
    bound=_HitenBaseInterface[Any, Any, ProblemT, ResultT, OutputsT],
)
EngineT = TypeVar(
    "EngineT",
    bound=_HitenBaseEngine[ProblemT, ResultT, OutputsT],
)
AdapterKeyT = TypeVar("AdapterKeyT")
AdapterT = TypeVar("AdapterT", bound="_HitenAdapter[Any, Any, Any, Any, Any]")


class _HitenAdapter(Generic[BackendT, InterfaceT, EngineT, ProblemT, ResultT], ABC):
    """Base class encapsulating backend, interface, and engine orchestration.

    Concrete adapters supply factories for each layer and may expose helper
    methods that tailor `create_problem` arguments for their specific domains.
    The base implementation focuses on lifecycle management and keeps lazy
    construction logic in a single place.
    """

    def __init__(
        self,
        *,
        backend_factory: Callable[[], BackendT],
        name: Optional[str] = None,
    ) -> None:
        self._backend_factory = backend_factory
        self._backend: Optional[BackendT] = None
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        return self._name

    @property
    def backend(self) -> BackendT:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def reset_backend(self) -> None:
        """Dispose of the cached backend so it can be recreated lazily."""

        self._backend = None

    @abstractmethod
    def build_interface(self, *args, **kwargs) -> InterfaceT:
        """Return a configured interface instance."""

    @abstractmethod
    def build_engine(self, interface: InterfaceT) -> EngineT:
        """Return an engine bound to the provided interface."""

    def create_problem(
        self,
        interface: InterfaceT,
        *,
        config: Optional[_HitenBaseConfig] = None,
        **kwargs,
    ) -> ProblemT:
        """Compose a backend problem using the supplied interface."""

        return interface.create_problem(config=config, **kwargs)

    def solve(
        self,
        interface: InterfaceT,
        problem: ProblemT,
    ) -> ResultT:
        """Execute an engine solve cycle with the provided artefacts."""

        engine = self.build_engine(interface)
        return engine.solve(problem)


class _AdapterRegistry(Generic[AdapterKeyT, AdapterT]):
    """Utility registry that keeps adapter singletons keyed by identifiers."""

    def __init__(self) -> None:
        self._registry: Dict[AdapterKeyT, AdapterT] = {}

    def get_or_create(
        self,
        key: AdapterKeyT,
        factory: Callable[[], AdapterT],
    ) -> AdapterT:
        """Return a cached adapter instance, constructing it if necessary."""

        if key not in self._registry:
            self._registry[key] = factory()
        return self._registry[key]

    def register(self, key: AdapterKeyT, adapter: AdapterT) -> AdapterT:
        """Register an adapter instance explicitly."""

        self._registry[key] = adapter
        return adapter

    def pop(self, key: AdapterKeyT, default: Optional[AdapterT] = None) -> Optional[AdapterT]:
        """Remove an adapter from the registry."""

        return self._registry.pop(key, default)

    def clear(self) -> None:
        """Clear the registry cache."""

        self._registry.clear()

    def __contains__(self, key: AdapterKeyT) -> bool:
        return key in self._registry

    def __len__(self) -> int:
        return len(self._registry)
