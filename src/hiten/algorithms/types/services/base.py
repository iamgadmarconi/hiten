"""Base adapter abstractions bridging system facades and algorithm engines.

These helpers standardise the way higher-level code constructs backends,
interfaces, and engines without exposing algorithm internals to the user
facing layers. Concrete adapters should inherit from the provided mixins
and specialise the factory methods for their specific domain objects.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

CacheValueT = TypeVar("CacheValueT")


class _PersistenceServiceBase(ABC):
    """Mixin offering a uniform persistence API around plain callables."""

    def __init__(
        self,
        *,
        save_fn: Callable[..., Any],
        load_fn: Callable[..., Any],
        load_inplace_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._save_fn = save_fn
        self._load_fn = load_fn
        self._load_inplace_fn = load_inplace_fn

    def save(self, target: Any, file_path: Any, **kwargs) -> Any:
        return self._save_fn(target, file_path, **kwargs)

    def load(self, file_path: Any, **kwargs) -> Any:
        return self._load_fn(file_path, **kwargs)

    def load_inplace(self, target: Any, file_path: Any, **kwargs) -> Any:
        if self._load_inplace_fn is None:
            raise NotImplementedError("load_inplace is not supported by this adapter")
        self._load_inplace_fn(target, file_path, **kwargs)
        return target


class _CacheServiceBase(Generic[CacheValueT]):
    """Helper providing lazy caching for dynamics-oriented adapters.
    
    Attributes
    ----------
    _cache : Dict[Any, CacheValueT]
        The cache dictionary.
    """
    def __init__(self) -> None:
        self._cache: Dict[Any, CacheValueT] = {}

    def get_or_create(self, key: Any, factory: Callable[[], CacheValueT]) -> CacheValueT:
        """Get or create a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        factory : Callable[[], CacheValueT]
            The factory function to create the cache value.
        """
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def get(self, key: Any) -> CacheValueT:
        """Get a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        """
        return self._cache[key]

    def set(self, key: Any, value: CacheValueT) -> CacheValueT:
        """Set a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        value : CacheValueT
            The cache value.
        """
        self._cache[key] = value
        return value

    def reset(self, key: Optional[Any] = None) -> None:
        """Reset the cache.
        
        Parameters
        ----------
        key : Any, optional
            The cache key to reset. If None, the entire cache is cleared.
        """
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def make_key(self, *args: Any) -> tuple[Any, ...]:
        """Create a cache key with the current function name as the first element.

        This helps avoid cache key collisions between different methods.
        
        Parameters
        ----------
        *args
            Additional arguments to include in the cache key.
            
        Returns
        -------
        tuple
            Cache key with function name as first element.
        """
        return (inspect.currentframe().f_back.f_code.co_name, *args)


class _DynamicsServiceBase(ABC):
    """Mixin offering a uniform dynamics API around plain callables."""

    def __init__(self, domain_obj: Any) -> None:
        self._domain_obj = domain_obj
        self._cache = _CacheServiceBase()
    
    def get_or_create(self, key: Any, factory: Callable[[], CacheValueT]) -> CacheValueT:
        """Get or create a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        factory : Callable[[], CacheValueT]
            The factory function to create the cache value.
        """
        return self._cache.get_or_create(key, factory)

    def make_key(self, *args: Any) -> tuple[Any, ...]:
        """Create a cache key with the current function name as the first element.
        
        Parameters
        ----------
        *args
            Additional arguments to include in the cache key.
        """
        return (inspect.currentframe().f_back.f_code.co_name, self._domain_obj, *args)

    def __getitem__(self, key: Any) -> CacheValueT:
        """Get a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        """
        return self._cache.get(key)

    def __setitem__(self, key: Any, value: CacheValueT) -> CacheValueT:
        """Set a cache value.
        
        Parameters
        ----------
        key : Any
            The cache key.
        value : CacheValueT
            The cache value.
        """
        return self._cache.set(key, value)

    def reset(self, key: Optional[Any] = None) -> None:
        """Reset the cache.
        
        Parameters
        ----------
        key : Any, optional
            The cache key to reset. If None, the entire cache is cleared.
        """
        return self._cache.reset(key)



class _ServiceBundleBase(ABC):
    """Lightweight helper for service bundles offering ergonomic helpers.
    """

    __slots__ = ()

    @classmethod
    @abstractmethod
    def default(cls) -> "_ServiceBundleBase":
        """Create a default service bundle."""
        pass

    @classmethod
    @abstractmethod
    def with_shared_dynamics(cls, dynamics: "_DynamicsServiceBase") -> "_ServiceBundleBase":
        """Create a service bundle with a shared dynamics service."""
        pass