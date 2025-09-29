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
    """Mixin offering a uniform persistence API around plain callables.
    
    Parameters
    ----------
    save_fn : Callable[..., Any]
        The function to save the object.
    load_fn : Callable[..., Any]
        The function to load the object.
    load_inplace_fn : Optional[Callable[..., Any]] = None
        The function to load the object in place.
    """

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

    def save(self, target: Any, filepath: Any, **kwargs) -> Any:
        """Save the object to a file.
        
        Parameters
        ----------
        target : Any
            The object to save.
        filepath : Any
            The path to the file to save the object to.
        """
        return self._save_fn(target, filepath, **kwargs)

    def load(self, filepath: Any, **kwargs) -> Any:
        """Load the object from a file.
        
        Parameters
        ----------
        filepath : Any
            The path to the file to load the object from.
        """
        return self._load_fn(filepath, **kwargs)

    def load_inplace(self, target: Any, filepath: Any, **kwargs) -> Any:
        """Load the object from a file in place.
        
        Parameters
        ----------
        target : Any
            The object to load the object into.
        filepath : Any
            The path to the file to load the object from.
        """
        if self._load_inplace_fn is None:
            raise NotImplementedError("load_inplace is not supported by this adapter")
        self._load_inplace_fn(target, filepath, **kwargs)
        return target


class _CacheServiceBase(Generic[CacheValueT]):
    """Helper providing lazy caching for dynamics-oriented adapters.
    
    Parameters
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
        Automatically handles unhashable types by converting them to hashable equivalents.
        
        Parameters
        ----------
        *args
            Additional arguments to include in the cache key.
            
        Returns
        -------
        tuple
            Cache key with function name as first element.
        """
        def _make_hashable(obj: Any) -> Any:
            """Convert unhashable objects to hashable equivalents."""
            # Check if object is actually hashable by trying to hash it
            try:
                hash(obj)
                return obj
            except TypeError:
                # Object is not hashable, convert it
                if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                    # Convert iterables (like numpy arrays) to tuples
                    try:
                        return tuple(_make_hashable(item) for item in obj)
                    except TypeError:
                        # If we can't iterate, convert to string representation
                        return str(obj)
                else:
                    # For other unhashable types, use string representation
                    return str(obj)
        
        hashable_args = [_make_hashable(arg) for arg in args]
        return (inspect.currentframe().f_back.f_code.co_name, *hashable_args)


class _DynamicsServiceBase(ABC):
    """Mixin offering a uniform dynamics API around plain callables.
    
    Parameters
    ----------
    domain_obj : Any
        The domain object.

    Attributes
    ----------
    cache : _CacheServiceBase
        The cache service.
    """

    def __init__(self, domain_obj: Any) -> None:
        self._domain_obj = domain_obj
        self._cache = _CacheServiceBase()

    @property
    def domain_obj(self) -> Any:
        """The domain object."""
        return self._domain_obj

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
        return self._cache.make_key(self.domain_obj, *args)

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
    
    def __getstate__(self):
        """Get state for pickling.
        
        Preserves computed properties and cache that are expensive to recompute.
        Excludes only the domain_obj to avoid circular references.
        """
        state = self.__dict__.copy()
        state.pop('_domain_obj', None)
        return state
    
    def __setstate__(self, state):
        """Set state after unpickling.
        
        Restores computed properties and cache, reinitializes only domain_obj.
        """
        self.__dict__.update(state)
        # Only reinitialize domain_obj (will be set by the parent object)
        self._domain_obj = None


class _ServiceBundleBase(ABC):
    """Lightweight helper for service bundles offering ergonomic helpers.

    Parameters
    ----------
    domain_obj : Any
        The domain object.
    """

    def __init__(self, domain_obj: Any) -> None:
        self._domain_obj = domain_obj

    @property
    def domain_obj(self) -> Any:
        """The domain object."""
        return self._domain_obj

    def __getitem__(self, service_name: str):
        """Get a service by name.
        
        This method allows dynamic access to any service in the service bundle.
        
        Parameters
        ----------
        service_name : str
            The name of the service to retrieve (e.g., 'correction', 'continuation')
            
        Returns
        -------
        Any
            The requested service instance
            
        Raises
        ------
        AttributeError
            If the requested service is not available
        """
        if hasattr(self, service_name):
            return getattr(self, service_name)
        raise AttributeError(f"No '{service_name}' service available in this service bundle")

    @classmethod
    @abstractmethod
    def default(cls, domain_obj: Any) -> "_ServiceBundleBase":
        """Create a default service bundle.
        
        Parameters
        ----------
        domain_obj : Any
            The domain object.
        """
        pass

    @classmethod
    @abstractmethod
    def with_shared_dynamics(cls, domain_obj: Any, dynamics: "_DynamicsServiceBase") -> "_ServiceBundleBase":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        domain_obj : Any
            The domain object.
        dynamics : _DynamicsServiceBase
            The dynamics service.
        """
        pass
