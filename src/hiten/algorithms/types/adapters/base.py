"""Base adapter abstractions bridging system facades and algorithm engines.

These helpers standardise the way higher-level code constructs backends,
interfaces, and engines without exposing algorithm internals to the user
facing layers. Concrete adapters should inherit from the provided mixins
and specialise the factory methods for their specific domain objects.
"""

import inspect
from abc import ABC
from dataclasses import fields, replace
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

CacheValueT = TypeVar("CacheValueT")
ServiceBundleT = TypeVar("ServiceBundleT", bound="_ServiceBundleBase")


class _PersistenceAdapterMixin(ABC):
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


class _CachedDynamicsAdapter(Generic[CacheValueT]):
    """Helper providing lazy caching for dynamics-oriented adapters."""

    def __init__(self) -> None:
        self._cache: Dict[Any, CacheValueT] = {}

    def _get_or_create(self, key: Any, factory: Callable[[], CacheValueT]) -> CacheValueT:
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def _set_cache(self, key: Any, value: CacheValueT) -> CacheValueT:
        self._cache[key] = value
        return value

    def reset_cache(self, key: Optional[Any] = None) -> None:
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def _make_cache_key(self, *args: Any) -> tuple[Any, ...]:
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


class _ServiceBundleBase:
    """Lightweight helper for service bundles offering ergonomic helpers."""

    __slots__ = ()

    def replace(self: ServiceBundleT, **changes: Any) -> ServiceBundleT:
        return replace(self, **changes)

    def as_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def get_or_create(self, key: Any, factory: Callable[[], Any]) -> Any:
        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = {}
            setattr(self, "_cache", cache)
        if key not in cache:
            cache[key] = factory()
        return cache[key]