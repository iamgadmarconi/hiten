"""Adapters for orbit family persistence and services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hiten.algorithms.types.adapters.base import (_PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.utils.io.family import load_family, load_family_inplace, save_family


class _OrbitFamilyPersistenceAdapter(_PersistenceAdapterMixin):
    """Handle HDF5 serialisation for orbit families."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda family, path, **kw: save_family(family, Path(path), **kw),
            load_fn=lambda path, **kw: load_family(Path(path), **kw),
            load_inplace_fn=lambda family, path, **kw: load_family_inplace(family, Path(path), **kw),
        )


@dataclass
class _OrbitFamilyServices(_ServiceBundleBase):
    persistence: _OrbitFamilyPersistenceAdapter

    @classmethod
    def default(cls) -> "_OrbitFamilyServices":
        return cls(persistence=_OrbitFamilyPersistenceAdapter())

