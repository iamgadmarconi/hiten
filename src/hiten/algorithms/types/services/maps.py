"""Adapters supporting center manifold numerics and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal

import numpy as np

from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)

from hiten.utils.io.map import load_poincare_map, save_poincare_map

if TYPE_CHECKING:
    pass



class _MapPersistenceService(_PersistenceServiceBase):
    """Handle persistence for map objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda map, path, **kw: save_poincare_map(map, Path(path), **kw),
            load_fn=lambda path, **kw: load_poincare_map(Path(path), **kw),
        )


class _MapDynamicsServiceBase(_DynamicsServiceBase):
    """Handle dynamics for map objects."""

    def __init__(self, map) -> None:
        super().__init__(map)

    