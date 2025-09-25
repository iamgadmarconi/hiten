"""Light-weight representation of a celestial body participating in a circular 
restricted three body problem (CR3BP) or standalone dynamical simulation.

The module defines the :class:`~hiten.system.body.Body` class, a minimal container that stores
basic physical quantities and plotting attributes while preserving the
hierarchical relation to a central body through the :attr:`~hiten.system.body.Body.parent`
attribute. Instances are used across the project to compute the mass
parameter mu and to provide readable identifiers in logs, plots and
high-precision calculations.

Notes
-----
All masses are expressed in kilograms and radii in metres (SI units).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hiten.algorithms.types.adapters.body import _BodyPersistenceAdapter
from hiten.algorithms.types.core import _HitenBase


@dataclass
class BodyContext:
    persistence: _BodyPersistenceAdapter

    @classmethod
    def default(cls) -> "BodyContext":
        return cls(persistence=_BodyPersistenceAdapter())


class Body(_HitenBase):
    """Celestial body container bound to a lightweight persistence adapter."""

    def __init__(self, name: str, mass: float, radius: float, color: str = "#000000", parent: Optional["Body"] = None, *, context: Optional[BodyContext] = None):
        super().__init__()
        self._name = name
        self._mass = mass
        self._radius = radius
        self._color = color
        self._parent = parent or self
        self._context = context or BodyContext.default()

    def __str__(self) -> str:
        parent_desc = f"orbiting {self.parent.name}" if self.parent is not self else "(Primary)"
        return f"{self.name} {parent_desc}"

    def __repr__(self) -> str:
        if self.parent is self:
            parent_repr = ""
        else:
            parent_repr = f", parent=Body(name='{self.parent.name}', ...)"

        return f"Body(name={self.name!r}, mass={self.mass}, radius={self.radius}, color={self.color!r}{parent_repr})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def radius(self) -> float:
        return self._radius
    
    @property
    def color(self) -> str:
        return self._color

    @property
    def parent(self) -> "Body":
        return self._parent
    
    def save(self, file_path: str | Path, **kwargs) -> None:
        self._context.persistence.save(self, file_path, **kwargs)

    @classmethod
    def load(cls, file_path: str | Path, **kwargs) -> "Body":
        adapter = BodyContext.default().persistence
        body = adapter.load(file_path, **kwargs)
        if isinstance(body, cls) and not getattr(body, "_context", None):
            body._context = BodyContext.default()
        return body

    def load_inplace(self, file_path: str | Path) -> "Body":
        self._context.persistence.load_inplace(self, file_path)
        return self
    
    def __getstate__(self):
        """Custom pickle state that excludes unpicklable context."""
        state = self.__dict__.copy()
        state.pop("_context", None)
        return state
    
    def __setstate__(self, state):
        """Restore state and recreate context."""
        self.__dict__.update(state)
        if not hasattr(self, "_context"):
            self._context = BodyContext.default()