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

from pathlib import Path
from typing import Optional

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.body import (_BodyPersistenceService,
                                                  _BodyServices)


class Body(_HitenBase):
    """Celestial body container bound to a lightweight persistence adapter."""

    def __init__(self, name: str, mass: float, radius: float, color: str = "#000000", parent: Optional["Body"] = None):
        self._name = name
        self._mass = mass
        self._radius = radius
        self._color = color
        self._parent = parent
        
        services = _BodyServices.default(self)
        super().__init__(services)

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
        return self.dynamics.name

    @property
    def mass(self) -> float:
        return self.dynamics.mass

    @property
    def radius(self) -> float:
        return self.dynamics.radius
    
    @property
    def color(self) -> str:
        return self.dynamics.color

    @property
    def parent(self) -> "Body":
        return self.dynamics.parent

    def __setstate__(self, state):
        """Restore the Body instance after unpickling.

        The heavy, non-serialisable dynamics is reconstructed lazily
        using the stored value of name, mass, radius, color and parent.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the Body.
        """
        super().__setstate__(state)
        self._setup_services(_BodyServices.default(self))

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "Body":
        """Load a Body from a file (new instance)."""
        return cls._load_with_services(
            filepath, 
            _BodyPersistenceService(), 
            _BodyServices.default, 
            **kwargs
        )
