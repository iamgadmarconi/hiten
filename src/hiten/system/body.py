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
    """Celestial body container bound to a lightweight persistence adapter.
    
    Parameters
    ----------
    name : str
        The name of the body.
    mass : float
        The mass of the body.
    radius : float
        The radius of the body.
    color : str
        The color of the body.
    parent : :class:`~hiten.system.body.Body`
        The parent of the body.
    """

    def __init__(self, name: str, mass: float, radius: float, color: str = "#000000", parent: Optional["Body"] = None):
        self._name = name
        self._mass = mass
        self._radius = radius
        self._color = color
        self._parent = parent if parent is not None else self
        
        services = _BodyServices.default(self)
        super().__init__(services)


    def __str__(self) -> str:
        parent_desc = f"orbiting {self.parent.name}" if self.parent is not self else "(Primary)"
        return f"{self.name} {parent_desc}"

    def __repr__(self) -> str:
        parent_repr = f", parent=Body(name='{self.parent.name}', ...)" if self.parent is not self else ""

        return f"Body(name={self.name!r}, mass={self.mass}, radius={self.radius}, color={self.color!r}{parent_repr})"

    @property
    def name(self) -> str:
        """Return the name of the body."""
        return self.dynamics.name

    @property
    def mass(self) -> float:
        """Return the mass of the body in kg."""
        return self.dynamics.mass

    @property
    def radius(self) -> float:
        """Return the radius of the body in km."""
        return self.dynamics.radius
    
    @property
    def color(self) -> str:
        """Return the color of the body."""
        return self.dynamics.color

    @property
    def parent(self) -> "Body":
        """Return the parent of the body."""
        return self.dynamics.parent

    def __setstate__(self, state):
        """Restore the Body instance after unpickling.

        The heavy, non-serialisable dynamics is reconstructed lazily
        using the stored value of name, mass, radius, color and parent.
        
        Parameters
        ----------
        state : dict[str, Any]
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
