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

from hiten.algorithms.utils.core import _HitenBase
from hiten.utils.io.body import load_body, load_body_inplace, save_body


class Body(_HitenBase):
    """
    Celestial body container.

    Parameters
    ----------
    name : str
        Human-readable identifier, for example "Earth" or "Sun".
    mass : float
        Gravitational mass in kilograms.
    radius : float
        Mean equatorial radius in metres.
    color : str, optional
        Hexadecimal RGB string used for visualisation. Default is "#000000".
    parent : :class:`~hiten.system.body.Body`, optional
        Internal parameter for parent body specification. If None, the
        object is treated as the primary and parent is set to the
        instance itself.

    Attributes
    ----------
    name : str
        Human-readable identifier.
    mass : float
        Gravitational mass in kilograms.
    radius : float
        Mean equatorial radius in metres.
    color : str
        Colour assigned for plotting purposes.
    parent : :class:`~hiten.system.body.Body`
        Central body around which this instance revolves.

    Notes
    -----
    The class performs no unit or consistency checks; the responsibility of
    providing coherent values lies with the caller.

    Examples
    --------
    >>> sun = Body("Sun", 1.98847e30, 6.957e8, color="#FDB813")
    >>> earth = Body("Earth", 5.9722e24, 6.371e6, parent=sun)
    >>> print(earth)
    Earth orbiting Sun
    """

    def __init__(self, name: str, mass: float, radius: float, color: str = "#000000", parent: Optional["Body"] = None):
        super().__init__()
        self._name = name
        self._mass = mass
        self._radius = radius
        self._color = color
        self._parent = parent or self

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
        save_body(self, Path(file_path))

    @classmethod
    def load(cls, file_path: str | Path, **kwargs) -> "Body":
        return load_body(Path(file_path))

    def load_inplace(self, file_path: str | Path) -> "Body":
        load_body_inplace(self, Path(file_path))
        return self
