r"""
hiten.system.body
===========

Light-weight representation of a celestial body participating in a circular restricted three 
body problem (CR3BP) or standalone dynamical simulation.

The module defines the :pyclass:`Body` class, a minimal container that stores
basic physical quantities and plotting attributes while preserving the
hierarchical relation to a central body through the :pyattr:`Body.parent`
attribute.  Instances are used across the project to compute the mass
parameter :math:`\mu` and to provide readable identifiers in logs, plots and
high-precision calculations.
"""

from __future__ import annotations

from typing import Optional

from hiten.utils.log_config import logger


class Body(object):
    r"""
    Celestial body container.

    Parameters
    ----------
    name : str
        Human-readable identifier, for example "Earth" or "Sun".
    mass : float
        Gravitational mass in SI units (kilograms).
    radius : float
        Mean equatorial radius in SI units (metres).
    color : str, optional
        Hexadecimal RGB string used for visualisation. Default is ``"#000000"``.
    parent : Body, optional
        Central :pyclass:`Body` that this instance orbits.  If *None*, the
        object is treated as the primary and :pyattr:`parent` is set to the
        instance itself.

    Attributes
    ----------
    name : str
        Same as the *name* parameter.
    mass : float
        Same as the *mass* parameter.
    radius : float
        Same as the *radius* parameter.
    color : str
        Colour assigned for plotting purposes.
    parent : Body
        Central body around which this instance revolves.

    Notes
    -----
    The class performs no unit or consistency checks; the responsibility of
    providing coherent values lies with the caller.

    Returns
    -------
    None

    Raises
    ------
    None

    Examples
    --------
    >>> sun = Body("Sun", 1.98847e30, 6.957e8, color="#FDB813")
    >>> earth = Body("Earth", 5.9722e24, 6.371e6, parent=sun)
    >>> print(earth)
    Earth orbiting Sun
    """
    name: str
    mass: float
    radius: float
    color: str
    parent: Body # A body's parent is always another Body instance (itself if it's the primary)

    def __init__(self, name: str, mass: float, radius: float, color: Optional[str] = None, parent: Optional[Body] = None):
        self.name = name
        self.mass = mass
        self.radius = radius
        self.color = color if color else "#000000"
        self.parent = parent if parent else self

        parent_name = self.parent.name if self.parent is not self else "None"
        logger.info(f"Created Body: name='{self.name}', mass={self.mass}, radius={self.radius}, color='{self.color}', parent='{parent_name}'")

    def __str__(self) -> str:
        r"""
        Return a concise human-readable description of the body.

        Returns
        -------
        str
            ``"<name> orbiting <parent>"`` when the body orbits another, or
            ``"<name> (Primary)"`` when it is the primary.
        """
        parent_desc = f"orbiting {self.parent.name}" if self.parent is not self else "(Primary)"
        return f"{self.name} {parent_desc}"

    def __repr__(self) -> str:
        r"""
        Unambiguous representation that can be evaluated by :pyfunc:`eval`.

        Returns
        -------
        str
            Python expression that recreates the :pyclass:`Body` instance.
        """
        parent_repr = f"parent={self.parent.name!r}" if self.parent is not self else "parent=self"
        return f"Body(name={self.name!r}, mass={self.mass}, radius={self.radius}, color={self.color!r}, {parent_repr})"
