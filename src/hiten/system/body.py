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

from dataclasses import dataclass, field
from typing import Optional

from hiten.utils.log_config import logger


@dataclass(frozen=True)
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
    color: str = "#000000"
    parent: Optional[Body] = field(default=None, init=False)
    _parent_input: Optional[Body] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Use object.__setattr__ to bypass frozen=True protection. This is a
        # standard pattern for initializing fields in frozen dataclasses.
        parent_to_set = self._parent_input or self
        object.__setattr__(self, 'parent', parent_to_set)

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
        Unambiguous representation that avoids recursion.

        Notes
        -----
        The representation of a secondary body is intentionally not evaluatable
        with :pyfunc:`eval` because it only includes the parent's name to
        prevent infinite recursion. A primary body's representation, however,
        can be evaluated.

        Returns
        -------
        str
            Python expression that describes the :pyclass:`Body` instance.
        """
        # For the parent, we show its name to avoid recursion.
        # This makes the repr not perfectly eval-able for secondaries, but it's safe.
        if self.parent is self:
            parent_repr = ""
        else:
            parent_repr = f", _parent_input=Body(name='{self.parent.name}', ...)"

        return f"Body(name={self.name!r}, mass={self.mass}, radius={self.radius}, color={self.color!r}{parent_repr})"
