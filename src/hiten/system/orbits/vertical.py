"""Periodic vertical orbits of the circular restricted three-body problem.

This module supplies concrete realisations of :class:`~hiten.system.orbits.base.PeriodicOrbit`
corresponding to the vertical family around the collinear libration points
L1 and L2. Each class provides an analytical first guess together with a
customised differential corrector that exploits the symmetries of the family.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from typing import Optional, Sequence, TYPE_CHECKING

from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.system.libration.collinear import CollinearPoint


class VerticalOrbit(PeriodicOrbit):
    """
    Vertical family about a collinear libration point.

    The orbit oscillates out of the synodic plane and is symmetric with
    respect to the x-z plane. Initial-guess generation is not
    yet available.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Target collinear libration point around which the orbit is computed.
    initial_state : Sequence[float] or None, optional
        Six-dimensional initial state vector [x, y, z, vx, vy, vz] in
        nondimensional units. If None, must be provided manually.

    Notes
    -----
    The implementation of the analytical seed and the Jacobian adjustment for
    the vertical family is work in progress.
    """
    
    _family = "vertical"

    def __init__(self, libration_point: "CollinearPoint", initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state)
