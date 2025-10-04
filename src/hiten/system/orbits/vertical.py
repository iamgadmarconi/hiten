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
    respect to the x-z plane. This is also known as a vertical Lyapunov orbit.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Target collinear libration point around which the orbit is computed.
    amplitude_z : float, optional
        z-amplitude of the vertical orbit in the synodic frame (nondimensional units).
        Required if initial_state is None.
    initial_state : Sequence[float] or None, optional
        Six-dimensional initial state vector [x, y, z, vx, vy, vz] in
        nondimensional units. When None an analytical initial guess is generated
        from amplitude_z.

    Attributes
    ----------
    amplitude_z : float or None
        z-amplitude of the vertical orbit in the synodic frame (nondimensional units).

    Raises
    ------
    ValueError
        If the required amplitude is missing and initial_state is None.
    TypeError
        If libration_point is not an instance of CollinearPoint.

    Notes
    -----
    The third-order analytical approximation is based on Richardson (1980).
    
    References
    ----------
    Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
    collinear libration points". Celestial Mechanics 22 (3):241–253.
    """
    
    _family = "vertical"

    def __init__(
            self, 
            libration_point: "CollinearPoint",
            *,
            initial_state: Optional[Sequence[float]] = None,
            amplitude_z: Optional[float] = None
        ):
        self._amplitude_z = amplitude_z
        super().__init__(libration_point, initial_state)
