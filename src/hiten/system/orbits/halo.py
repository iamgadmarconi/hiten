"""Generation and refinement of halo periodic orbits about the collinear
libration points of the Circular Restricted Three-Body Problem (CRTBP).

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
collinear libration points".
"""

from typing import TYPE_CHECKING, Literal, Optional, Sequence

from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.system.libration.collinear import CollinearPoint


class HaloOrbit(PeriodicOrbit):
    """
    Halo orbit class.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Target collinear libration point around which the halo orbit is computed.
    amplitude_z : float, optional
        z-amplitude of the halo orbit in the synodic frame (nondimensional units).
        Required if initial_state is None.
    zenith : {'northern', 'southern'}, optional
        Indicates the symmetry branch with respect to the xy-plane.
        Required if initial_state is None.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector [x, y, z, vx, vy, vz] in the rotating
        synodic frame. When None an analytical initial guess is generated
        from amplitude_z and zenith.

    Attributes
    ----------
    amplitude_z : float or None
        z-amplitude of the halo orbit in the synodic frame (nondimensional units).
    zenith : {'northern', 'southern'} or None
        Indicates the symmetry branch with respect to the xy-plane.

    Raises
    ------
    ValueError
        If the required amplitude or branch is missing and initial_state
        is None.
    TypeError
        If libration_point is not an instance of CollinearPoint.
    """
    
    _family = "halo"

    def __init__(
            self, 
            libration_point: "CollinearPoint", 
            initial_state: Optional[Sequence[float]] = None,
            amplitude_z: Optional[float] = None,
            zenith: Optional[Literal["northern", "southern"]] = None
        ):

        self._amplitude_z = amplitude_z
        self._zenith = zenith

        super().__init__(libration_point, initial_state=initial_state)

    @property
    def zenith(self) -> Literal["northern", "southern"]:
        """(Read-only) Current zenith of the orbit.
        
        Returns
        -------
        Literal["northern", "southern"]
            The orbit zenith.
        """
        return self.dynamics.zenith
