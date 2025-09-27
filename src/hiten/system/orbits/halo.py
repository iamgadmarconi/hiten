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

from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import _OrbitContinuationConfig
    from hiten.algorithms.corrector.config import _OrbitCorrectionConfig


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
    
    amplitude_z: Optional[float] # Amplitude of the halo orbit
    zenith: Optional[Literal["northern", "southern"]]

    def __init__(
            self, 
            libration_point: LibrationPoint, 
            amplitude_z: Optional[float] = None,
            zenith: Optional[Literal["northern", "southern"]] = None,
            initial_state: Optional[Sequence[float]] = None
        ):
        super().__init__(libration_point, initial_state=initial_state, amplitude_z=amplitude_z, zenith=zenith)


    @property
    def amplitude(self) -> float:
        """(Read-only) Current z-amplitude of the orbit in the synodic frame.
        
        Returns
        -------
        float
            The z-amplitude in nondimensional units.
        """
        return self.dynamics.amplitude

    @property
    def zenith(self) -> Literal["northern", "southern"]:
        """(Read-only) Current zenith of the orbit.
        
        Returns
        -------
        Literal["northern", "southern"]
            The orbit zenith.
        """
        return self.dynamics.zenith

    @property
    def correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The correction configuration for halo orbits.
        """
        return self._correction.correction_config

    @correction_config.setter
    def correction_config(self, value: "_OrbitCorrectionConfig"):
        """Set the correction configuration."""
        self._correction.correction_config = value

    @property
    def continuation_config(self) -> "_OrbitContinuationConfig":
        """Provides the continuation configuration for halo orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The continuation configuration for halo orbits.
        """
        return self._continuation.continuation_config

    @continuation_config.setter
    def continuation_config(self, value: "_OrbitContinuationConfig"):
        """Set the continuation configuration."""
        self._continuation.continuation_config = value