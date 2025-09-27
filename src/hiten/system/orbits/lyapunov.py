"""Periodic Lyapunov orbits of the circular restricted three-body problem.

This module supplies concrete realisations of :class:`~hiten.system.orbits.base.PeriodicOrbit`
corresponding to the planar families around the collinear libration points
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

from typing import TYPE_CHECKING, Optional, Sequence

from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import _OrbitContinuationConfig
    from hiten.algorithms.corrector.config import _OrbitCorrectionConfig


class LyapunovOrbit(PeriodicOrbit):
    """
    Planar Lyapunov family around a collinear libration point.

    The orbit lies in the (x, y) plane and is symmetric with respect to
    the x-axis. A linear analytical approximation is used to build the
    initial guess which is subsequently refined by a differential corrector.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Target collinear libration point around
        which the orbit is computed.
    amplitude_x : float, optional
        Requested amplitude Ax along the x-direction in nondimensional units.
        Required if initial_state is None.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector
        (x, y, z, vx, vy, vz) expressed in synodic
        coordinates in nondimensional units. If None, an analytical guess is generated.

    Attributes
    ----------
    amplitude_x : float
        Requested amplitude Ax along the x-direction (nondimensional units).
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Equilibrium point about which the orbit is continued.

    Raises
    ------
    TypeError
        If *libration_point* is not an instance of
        :class:`~hiten.system.libration.collinear.CollinearPoint`.
    NotImplementedError
        If the selected point corresponds to L3, which is not
        supported for Lyapunov orbits.
    ValueError
        If conflicting parameters are provided or required parameters are missing.
    """
    
    _family = "lyapunov"
    
    def __init__(
            self, 
            libration_point: LibrationPoint, 
            amplitude_x: Optional[float] = None,
            initial_state: Optional[Sequence[float]] = None
        ):

        super().__init__(libration_point, initial_state, amplitude_x=amplitude_x)

    @property
    def amplitude(self) -> float:
        """(Read-only) Current x-amplitude relative to the libration point.
        
        Returns
        -------
        float
            The x-amplitude in nondimensional units.
        """
        return self.dynamics.amplitude

    @property
    def correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for planar Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The correction configuration for Lyapunov orbits.
        """
        return self._correction.correction_config

    @correction_config.setter
    def correction_config(self, value: "_OrbitCorrectionConfig"):
        """Set the correction configuration."""
        self._correction.correction_config = value

    @property
    def continuation_config(self) -> "_OrbitContinuationConfig":
        """Provides the continuation configuration for Lyapunov orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The continuation configuration for Lyapunov orbits.
        """
        return self._continuation.continuation_config

    @continuation_config.setter
    def continuation_config(self, value: "_OrbitContinuationConfig"):
        """Set the continuation configuration."""
        self._continuation.continuation_config = value
