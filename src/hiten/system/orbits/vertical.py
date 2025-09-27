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

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from hiten.algorithms.poincare.singlehit.backend import _z_plane_crossing
from hiten.algorithms.types.states import SynodicState
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import _OrbitContinuationConfig
    from hiten.algorithms.corrector.config import _OrbitCorrectionConfig


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

    Attributes
    ----------
    _amplitude_z : float or None
        z-amplitude of the vertical orbit (nondimensional units).

    Notes
    -----
    The implementation of the analytical seed and the Jacobian adjustment for
    the vertical family is work in progress.
    """
    
    _family = "vertical"

    def __init__(self, libration_point: CollinearPoint, initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state)

    @property
    def amplitude(self) -> float:
        """(Read-only) Current z-amplitude of the vertical orbit.
        
        Returns
        -------
        float
            The z-amplitude in nondimensional units.
        """
        return self.dynamics.amplitude

    @property
    def correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for vertical orbits.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The correction configuration for vertical orbits.
        """
        return self._correction.correction_config

    @correction_config.setter
    def correction_config(self, value: "_OrbitCorrectionConfig"):
        """Set the correction configuration."""
        self._correction.correction_config = value

    @property
    def _continuation_config(self) -> "_OrbitContinuationConfig":
        """Default continuation parameter: vary the out-of-plane amplitude.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The continuation configuration for vertical orbits.
        """
        from hiten.algorithms.continuation.config import _OrbitContinuationConfig
        return _OrbitContinuationConfig(state=SynodicState.Z, amplitude=True)
