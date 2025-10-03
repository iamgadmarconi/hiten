"""Generation and refinement of Lissajous quasi-periodic orbits about the collinear
libration points of the Circular Restricted Three-Body Problem (CRTBP).

Lissajous orbits are quasi-periodic trajectories that result from combining the in-plane
and out-of-plane oscillations with independent frequencies and phase angles. Unlike Halo
orbits which are periodic, Lissajous orbits generally fill a torus around the libration point.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Richardson, D. L. (1980). "Analytic construction of periodic orbits about the
collinear libration points".
"""

from typing import TYPE_CHECKING, Optional, Sequence

from hiten.system.orbits.base import PeriodicOrbit

if TYPE_CHECKING:
    from hiten.system.libration.collinear import CollinearPoint


class LissajousOrbit(PeriodicOrbit):
    """
    Lissajous orbit class for quasi-periodic trajectories around collinear libration points.

    Lissajous orbits combine in-plane and out-of-plane oscillations with independent
    frequencies (lambda and nu) and phase angles (phi and psi). These orbits are
    generally quasi-periodic and fill a torus around the libration point.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        Target collinear libration point around which the Lissajous orbit is computed.
    amplitude_y : float, optional
        y-amplitude of the Lissajous orbit in the synodic frame (nondimensional units).
        Required if initial_state is None.
    amplitude_z : float, optional
        z-amplitude of the Lissajous orbit in the synodic frame (nondimensional units).
        Required if initial_state is None.
    phi : float, optional
        Phase angle for in-plane motion in radians. Default is 0.0.
    psi : float, optional
        Phase angle for out-of-plane motion in radians. Default is 0.0.
    initial_state : Sequence[float] or None, optional
        Six-dimensional state vector [x, y, z, vx, vy, vz] in the rotating
        synodic frame. When None, an analytical initial guess is generated
        from amplitude_y, amplitude_z, phi, and psi.

    Attributes
    ----------
    amplitude_y : float or None
        y-amplitude of the Lissajous orbit in the synodic frame (nondimensional units).
    amplitude_z : float or None
        z-amplitude of the Lissajous orbit in the synodic frame (nondimensional units).
    phi : float
        Phase angle for in-plane motion in radians.
    psi : float
        Phase angle for out-of-plane motion in radians.

    Raises
    ------
    ValueError
        If the required amplitudes are missing and initial_state is None.
    TypeError
        If libration_point is not an instance of CollinearPoint.

    Notes
    -----
    The third-order analytical approximation is based on Richardson's work and
    provides a good initial guess for quasi-periodic Lissajous trajectories.
    For periodic Lissajous orbits (when lambda/nu is rational), differential
    correction can be applied to refine the solution.
    """
    
    _family = "lissajous"

    def __init__(
            self, 
            libration_point: "CollinearPoint", 
            initial_state: Optional[Sequence[float]] = None,
            amplitude_y: Optional[float] = None,
            amplitude_z: Optional[float] = None,
            phi: float = 0.0,
            psi: float = 0.0
        ):

        self._amplitude_y = amplitude_y
        self._amplitude_z = amplitude_z
        self._phi = phi
        self._psi = psi

        super().__init__(libration_point, initial_state=initial_state)

    @property
    def phi(self) -> float:
        """(Read-only) Phase angle for in-plane motion.
        
        Returns
        -------
        float
            Phase angle in radians.
        """
        return self.dynamics.phi

    @property
    def psi(self) -> float:
        """(Read-only) Phase angle for out-of-plane motion.
        
        Returns
        -------
        float
            Phase angle in radians.
        """
        return self.dynamics.psi

    @property
    def amplitude_y(self) -> Optional[float]:
        """(Read-only) y-amplitude of the orbit.
        
        Returns
        -------
        float or None
            The y-amplitude in nondimensional units.
        """
        return self.dynamics.amplitude_y

    @property
    def amplitude_z(self) -> Optional[float]:
        """(Read-only) z-amplitude of the orbit.
        
        Returns
        -------
        float or None
            The z-amplitude in nondimensional units.
        """
        return self.dynamics.amplitude_z

