"""
hiten.system.libration.triangular
==========================

Triangular Libration points (L4 and L5) of the Circular Restricted Three-Body Problem (CR3BP).

This module provides concrete implementations of the triangular libration points
in the Circular Restricted Three-Body Problem. These points form equilateral
triangles with the two primary bodies and are characterized by center-type
stability when the mass ratio is below Routh's critical value.

Classes
-------
:class:`~hiten.system.libration.triangular.TriangularPoint`
    Abstract base class for triangular libration points.
:class:`~hiten.system.libration.triangular.L4Point`
    L4 libration point located above the x-axis (positive y).
:class:`~hiten.system.libration.triangular.L5Point`
    L5 libration point located below the x-axis (negative y).

Notes
-----
All positions and distances are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.system.libration.base import LibrationPoint
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.base import System


class TriangularPoint(LibrationPoint):
    """
    Abstract helper for the triangular Libration points.

    The triangular points form equilateral triangles with the two primary
    bodies. They behave as centre-type equilibria when the mass ratio
    mu is below Routh's critical value.

    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        CR3BP model supplying the mass parameter mu.

    Attributes
    ----------
    mu : float
        Mass ratio mu = m2 / (m1 + m2) taken from system (dimensionless).
    ROUTH_CRITICAL_MU : float
        Critical value mu_R delimiting linear stability (dimensionless).
    sign : int
        +1 for :class:`~hiten.system.libration.triangular.L4Point`, -1 
        for :class:`~hiten.system.libration.triangular.L5Point`.
    a : float
        Offset used by local <-> synodic frame transformations (dimensionless).

    Notes
    -----
    A warning is logged if mu > mu_R.
    """
    ROUTH_CRITICAL_MU = (1.0 - np.sqrt(1.0 - (1.0/27.0))) / 2.0 # approx 0.03852
    
    def __init__(self, system: "System", *, services=None):
        super().__init__(system, services=services)
        # Log stability warning based on mu
        if system.mu > self.ROUTH_CRITICAL_MU:
            logger.warning(f"Triangular points are potentially unstable for mu > {self.ROUTH_CRITICAL_MU:.6f} (current mu = {system.mu})")

    @property
    def sign(self) -> int:
        """
        Sign convention distinguishing L4 and L5.

        Returns
        -------
        int
            +1 for :class:`~hiten.system.libration.triangular.L4Point`, -1 for :class:`~hiten.system.libration.triangular.L5Point`.
        """
        return 1 if isinstance(self, L4Point) else -1
    
    @property
    def a(self) -> float:
        """
        Offset a along the x axis used in frame changes.
        
        Returns
        -------
        float
            The offset value a (dimensionless).
        """
        return self.sign * 3 * np.sqrt(3) / 4 * (1 - 2 * self.mu)

    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of a triangular point (L4 or L5).
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            3D vector [x, y, 0] giving the position in nondimensional units.
        """
        return self._services.dynamics.position()

    def _get_linear_data(self):
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        tuple
            (None, omega1, omega2, omega_z, C, Cinv)
            Object containing the linear data for the Libration point.
        """
        return self._services.dynamics.linear_data()

    def normal_form_transform(self):
        """
        Build the 6x6 symplectic matrix C that sends H2 to normal form.

        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix and Cinv is its inverse.
        """
        return self._services.dynamics.triangular_normal_form()

    

    @property
    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (omega_1, omega_2, omega_z) where:
            - omega_1 > 0 with omega_1^2 < 1/2 (small positive planar frequency)
            - omega_2 < 0 (negative planar frequency)  
            - omega_z = 1.0 (vertical frequency)
            For triangular points all eigenvalues are purely imaginary so no
            hyperbolic mode is present.
        """
        return self._services.dynamics.triangular_linear_modes()

class L4Point(TriangularPoint):
    """
    L4 Libration point, forming an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    The L4 point is situated above the x-axis, forming an equilateral triangle
    with the two primary bodies. It is characterized by center-type stability
    when the mass ratio is below Routh's critical value.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System", *, services=None):
        super().__init__(system, services=services)
    
    @property
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (4 for L4).
        """
        return 4


class L5Point(TriangularPoint):
    """
    L5 Libration point, forming an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    The L5 point is situated below the x-axis, forming an equilateral triangle
    with the two primary bodies. It is characterized by center-type stability
    when the mass ratio is below Routh's critical value.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System", *, services=None):
        super().__init__(system, services=services)
    
    @property
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (5 for L5).
        """
        return 5
