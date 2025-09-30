"""Collinear libration points L1, L2 and L3 of the circular restricted three body problem (CR3BP).

This module provides concrete implementations of the collinear libration points
in the Circular Restricted Three-Body Problem. These points lie on the line
connecting the two primary bodies and are characterized by saddle-center
stability (one unstable direction, two center directions).

Notes
-----
All positions and distances are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.system.libration.base import LibrationPoint

if TYPE_CHECKING:
    from hiten.system.base import System


class CollinearPoint(LibrationPoint):
    """
    Base class for collinear Libration points (L1, L2, L3).
    
    The collinear points lie on the x-axis connecting the two primary
    bodies. They are characterized by having unstable dynamics with
    saddle-center stability (one unstable direction, two center directions).
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
        
    Attributes
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass,
        dimensionless).
    gamma : float
        Distance ratio from the libration point to the nearest primary
        (dimensionless).
    sign : int
        Sign convention for coordinate transformations (+1 for L3, -1 for L1/L2).
    a : float
        Offset along the x-axis used in frame changes (dimensionless).
    linear_modes : tuple
        (lambda1, omega1, omega2) values for the linearized system.
    """
    def __init__(self, system: "System"):
        if not 0 < system.mu <= 0.5:
            raise ValueError(f"Mass parameter mu must be in range (0, 0.5), got {system.mu}")
        super().__init__(system)

    @property
    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values in nondimensional units.
        """
        return self.dynamics.linear_modes

    @property
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.dynamics.normal_form_transform

    @property
    def linear_data(self):
        return self.dynamics.linear_data

    @property
    def gamma(self) -> float:
        """
        Get the gamma value (distance ratio) for the collinear libration point.
        
        Returns
        -------
        float
            The gamma value (dimensionless), representing the distance ratio
            from the libration point to the nearest primary.
        """
        return self.dynamics.gamma


class L1Point(CollinearPoint):
    """
    L1 Libration point, located between the two primary bodies.
    
    The L1 point is situated between the two primary bodies on the line
    connecting them. It is characterized by saddle-center stability with
    one unstable direction and two center directions.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System"):
        super().__init__(system)

    @property
    def idx(self) -> int:
        """
        Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (1 for L1).
        """
        return 1


class L2Point(CollinearPoint):
    """
    L2 Libration point, located beyond the smaller primary body.
    
    The L2 point is situated beyond the smaller primary body on the line
    connecting the primaries. It is characterized by saddle-center stability
    with one unstable direction and two center directions.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System"):
        super().__init__(system)

    @property
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (2 for L2).
        """
        return 2


class L3Point(CollinearPoint):
    """
    L3 Libration point, located beyond the larger primary body.
    
    The L3 point is situated beyond the larger primary body on the line
    connecting the primaries. It is characterized by saddle-center stability
    with one unstable direction and two center directions.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System"):
        super().__init__(system)

    @property
    def idx(self) -> int:
        """
        Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (3 for L3).
        """
        return 3
