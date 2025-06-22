from typing import TYPE_CHECKING

import numpy as np

from system.libration.base import LibrationPoint
from utils.log_config import logger

if TYPE_CHECKING:
    from system.base import System


class TriangularPoint(LibrationPoint):
    """
    Base class for triangular Libration points (L4, L5).
    
    The triangular points form equilateral triangles with the two primary
    bodies. They are characterized by having center stability (stable)
    for mass ratios μ < Routh's critical mass ratio (~0.0385), 
    and unstable for larger mass ratios.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    ROUTH_CRITICAL_MU = (1.0 - np.sqrt(1.0 - (1.0/27.0))) / 2.0 # approx 0.03852
    
    def __init__(self, system: "System"):
        """Initialize a triangular Libration point."""
        super().__init__(system)
        # Log stability warning based on mu
        if system.mu > self.ROUTH_CRITICAL_MU:
            logger.warning(f"Triangular points are potentially unstable for mu > {self.ROUTH_CRITICAL_MU:.6f} (current mu = {system.mu})")

    @property
    def sign(self) -> int:
        """Sign convention (±1) used for local ↔ synodic transformations.

        Following the convention adopted in Gómez et al. (2001):
        """
        return 1 if isinstance(self, L4Point) else -1
    
    @property
    def a(self) -> float:
        """Offset *a* along the x axis used in frame changes.
        """
        return self.sign * 3 * np.sqrt(3) / 4 * (1 - 2 * self.mu)

    def _find_position(self, y_sign: int) -> np.ndarray:
        """
        Calculate the position of a triangular point (L4 or L5).
        
        Parameters
        ----------
        y_sign : int
            Sign for y-coordinate: +1 for L4, -1 for L5
            
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position
        """
        point_name = self.__class__.__name__
        logger.debug(f"Calculating {point_name} position directly.")
        
        x = 0.5 - self.mu
        y = y_sign * np.sqrt(3) / 2.0
        
        logger.info(f"{point_name} position calculated: x = {x:.6f}, y = {y:.6f}")
        return np.array([x, y, 0], dtype=np.float64)

    def _get_linear_data(self):
        raise NotImplementedError("Not implemented for triangular points.")

    def normal_form_transform(self):
        raise NotImplementedError("Not implemented for triangular points.")


class L4Point(TriangularPoint):
    """
    L4 Libration point, forming an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        """Initialize the L4 Libration point."""
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L4 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L4
        """
        return self._find_position(y_sign=+1)

    @property
    def idx(self) -> int:
        return 4


class L5Point(TriangularPoint):
    """
    L5 Libration point, forming an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        """Initialize the L5 Libration point."""
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L5 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L5
        """
        return self._find_position(y_sign=-1)

    @property
    def idx(self) -> int:
        return 5
