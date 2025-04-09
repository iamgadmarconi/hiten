"""
Libration point model for the CR3BP.

This module defines a hierarchy of classes representing Libration (libration) points
in the Circular Restricted Three-Body Problem (CR3BP). The implementation provides
a clean object-oriented interface to the dynamics and stability properties of
Libration points, with specialized handling for collinear points (L1, L2, L3) and
triangular points (L4, L5).

The class hierarchy consists of:
- LibrationPoint (abstract base class)
- CollinearPoint (for L1, L2, L3)
- TriangularPoint (for L4, L5)
- Concrete classes for each point (L1Point, L2Point, etc.)

Each class provides methods for computing position, stability analysis, and
eigenvalue decomposition appropriate to the specific dynamics of that point type.
"""

import numpy as np
import mpmath as mp
from abc import ABC, abstractmethod
from typing import Tuple

# Import existing dynamics functionality
from algorithms.dynamics import jacobian_crtbp
from algorithms.linalg import eigenvalue_decomposition

# Import custom logger
from log_config import logger

# Set mpmath precision to 50 digits for root finding
# mp.mp.dps = 50 # Removed global setting

# Constants for stability analysis mode
CONTINUOUS_SYSTEM = 0
DISCRETE_SYSTEM = 1


class LibrationPoint(ABC):
    """
    Abstract base class for Libration points in the CR3BP.
    
    This class provides the common interface and functionality for all 
    Libration points. Specific point types (collinear, triangular) will
    extend this class with specialized implementations.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize a Libration point with the mass parameter and point index."""
        self.mu = mu
        self._position = None
        self._stability_info = None
        
        # Log initialization - using type(self).__name__ to get the specific subclass name
        logger.debug(f"Initialized {type(self).__name__} with mu = {self.mu}")
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu})"

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the Libration point in the rotating frame.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        if self._position is None:
            logger.debug(f"Calculating position for {type(self).__name__} (mu={self.mu}).")
            self._position = self._calculate_position()
        return self._position
    
    @property
    def is_stable(self) -> bool:
        """
        Check if the Libration point is stable.
        """
        indices = self._stability_info[0]  # nu values from stability_indices
        
        # An orbit is stable if all stability indices have magnitude <= 1
        return np.all(np.abs(indices) <= 1.0)

    @property
    def is_unstable(self) -> bool:
        """
        Check if the Libration point is unstable.
        """
        return not self.is_stable

    def analyze_stability(self, discrete: int = CONTINUOUS_SYSTEM, delta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze the stability properties of the Libration point.
        
        Parameters
        ----------
        discrete : int, optional
            Classification mode for eigenvalues:
            * CONTINUOUS_SYSTEM (0): continuous-time system (classify by real part sign)
            * DISCRETE_SYSTEM (1): discrete-time system (classify by magnitude relative to 1)
        delta : float, optional
            Tolerance for classification
            
        Returns
        -------
        tuple
            (sn, un, cn, Ws, Wu, Wc) containing:
            - sn: stable eigenvalues
            - un: unstable eigenvalues
            - cn: center eigenvalues
            - Ws: eigenvectors spanning stable subspace
            - Wu: eigenvectors spanning unstable subspace
            - Wc: eigenvectors spanning center subspace
        """
        if self._stability_info is None:
            mode_str = "Continuous" if discrete == CONTINUOUS_SYSTEM else "Discrete"
            logger.info(f"Analyzing stability for {type(self).__name__} (mu={self.mu}), mode={mode_str}, delta={delta}.")
            # Compute the system Jacobian at the Libration point
            pos = self.position
            A = jacobian_crtbp(pos[0], pos[1], pos[2], self.mu)
            
            logger.debug(f"Jacobian calculated at position {pos}:\\n{A}")

            # Perform eigenvalue decomposition and classification
            self._stability_info = eigenvalue_decomposition(A, discrete, delta)
            
            sn, un, cn, _, _, _ = self._stability_info
            logger.info(f"Stability analysis complete: {len(sn)} stable, {len(un)} unstable, {len(cn)} center eigenvalues.")
        
        return self._stability_info
    
    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvalues of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvalues, unstable_eigenvalues, center_eigenvalues)
        """
        sn, un, cn, _, _, _ = self.analyze_stability()
        return (sn, un, cn)
    
    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvectors of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvectors, unstable_eigenvectors, center_eigenvectors)
        """
        _, _, _, Ws, Wu, Wc = self.analyze_stability()
        return (Ws, Wu, Wc)
    
    @abstractmethod
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the Libration point.
        
        This is an abstract method that must be implemented by subclasses.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        pass


class CollinearPoint(LibrationPoint):
    """
    Base class for collinear Libration points (L1, L2, L3).
    
    The collinear points lie on the x-axis connecting the two primary
    bodies. They are characterized by having unstable dynamics with
    saddle-center stability (one unstable direction, two center directions).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    point_index : int
        The Libration point index (must be 1, 2, or 3)
    """
    
    def __init__(self, mu: float):
        """Initialize a collinear Libration point."""
        super().__init__(mu)
    
    def _dOmega_dx(self, x: float) -> float:
        """
        Compute the derivative of the effective potential with respect to x.
        
        Parameters
        ----------
        x : float
            x-coordinate in the rotating frame
        
        Returns
        -------
        float
            Value of dΩ/dx at the given x-coordinate
        """
        mu = self.mu
        r1 = abs(x + mu)
        r2 = abs(x - (1 - mu))
        return x - (1 - mu) * (x + mu) / (r1**3) - mu * (x - (1 - mu)) / (r2**3)


class L1Point(CollinearPoint):
    """
    L1 Libration point, located between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize the L1 Libration point."""
        super().__init__(mu)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L1 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L1
        """
        interval = [-self.mu + 0.01, 1 - self.mu - 0.01]
        logger.debug(f"L1: Finding root of dOmega/dx in interval {interval}")
        with mp.workdps(50):
            func = lambda x_val: self._dOmega_dx(x_val)
            x = mp.findroot(func, interval)
        x = float(x)
        logger.info(f"L1 position calculated: x = {x}")
        return np.array([x, 0, 0], dtype=np.float64)


class L2Point(CollinearPoint):
    """
    L2 Libration point, located beyond the smaller primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize the L2 Libration point."""
        super().__init__(mu)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L2 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L2
        """
        interval = [1.0, 2.0] # Initial guess interval for L2
        logger.debug(f"L2: Finding root of dOmega/dx in interval {interval}")
        with mp.workdps(50):
            func = lambda x_val: self._dOmega_dx(x_val)
            x = mp.findroot(func, interval)
        x = float(x)
        logger.info(f"L2 position calculated: x = {x}")
        return np.array([x, 0, 0], dtype=np.float64)


class L3Point(CollinearPoint):
    """
    L3 Libration point, located beyond the larger primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize the L3 Libration point."""
        super().__init__(mu)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L3 point.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L3
        """
        interval = [-self.mu - 0.01, -2.0] # Initial guess interval for L3
        logger.debug(f"L3: Finding root of dOmega/dx in interval {interval}")
        with mp.workdps(50):
            func = lambda x_val: self._dOmega_dx(x_val)
            x = mp.findroot(func, interval)
        x = float(x)
        logger.info(f"L3 position calculated: x = {x}")
        return np.array([x, 0, 0], dtype=np.float64)


class TriangularPoint(LibrationPoint):
    """
    Base class for triangular Libration points (L4, L5).
    
    The triangular points form equilateral triangles with the two primary
    bodies. They are characterized by having center stability (stable)
    for mass ratios μ < 0.0385, and unstable for larger mass ratios.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize a triangular Libration point."""
        super().__init__(mu)
        
        # Check stability based on mass ratio
        if mu > 0.0385:
            logger.warning(f"Triangular points are unstable for mu > 0.0385 (current mu = {mu})")


class L4Point(TriangularPoint):
    """
    L4 Libration point, forming an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize the L4 Libration point."""
        super().__init__(mu)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L4 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L4
        """
        logger.debug(f"Calculating L4 position directly.")
        x = 1 / 2 - self.mu
        y = np.sqrt(3) / 2
        logger.info(f"L4 position calculated: x = {x}, y = {y}")
        return np.array([x, y, 0], dtype=np.float64)


class L5Point(TriangularPoint):
    """
    L5 Libration point, forming an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, mu: float):
        """Initialize the L5 Libration point."""
        super().__init__(mu)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L5 point.
        
        Returns
        -------
        ndarray
            3D vector [x, y, 0] giving the position of L5
        """
        logger.debug(f"Calculating L5 position directly.")
        x = 1 / 2 - self.mu
        y = -np.sqrt(3) / 2
        logger.info(f"L5 position calculated: x = {x}, y = {y}")
        return np.array([x, y, 0], dtype=np.float64)
