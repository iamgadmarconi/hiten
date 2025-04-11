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
from typing import Tuple, Union

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
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

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
        # Analyze stability if not already done
        if self._stability_info is None:
            self.analyze_stability() 
        
        # Access stability indices (nu values)
        indices = self._stability_info[0] 
        
        # An orbit is stable if all stability indices have magnitude <= 1
        # Use a small tolerance for floating point comparisons
        return np.all(np.abs(indices) <= 1.0 + 1e-9)

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
        # Only recalculate if stability info is not cached OR if parameters change
        # Simple approach: always recalculate if called explicitly
        # A more complex cache could check if discrete/delta match cached values
        # For now, let's keep it simple: explicit call recalculates.
        mode_str = "Continuous" if discrete == CONTINUOUS_SYSTEM else "Discrete"
        logger.info(f"Analyzing stability for {type(self).__name__} (mu={self.mu}), mode={mode_str}, delta={delta}.")
        # Compute the system Jacobian at the Libration point
        pos = self.position # Ensures position is calculated first
        A = jacobian_crtbp(pos[0], pos[1], pos[2], self.mu)
        
        logger.debug(f"Jacobian calculated at position {pos}:\n{A}")

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
        if self._stability_info is None:
             self.analyze_stability() # Ensure stability is analyzed
        sn, un, cn, _, _, _ = self._stability_info
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
        if self._stability_info is None:
             self.analyze_stability() # Ensure stability is analyzed
        _, _, _, Ws, Wu, Wc = self._stability_info
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
    """
    def __init__(self, mu: float):
        """Initialize a collinear Libration point."""
        super().__init__(mu)
        self._gamma = None # Cache for gamma value

    @property
    def gamma(self, precision: int = 50) -> float:
        """
        Get the distance ratio gamma for the libration point, calculated
        with high precision.

        Gamma is defined as the distance from the libration point to the nearest primary,
        normalized by the distance between the primaries.
        - For L1 and L2, gamma = |x_L - (1-mu)|
        - For L3, gamma = |x_L - (-mu)| 
        (Note: This is equivalent to the root of the specific polynomial for each point).

        Parameters
        ----------
        precision : int, optional
            Number of decimal places for high precision calculation. Default is 50.

        Returns
        -------
        float
            The gamma value calculated with high precision.
        """
        if self._gamma is None:
            logger.debug(f"Calculating gamma for {type(self).__name__} (mu={self.mu}) with {precision} dps.")
            
            # Step 1: Get initial approximation using np.roots()
            poly_coeffs = self._get_gamma_poly_coeffs()
            roots = np.roots(poly_coeffs)
            
            # Find the physically relevant real root for initial guess
            x0 = self._find_relevant_real_root(roots)
            
            if x0 is None:
                 logger.warning(f"np.roots failed to find a suitable real root for {type(self).__name__}. Falling back to rough estimate.")
                 x0 = self._get_fallback_gamma_estimate()
            
            logger.debug(f"Initial estimate for {type(self).__name__} gamma: x0 = {x0}")

            # Step 2: Refine using high precision mp.findroot()
            with mp.workdps(precision):
                # The polynomial function is defined by the subclass
                poly_func = lambda x_val: self._gamma_poly(x_val)
                gamma_val = mp.findroot(poly_func, x0)
                self._gamma = float(gamma_val)

            logger.info(f"Gamma for {type(self).__name__} calculated: gamma = {self._gamma}")
            
        return self._gamma

    @abstractmethod
    def _get_gamma_poly_coeffs(self) -> list[float]:
        """Return the coefficients of the polynomial whose root is gamma."""
        pass
        
    @abstractmethod
    def _gamma_poly(self, x: float) -> float:
         """Evaluate the polynomial whose root is gamma at point x."""
         pass
         
    @abstractmethod
    def _find_relevant_real_root(self, roots: np.ndarray) -> float | None:
        """From the roots of the polynomial, find the one relevant to this point."""
        pass
        
    @abstractmethod
    def _get_fallback_gamma_estimate(self) -> float:
        """Provide a rough estimate for gamma if np.roots fails."""
        pass
        
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
        # Handle potential division by zero if x coincides with primary positions
        # Although for L1/L2/L3 this shouldn't happen
        r1_sq = (x + mu)**2
        r2_sq = (x - (1 - mu))**2
        
        # Avoid division by zero, though unlikely for L-points
        r1_3 = r1_sq**1.5 if r1_sq > 1e-16 else 0
        r2_3 = r2_sq**1.5 if r2_sq > 1e-16 else 0

        term1 = x
        term2 = -(1 - mu) * (x + mu) / r1_3 if r1_3 > 0 else 0
        term3 = -mu * (x - (1 - mu)) / r2_3 if r2_3 > 0 else 0
        
        return term1 + term2 + term3


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
        Calculate the position of the L1 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L1
        """
        interval = [-self.mu + 0.01, 1 - self.mu - 0.01]
        logger.debug(f"L1: Finding root of dOmega/dx in interval {interval}")
        
        # Use high precision root finding
        try:
             with mp.workdps(50):
                 # Use lambda to pass self implicitly
                 func = lambda x_val: self._dOmega_dx(x_val)
                 # Provide the interval as a bracket if possible
                 x = mp.findroot(func, interval)
             x = float(x)
             logger.info(f"L1 position calculated: x = {x}")
             return np.array([x, 0, 0], dtype=np.float64)
        except ValueError as e:
             # Handle cases where findroot fails (e.g., no sign change in interval)
             logger.error(f"Failed to find L1 root in interval {interval}: {e}")
             # Optionally, could try a different solver or wider interval
             # For now, re-raise or return NaN/error indicator
             raise RuntimeError(f"L1 position calculation failed.") from e
             
    def _get_gamma_poly_coeffs(self) -> list[float]:
        mu = self.mu
        return [1, -(3-mu), (3-2*mu), -mu, 2*mu, -mu]
        
    def _gamma_poly(self, x: float) -> float:
         mu = self.mu
         term1 = x**5
         term2 = -(3-mu) * x**4
         term3 = (3-2*mu) * x**3
         term4 = -mu*x**2
         term5 = 2*mu*x
         term6 = -mu
         return term1 + term2 + term3 + term4 + term5 + term6
         
    def _find_relevant_real_root(self, roots: np.ndarray) -> float | None:
        # For L1, gamma should be positive and small (distance from m2)
        # Position x is 1 - mu - gamma. Gamma = 1 - mu - x
        # Since -mu < x < 1-mu, we expect 0 < gamma < 1.
        # The polynomial root *is* gamma directly.
        mu = self.mu
        for r in roots:
            if np.isreal(r):
                real_r = float(r.real)
                # Gamma for L1 should be positive and typically less than 1
                if 0 < real_r < 1.0:
                     # Further check: is it physically plausible?
                     # L1 position x = 1 - mu - real_r. Check if -mu < x < 1-mu
                     x_pos = 1 - mu - real_r
                     if -mu < x_pos < (1-mu):
                         return real_r
        return None
        
    def _get_fallback_gamma_estimate(self) -> float:
        # Rough estimate for gamma_L1 (distance from m2)
        return (self.mu / 3)**(1/3)


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
        Calculate the position of the L2 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L2
        """
        interval = [1.0, 2.0] # Initial guess interval for L2
        logger.debug(f"L2: Finding root of dOmega/dx in interval {interval}")
        try:
             with mp.workdps(50):
                 func = lambda x_val: self._dOmega_dx(x_val)
                 x = mp.findroot(func, interval)
             x = float(x)
             logger.info(f"L2 position calculated: x = {x}")
             return np.array([x, 0, 0], dtype=np.float64)
        except ValueError as e:
             logger.error(f"Failed to find L2 root in interval {interval}: {e}")
             # Try a wider interval as fallback?
             logger.debug(f"L2: Retrying root finding in wider interval {[1 - self.mu + 1e-9, 2.0]}")
             try:
                 with mp.workdps(50):
                    func = lambda x_val: self._dOmega_dx(x_val)
                    x = mp.findroot(func, [1 - self.mu + 1e-9, 2.0])
                 x = float(x)
                 logger.info(f"L2 position calculated (retry): x = {x}")
                 return np.array([x, 0, 0], dtype=np.float64)
             except ValueError as e2:
                logger.error(f"Failed to find L2 root even in wider interval: {e2}")
                raise RuntimeError(f"L2 position calculation failed.") from e2
             
    def _get_gamma_poly_coeffs(self) -> list[float]:
        mu = self.mu
        return [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]

    def _gamma_poly(self, x: float) -> float:
         mu = self.mu
         term1 = x**5
         term2 = (3-mu) * x**4
         term3 = (3-2*mu) * x**3
         term4 = -mu*x**2
         term5 = -2*mu*x
         term6 = -mu
         return term1 + term2 + term3 + term4 + term5 + term6
         
    def _find_relevant_real_root(self, roots: np.ndarray) -> float | None:
        # For L2, gamma should be positive and small (distance from m2)
        # Position x = 1 - mu + gamma. Gamma = x - (1 - mu)
        # Since x > 1-mu, we expect gamma > 0.
        # The polynomial root *is* gamma directly.
        mu = self.mu
        for r in roots:
            if np.isreal(r):
                real_r = float(r.real)
                # Gamma for L2 should be positive and typically less than 1
                if 0 < real_r < 1.0:
                     # Further check: is it physically plausible?
                     # L2 position x = 1 - mu + real_r. Check if x > 1-mu
                     x_pos = 1 - mu + real_r
                     if x_pos > (1-mu):
                         return real_r
        return None
        
    def _get_fallback_gamma_estimate(self) -> float:
        # Rough estimate for gamma_L2 (distance from m2)
        return (self.mu / 3)**(1/3)


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
        Calculate the position of the L3 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L3
        """
        interval = [-self.mu - 0.01, -2.0] # Initial guess interval for L3
        logger.debug(f"L3: Finding root of dOmega/dx in interval {interval}")
        try:
             with mp.workdps(50):
                 func = lambda x_val: self._dOmega_dx(x_val)
                 x = mp.findroot(func, interval)
             x = float(x)
             logger.info(f"L3 position calculated: x = {x}")
             return np.array([x, 0, 0], dtype=np.float64)
        except ValueError as e:
             logger.error(f"Failed to find L3 root in interval {interval}: {e}")
             # Try a wider interval as fallback?
             logger.debug(f"L3: Retrying root finding in wider interval {[-2.0, -self.mu - 1e-9]}")
             try:
                 with mp.workdps(50):
                    func = lambda x_val: self._dOmega_dx(x_val)
                    x = mp.findroot(func, [-2.0, -self.mu - 1e-9])
                 x = float(x)
                 logger.info(f"L3 position calculated (retry): x = {x}")
                 return np.array([x, 0, 0], dtype=np.float64)
             except ValueError as e2:
                logger.error(f"Failed to find L3 root even in wider interval: {e2}")
                raise RuntimeError(f"L3 position calculation failed.") from e2
             
    def _get_gamma_poly_coeffs(self) -> list[float]:
        mu = self.mu
        mu2 = 1 - mu # mu1 in some notations
        return [1, (2+mu), (1+2*mu), -mu2, -2*mu2, -mu2]
        
    def _gamma_poly(self, x: float) -> float:
         # Note: The root x of this polynomial is gamma_L3 = |x_L3 - (-mu)| = |-1 + delta - (-mu)| = |mu - 1 + delta|
         # where x_L3 = -1 + delta. This x IS the distance gamma_L3.
         mu = self.mu
         mu2 = 1 - mu
         term1 = x**5
         term2 = (2+mu) * x**4
         term3 = (1+2*mu) * x**3
         term4 = -mu2 * x**2
         term5 = -2*mu2*x
         term6 = -mu2
         return term1 + term2 + term3 + term4 + term5 + term6
         
    def _find_relevant_real_root(self, roots: np.ndarray) -> float | None:
        # For L3, gamma is the distance from m1: gamma = |x_L3 - (-mu)|.
        # Since x_L3 is approx -1, gamma_L3 is approx |-1 - (-mu)| = |mu-1| which is approx 1.
        # The polynomial root *is* gamma directly.
        mu = self.mu
        for r in roots:
            if np.isreal(r):
                real_r = float(r.real)
                # Gamma for L3 should be positive and close to 1
                if 0.5 < real_r < 1.5: # Fairly wide check around 1
                     # Further check: is it physically plausible?
                     # L3 position x = -mu - real_r. Check if x < -mu
                     x_pos = -mu - real_r
                     if x_pos < -mu:
                         # Need to be careful: L3 poly root is gamma, distance from m1
                         return real_r 
        return None

    def _get_fallback_gamma_estimate(self) -> float:
        # Rough estimate for gamma_L3 (distance from m1)
        # x_L3 approx -(1 - 7/12*mu). gamma = |x_L3 - (-mu)| = |-1 + 7/12*mu + mu| = |mu*19/12 - 1|
        # A simpler estimate is often just 1.
        # Or using the relation from Szebehely, gamma_L3 approx 1 - (7/12)mu
        return 1.0 - (7.0 / 12.0) * self.mu


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
    
    def __init__(self, mu: float):
        """Initialize a triangular Libration point."""
        super().__init__(mu)
        # Log stability warning based on mu
        if mu > self.ROUTH_CRITICAL_MU:
             logger.warning(f"Triangular points are potentially unstable for mu > {self.ROUTH_CRITICAL_MU:.6f} (current mu = {mu})")


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
        x = 0.5 - self.mu
        y = np.sqrt(3) / 2.0
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
        x = 0.5 - self.mu
        y = -np.sqrt(3) / 2.0
        logger.info(f"L5 position calculated: x = {x}, y = {y}")
        return np.array([x, y, 0], dtype=np.float64)
