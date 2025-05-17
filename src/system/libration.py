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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

import mpmath as mp
import numpy as np
import symengine as se

from algorithms.dynamics import jacobian_crtbp
# Import existing dynamics functionality
from algorithms.energy import crtbp_energy, energy_to_jacobi
from algorithms.linalg import eigenvalue_decomposition
from algorithms.variables import (get_vars, linear_modes_vars,
                                  scale_factors_vars)
# Import custom logger
from log_config import logger

# Set mpmath precision to 50 digits for root finding
# mp.mp.dps = 50 # Removed global setting

# Constants for stability analysis mode
CONTINUOUS_SYSTEM = 0
DISCRETE_SYSTEM = 1

omega1_sym, omega2_sym, lambda1_sym, c2_sym = get_vars(linear_modes_vars)
s1_sym, s2_sym = get_vars(scale_factors_vars)


@dataclass(slots=True)
class LinearData:
    mu: float
    point: str        # 'L1', 'L2', 'L3'
    lambda1: float
    omega1: float
    omega2: float
    C: np.ndarray     # 6×6 symplectic transform
    Cinv: np.ndarray  # inverse


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
        self._linear_data = None
        self._energy = None
        self._jacobi_constant = None
        
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
    def energy(self) -> float:
        """
        Get the energy of the Libration point.
        """
        if self._energy is None:
            self._energy = self._compute_energy()
        return self._energy
    
    @property
    def jacobi_constant(self) -> float:
        """
        Get the Jacobi constant of the Libration point.
        """
        if self._jacobi_constant is None:
            self._jacobi_constant = self._compute_jacobi_constant()
        return self._jacobi_constant
    
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

    @property
    def linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        """
        if self._linear_data is None:
            self._linear_data = self._get_linear_data()
        return self._linear_data

    def _compute_energy(self) -> float:
        """
        Compute the energy of the Libration point.
        """
        state = np.concatenate([self.position, [0, 0, 0]])
        return crtbp_energy(state, self.mu)

    def _compute_jacobi_constant(self) -> float:
        """
        Compute the Jacobi constant of the Libration point.
        """
        return energy_to_jacobi(self.energy)

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

    @abstractmethod
    def _get_linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        """
        pass

    @abstractmethod
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the normal form transform for the Libration point.
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
        self._cn_cache = {}  # Cache for cn values
        self._linear_modes_cache = None  # Cache for linear modes

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

    def _cn_cached(self, n: int) -> float:
        """
        Get the cached value of cn(mu) or compute it if not available.
        
        Parameters
        ----------
        n : int
            The index for the cn coefficient
            
        Returns
        -------
        float
            The value of cn(mu)
        """
        if n not in self._cn_cache:
            # Compute and cache the value
            self._cn_cache[n] = self._compute_cn(n)
            logger.info(f"c{n}(mu) = {self._cn_cache[n]}")
        else:
            logger.debug(f"Using cached value for c{n}(mu) = {self._cn_cache[n]}")
            
        return self._cn_cache[n]
        
    @abstractmethod
    def _compute_cn(self, n: int) -> float:
        """
        Compute the actual value of cn(mu) without caching.
        This needs to be implemented by subclasses.
        """
        pass

    def _cn(self, n: int) -> float:
        """
        Get the cn coefficient. This is a wrapper that uses caching.
        """
        return self._cn_cached(n)

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
        
        expr = term1 + term2 + term3

        return expr

    def _planar_matrix(self) -> np.ndarray:
        """
        Return the 4x4 matrix M of eq. (9) restricted to (x,y,px,py) coordinates.
        We are not using the full 6x6 matrix since the z direction is decoupled.
        """
        c2 = self._cn(2)
        return np.array([[0, 1, 1, 0],
                        [-1, 0, 0, 1],
                        [2*c2, 0, 0, 1],
                        [0, -c2, -1, 0]], dtype=np.float64)

    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values
        """
        if self._linear_modes_cache is None:
            logger.debug(f"Computing linear modes for {type(self).__name__}")
            self._linear_modes_cache = self._compute_linear_modes()
        else:
            logger.debug(f"Using cached linear modes for {type(self).__name__}")
            
        return self._linear_modes_cache
            
    def _compute_linear_modes(self):
        """
        Compute the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values for the libration point
        """
        try:
            with mp.workdps(50):
                c2 = self._cn(2)
                a = 1.0  # coefficient of x²
                b = 2.0 - c2  # coefficient of x
                c = 1.0 + c2 - 2.0*c2**2  # constant term
                
                # Convert coefficients to mpmath objects
                mp_a = mp.mpf(a)
                mp_b = mp.mpf(b)
                mp_c = mp.mpf(c)
                
                # Use quadratic formula: x = (-b ± sqrt(b² - 4ac))/(2a)
                discriminant = mp_b**2 - 4*mp_a*mp_c
                
                # Find both roots
                eta1 = (-mp_b - mp.sqrt(discriminant)) / (2*mp_a)
                eta2 = (-mp_b + mp.sqrt(discriminant)) / (2*mp_a)
                
                # Convert to floats for further calculations
                eta1_float = float(eta1)
                eta2_float = float(eta2)
                
                # Calculate the required values
                lambda1 = float(mp.sqrt(max(eta1_float, eta2_float))) if max(eta1_float, eta2_float) > 0 else 0.0
                omega1 = float(mp.sqrt(-min(eta1_float, eta2_float))) if min(eta1_float, eta2_float) < 0 else 0.0
                omega2 = float(mp.sqrt(c2))
                
            logger.info(f"Quadratic roots: eta1={eta1_float}, eta2={eta2_float}")
            logger.info(f"Calculated: lambda1={lambda1}, omega1={omega1}, omega2={omega2}")
            return lambda1, omega1, omega2
        except Exception as e:
            # Handle errors more appropriately
            logger.error(f"Failed to calculate linear modes: {e}")
            raise RuntimeError(f"Linear modes calculation failed.") from e

    def _scale_factor(self, lambda1, omega1, omega2):
        """
        Calculate the normalization factors s1 and s2 used in the normal form transformation.
        
        Parameters
        ----------
        lambda1 : float
            The hyperbolic mode value
        omega1 : float
            The elliptic mode value
        omega2 : float
            The vertical oscillation frequency
            
        Returns
        -------
        s1, s2 : tuple of float
            The normalization factors for the hyperbolic and elliptic components
        """
        c2 = self._cn(2)  # Get c2 coefficient
        
        # Calculate the expressions inside the square roots
        expr1 = 2*lambda1*((4+3*c2)*lambda1**2 + 4 + 5*c2 - 6*c2**2)
        expr2 = omega1*((4+3*c2)*omega1**2 - 4 - 5*c2 + 6*c2**2)
        
        # Check if values are positive
        if expr1 < 0:
            err = f"Expression for s1 is negative: {expr1}."
            logger.error(err)
            raise RuntimeError(err)
            
        if expr2 < 0:
            err = f"Expression for s2 is negative: {expr2}."
            logger.error(err)
            raise RuntimeError(err)
        
        # Calculate scale factors
        s1 = np.sqrt(expr1)
        s2 = np.sqrt(expr2)
        
        logger.debug(f"Normalization factors calculated: s1={s1}, s2={s2}")
        return s1, s2

    def _symbolic_normal_form_transform(self) -> Tuple[se.Matrix, se.Matrix]:
        """
        Build the 6x6 symplectic matrix C symbolically as in eq. (10) that sends H_2 to
        lambda_1 x px + (omega_1/2)(y²+p_y²) + (omega_2/2)(z²+p_z²).

        Returns
        -------
        tuple
            (C, Cinv) where C is the symbolic symplectic transformation matrix and Cinv is its inverse
        """
        logger.debug(f"Computing symbolic normal form transform for {type(self).__name__}")

        # Create the symbolic matrix C based on the mathematical expression in the image (eq. 10)
        # Create a zero matrix (symengine doesn't have Matrix.zeros like numpy)
        C = se.Matrix([[0 for _ in range(6)] for _ in range(6)])
        
        # First row
        C[0, 0] = se.Integer(2) * lambda1_sym / s1_sym
        C[0, 1] = se.Integer(0)
        C[0, 2] = se.Integer(0)
        C[0, 3] = -se.Integer(2) * lambda1_sym / s1_sym
        C[0, 4] = se.Integer(2) * omega1_sym / s2_sym
        C[0, 5] = se.Integer(0)
        
        # Second row
        C[1, 0] = (lambda1_sym**se.Integer(2) - se.Integer(2)*c2_sym - se.Integer(1)) / s1_sym
        C[1, 1] = (-omega1_sym**se.Integer(2) - se.Integer(2)*c2_sym - se.Integer(1)) / s2_sym
        C[1, 2] = se.Integer(0)
        C[1, 3] = (lambda1_sym**se.Integer(2) - se.Integer(2)*c2_sym - se.Integer(1)) / s1_sym
        C[1, 4] = se.Integer(0)
        C[1, 5] = se.Integer(0)
        
        # Third row
        C[2, 0] = se.Integer(0)
        C[2, 1] = se.Integer(0)
        C[2, 2] = se.Integer(1) / se.sqrt(omega2_sym)
        C[2, 3] = se.Integer(0)
        C[2, 4] = se.Integer(0)
        C[2, 5] = se.Integer(0)
        
        # Fourth row
        C[3, 0] = (lambda1_sym**se.Integer(2) + se.Integer(2)*c2_sym + se.Integer(1)) / s1_sym
        C[3, 1] = (-omega1_sym**se.Integer(2) + se.Integer(2)*c2_sym + se.Integer(1)) / s2_sym
        C[3, 2] = se.Integer(0)
        C[3, 3] = (lambda1_sym**se.Integer(2) + se.Integer(2)*c2_sym + se.Integer(1)) / s1_sym
        C[3, 4] = se.Integer(0)
        C[3, 5] = se.Integer(0)
        
        # Fifth row
        C[4, 0] = (lambda1_sym**se.Integer(3) + (se.Integer(1) - se.Integer(2)*c2_sym)*lambda1_sym) / s1_sym
        C[4, 1] = se.Integer(0)
        C[4, 2] = se.Integer(0)
        C[4, 3] = (-lambda1_sym**se.Integer(3) - (se.Integer(1) - se.Integer(2)*c2_sym)*lambda1_sym) / s1_sym
        C[4, 4] = (-omega1_sym**se.Integer(3) + (se.Integer(1) - se.Integer(2)*c2_sym)*omega1_sym) / s2_sym
        C[4, 5] = se.Integer(0)
        
        # Sixth row
        C[5, 0] = se.Integer(0)
        C[5, 1] = se.Integer(0)
        C[5, 2] = se.Integer(0)
        C[5, 3] = se.Integer(0)
        C[5, 4] = se.Integer(0)
        C[5, 5] = se.sqrt(omega2_sym)
        
        # Compute the symbolic inverse
        Cinv = C.inv()
        
        logger.debug(f"Symbolic normal form transformation matrix computed")
        return C, Cinv

    def normal_form_transform(self):
        """
        Build the 6x6 symplectic matrix C of eq. (10) that sends H_2 to
        lambda_1 x px + (omega_1/2)(y²+p_y²) + (omega_2/2)(z²+p_z²).

        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix and Cinv is its inverse
        """
        logger.debug(f"Computing normal form transform for {type(self).__name__} with mu={self.mu}")
        
        # Get the symbolic form of the matrix
        C_sym, Cinv_sym = self._symbolic_normal_form_transform()
        
        # Use the cached linear modes
        lambda1_num, omega1_num, omega2_num = self.linear_modes()
        logger.debug(f"Using linear modes: lambda1={lambda1_num}, omega1={omega1_num}, omega2={omega2_num}")
        
        c2 = self._cn(2)
        logger.debug(f"Using c2 coefficient: c2={c2}")

        # Get normalization factors s1, s2
        s1, s2 = self._scale_factor(lambda1_num, omega1_num, omega2_num)
        logger.debug(f"Scale factors: s1={s1}, s2={s2}")

        # Substitute the symbolic variables with their numerical values
        subs_dict = {
            lambda1_sym: float(lambda1_num),
            omega1_sym: float(omega1_num),
            omega2_sym: float(omega2_num),
            s1_sym: float(s1),
            s2_sym: float(s2),
            c2_sym: float(c2)
        }
        
        # Convert to numerical matrices
        C = np.array(C_sym.subs(subs_dict).tolist(), dtype=np.float64)
        Cinv = np.array(Cinv_sym.subs(subs_dict).tolist(), dtype=np.float64)
        
        logger.info(f"Normal form transformation matrix computed for {type(self).__name__}")

        return C, Cinv

    def _get_linear_data(self):
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        LinearData
            Object containing the linear data for the Libration point
        """
        logger.debug(f"Getting linear data for {type(self).__name__}")
        
        # Use cached linear modes
        lambda1, omega1, omega2 = self.linear_modes()
        c2 = self._cn(2)
        
        # Get normalization factors s1, s2
        s1, s2 = self._scale_factor(lambda1, omega1, omega2)
        
        # Get symbolic transformation matrices
        C_sym, Cinv_sym = self._symbolic_normal_form_transform()
        
        # Substitute the symbolic variables with their numerical values
        subs_dict = {
            lambda1_sym: float(lambda1),
            omega1_sym: float(omega1),
            omega2_sym: float(omega2),
            s1_sym: float(s1),
            s2_sym: float(s2),
            c2_sym: float(c2)
        }
        
        # Convert to numerical matrices
        C = np.array(C_sym.subs(subs_dict).tolist(), dtype=np.float64)
        Cinv = np.array(Cinv_sym.subs(subs_dict).tolist(), dtype=np.float64)
        
        # Create and return the LinearData object
        return LinearData(
            mu=self.mu,
            point=type(self).__name__[:2],  # 'L1', 'L2', 'L3'
            lambda1=lambda1, 
            omega1=omega1, 
            omega2=omega2,
            C=C, 
            Cinv=Cinv
        )

    def get_symbolic_transform(self) -> Tuple[se.Matrix, se.Matrix, dict]:
        """
        Get the symbolic normal form transformation matrices and the parameters.
        
        Returns
        -------
        tuple
            (C_sym, Cinv_sym, params) where:
            - C_sym is the symbolic transformation matrix
            - Cinv_sym is its symbolic inverse
            - params is a dictionary mapping symbolic parameters to their names
        """
        # Get the symbolic matrices
        C_sym, Cinv_sym = self._symbolic_normal_form_transform()
        
        # Create a dictionary to map symbolic variables to their descriptions
        params = {
            lambda1_sym: 'Hyperbolic eigenvalue',
            omega1_sym: 'Planar oscillation frequency',
            omega2_sym: 'Vertical oscillation frequency',
            s1_sym: 'Hyperbolic normalization factor',
            s2_sym: 'Elliptic normalization factor',
            c2_sym: 'Second-order coefficient in the potential'
        }
        
        return C_sym, Cinv_sym, params

    def get_normal_form_parameters(self) -> dict:
        """
        Get the numerical values of the parameters used in the normal form transformation.
        
        Returns
        -------
        dict
            A dictionary mapping parameter names to their numerical values
        """
        # Calculate the values if not already cached
        lambda1, omega1, omega2 = self.linear_modes()
        c2 = self._cn(2)
        s1, s2 = self._scale_factor(lambda1, omega1, omega2)
        
        # Create a dictionary with parameter names and values
        params = {
            'lambda1': lambda1,
            'omega1': omega1,
            'omega2': omega2,
            's1': s1,
            's2': s2,
            'c2': c2
        }
        
        return params

    def substitute_parameters(self, expression: se.Basic) -> se.Basic:
        """
        Substitute the numerical parameter values into a symbolic expression.
        
        Parameters
        ----------
        expression : se.Basic
            The symbolic expression with parameters like lambda1, omega1, etc.
            
        Returns
        -------
        se.Basic
            The expression with numerical values substituted
        """
        # Get parameter values
        params = self.get_normal_form_parameters()
        
        # Create substitution dictionary for symengine
        subs_dict = {
            lambda1_sym: float(params['lambda1']),
            omega1_sym: float(params['omega1']),
            omega2_sym: float(params['omega2']),
            s1_sym: float(params['s1']),
            s2_sym: float(params['s2']),
            c2_sym: float(params['c2'])
        }
        
        # Apply substitution
        return expression.subs(subs_dict)


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
        coeffs = self._get_gamma_poly_coeffs()
        term1 = x**5
        term2 = coeffs[1] * x**4
        term3 = coeffs[2] * x**3
        term4 = coeffs[3] * x**2
        term5 = coeffs[4] * x
        term6 = coeffs[5]
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

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn(mu) as in Jorba & Masdemont (1999), eq. (3) using self.gamma for L1.
        
        Parameters
        ----------
        n : int
            Index of the coefficient to compute
            
        Returns
        -------
        float
            The cn coefficient value
        """
        term1 = 1 / self.gamma ** 3
        term2 = (1) ** n * self.mu
        term3 = (-1) ** n * ( (1-self.mu) * self.gamma ** (n+1) / (1 - self.gamma) ** (n+1) )

        c_n = term1 * (term2 + term3)
        return c_n


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
        coeffs = self._get_gamma_poly_coeffs()
        term1 = x**5
        term2 = coeffs[1] * x**4
        term3 = coeffs[2] * x**3
        term4 = coeffs[3] * x**2
        term5 = coeffs[4] * x
        term6 = coeffs[5]
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

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn(mu) as in Jorba & Masdemont (1999), eq. (3) using self.gamma for L2.
        
        Parameters
        ----------
        n : int
            Index of the coefficient to compute
            
        Returns
        -------
        float
            The cn coefficient value
        """
        term1 = 1 / self.gamma ** 3
        term2 = (-1) ** n * self.mu
        term3 = (-1) ** n * ( (1-self.mu) * self.gamma ** (n+1) / (1 + self.gamma) ** (n+1) )

        c_n = term1 * (term2 + term3)
        return c_n


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
        coeffs = self._get_gamma_poly_coeffs()
        term1 = x**5
        term2 = coeffs[1] * x**4
        term3 = coeffs[2] * x**3
        term4 = coeffs[3] * x**2
        term5 = coeffs[4] * x
        term6 = coeffs[5]
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

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn(mu) as in Jorba & Masdemont (1999), eq. (3) using self.gamma for L3.
        
        Parameters
        ----------
        n : int
            Index of the coefficient to compute
            
        Returns
        -------
        float
            The cn coefficient value
        """
        term1 = (-1) ** n / self.gamma ** 3
        term2 = 1-self.mu
        term3 = self.mu * self.gamma ** (n+1) / (1 + self.gamma) ** (n+1)

        c_n = term1 * (term2 + term3)
        return c_n


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
