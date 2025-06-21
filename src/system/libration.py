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
from typing import TYPE_CHECKING, Tuple

import numpy as np

from algorithms.dynamics.rtbp import _jacobian_crtbp, create_var_eq_system
from algorithms.energy import crtbp_energy, energy_to_jacobi
from algorithms.linalg import eigenvalue_decomposition
from config import MPMATH_DPS
from utils.log_config import logger
from utils.precision import find_root, hp

if TYPE_CHECKING:
    from system.base import System

# Constants for stability analysis mode
CONTINUOUS_SYSTEM = 0
DISCRETE_SYSTEM = 1


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
    
    def __init__(self, system: "System"):
        """Initialize a Libration point with the mass parameter and point index."""
        self.system = system
        self.mu = system.mu
        self._position = None
        self._stability_info = None
        self._linear_data = None
        self._energy = None
        self._jacobi_constant = None
        self._cache = {}
        self._var_eq_system = create_var_eq_system(self.mu, name=f"CR3BP Variational Equations for {self.__class__.__name__}")
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    @property
    def idx(self) -> int:
        pass

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
        if self._stability_info is None:
            self.analyze_stability() 
        
        indices = self._stability_info[0] 
        return np.all(np.abs(indices) <= 1.0 + 1e-9)

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
        # Check cache first
        cache_key = ('stability_analysis', discrete, delta)
        cached = self.cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Using cached stability analysis for {type(self).__name__}")
            self._stability_info = cached  # Update instance variable for property access
            return cached
        
        mode_str = "Continuous" if discrete == CONTINUOUS_SYSTEM else "Discrete"
        logger.info(f"Analyzing stability for {type(self).__name__} (mu={self.mu}), mode={mode_str}, delta={delta}.")
        pos = self.position
        A = _jacobian_crtbp(pos[0], pos[1], pos[2], self.mu)
        
        logger.debug(f"Jacobian calculated at position {pos}:\n{A}")

        # Perform eigenvalue decomposition and classification
        stability_info = eigenvalue_decomposition(A, discrete, delta)
        
        # Cache and store in instance variable
        self._stability_info = stability_info
        self.cache_set(cache_key, stability_info)
        
        sn, un, cn, _, _, _ = stability_info
        logger.info(f"Stability analysis complete: {len(sn)} stable, {len(un)} unstable, {len(cn)} center eigenvalues.")
        
        return stability_info

    def cache_get(self, key) -> any:
        """Get item from cache."""
        return self._cache.get(key)
    
    def cache_set(self, key, value) -> any:
        """Set item in cache and return the value."""
        self._cache[key] = value
        return value
    
    def cache_clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.debug(f"Cache cleared for {type(self).__name__}")

    def get_center_manifold(self, max_degree: int):
        """Return (and lazily construct) a CenterManifold of given degree.

        Heavy polynomial data (Hamiltonians in multiple coordinate systems,
        Lie generators, etc.) are cached *inside* the returned CenterManifold,
        not in the LibrationPoint itself.
        """
        from algorithms.center.base import CenterManifold

        if max_degree not in self._cm_registry:
            self._cm_registry[max_degree] = CenterManifold(self, max_degree)
        return self._cm_registry[max_degree]

    def hamiltonian(self, max_deg: int) -> dict:
        """Return all Hamiltonian representations from the associated CenterManifold.

        Keys: 'physical', 'real_normal', 'complex_normal', 'normalized',
        'center_manifold_complex', 'center_manifold_real'.
        """
        cm = self.get_center_manifold(max_deg)
        cm.compute()  # ensures all representations are cached

        reprs = {}
        for label in (
            'physical',
            'real_normal',
            'complex_normal',
            'normalized',
            'center_manifold_complex',
            'center_manifold_real',
        ):
            data = cm.cache_get(('hamiltonian', max_deg, label))
            if data is not None:
                reprs[label] = [arr.copy() for arr in data]
        return reprs

    def generating_functions(self, max_deg: int):
        """Return the Lie-series generating functions from CenterManifold."""
        cm = self.get_center_manifold(max_deg)
        cm.compute()  # ensure they exist
        data = cm.cache_get(('generating_functions', max_deg))
        return [] if data is None else [g.copy() for g in data]

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
    def __init__(self, system: "System"):
        """Initialize a collinear Libration point."""
        if not 0 < system.mu < 0.5:
            raise ValueError(f"Mass parameter mu must be in range (0, 0.5), got {system.mu}")
        super().__init__(system)

    def _find_position(self, primary_interval: list) -> float:
        """
        Find the x-coordinate of a collinear point using retry logic.
        
        Parameters
        ----------
        primary_interval : list
            Initial interval [a, b] to search for the root
            
        Returns
        -------
        float
            x-coordinate of the libration point
            
        Raises
        ------
        RuntimeError
            If both primary and fallback searches fail
        """
        func = lambda x_val: self._dOmega_dx(x_val)
        
        # Try primary interval first
        logger.debug(f"{self.__class__.__name__}: Finding root of dOmega/dx in primary interval {primary_interval}")
        try:
            x = find_root(func, primary_interval, precision=MPMATH_DPS)
            logger.info(f"{self.__class__.__name__} position calculated with primary interval: x = {x}")
            return x
        except ValueError as e:
            err = f"{self.__class__.__name__}: Primary interval {primary_interval} failed: {e}"
            logger.error(err)
            raise RuntimeError(err) from e

    def _solve_gamma_polynomial(self, coeffs: list, gamma_range: tuple) -> float:
        """
        Solve the quintic polynomial for gamma with validation and fallback.
        
        Parameters
        ----------
        coeffs : list
            Polynomial coefficients from highest to lowest degree
        gamma_range : tuple
            (min_gamma, max_gamma) valid range for this point type
        fallback_approx : float
            Fallback approximation if polynomial solving fails
            
        Returns
        -------
        float
            The gamma value for this libration point
        """
        try:
            roots = np.roots(coeffs)
        except Exception as e:
            err = f"{self.__class__.__name__}: Polynomial root finding failed: {e}"
            logger.error(err)
            raise RuntimeError(err) from e
        
        min_gamma, max_gamma = gamma_range
        point_name = self.__class__.__name__[:2]  # 'L1', 'L2', 'L3'
        
        # Find the valid real root
        for root in roots:
            if not np.isreal(root):
                continue
                
            gamma_val = float(root.real)
            
            # Check if it's in the valid range
            if not (min_gamma < gamma_val < max_gamma):
                continue

            return gamma_val
        
        err = f"No valid polynomial root found for {point_name}"
        logger.error(err)
        raise RuntimeError(err)

    @property
    def gamma(self) -> float:
        """
        Get the distance ratio gamma for the libration point, calculated
        with high precision.

        Gamma is defined as the distance from the libration point to the nearest primary,
        normalized by the distance between the primaries.
        - For L1 and L2, gamma = |x_L - (1-mu)|
        - For L3, gamma = |x_L - (-mu)| 
        (Note: This is equivalent to the root of the specific polynomial for each point).

        Returns
        -------
        float
            The gamma value calculated with high precision.
        """
        cached = self.cache_get(('gamma',))
        if cached is not None:
            return cached

        gamma = self._compute_gamma()
        logger.info(f"Gamma for {type(self).__name__} = {gamma}")
        
        return self.cache_set(('gamma',), gamma)

    @property
    def sign(self) -> int:
        """Sign convention (±1) used for local ↔ synodic transformations.

        Following the convention adopted in Gómez et al. (2001):

        * L1, L2  →  -1 ("lower" sign)
        * L3      →  +1 ("upper" sign)
        """
        return 1 if isinstance(self, L3Point) else -1

    @property
    def a(self) -> float:
        """Offset *a* along the x axis used in frame changes.

        The relation x_L = μ + a links the equilibrium x coordinate in
        synodic coordinates (x_L) with the mass parameter μ.  Using the
        distance gamma (``self.gamma``) to the closest primary we obtain:

            a = -1 + gamma   (L1)
            a = -1 - gamma   (L2)
            a =  gamma       (L3)
        """
        if isinstance(self, L1Point):
            return -1 + self.gamma
        elif isinstance(self, L2Point):
            return -1 - self.gamma
        elif isinstance(self, L3Point):
            return self.gamma
        else:
            raise AttributeError("Offset 'a' undefined for this point type.")

    @abstractmethod
    def _compute_gamma(self) -> float:
        """
        Compute the gamma value for this specific libration point.
        
        Returns
        -------
        float
            The gamma value calculated with high precision
        """
        pass

    @abstractmethod
    def _compute_cn(self, n: int) -> float:
        """
        Compute the actual value of cn(mu) without caching.
        This needs to be implemented by subclasses.
        """
        pass

    def _cn(self, n: int) -> float:
        """
        Get the cn coefficient with caching.
        """
        if n < 0:
            raise ValueError(f"Coefficient index n must be non-negative, got {n}")
            
        cached = self.cache_get(('cn', n))
        if cached is not None:
            logger.debug(f"Using cached value for c{n}(mu) = {cached}")
            return cached
            
        # Compute and cache the value
        value = self._compute_cn(n)
        logger.info(f"c{n}(mu) = {value}")
        return self.cache_set(('cn', n), value)

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
        
        # Avoid division by zero (though unlikely for libration points)
        if r1_sq < 1e-16 or r2_sq < 1e-16:
            err = f"x-coordinate too close to primary masses: x={x}"
            logger.error(err)
            raise ValueError(err)

        r1_3 = r1_sq**1.5
        r2_3 = r2_sq**1.5

        term1 = x
        term2 = -(1 - mu) * (x + mu) / r1_3
        term3 = -mu * (x - (1 - mu)) / r2_3
        
        return term1 + term2 + term3

    @property
    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values
        """
        cached = self.cache_get(('linear_modes',))
        if cached is not None:
            return cached
            
        result = self._compute_linear_modes()
        return self.cache_set(('linear_modes',), result)

    def _compute_linear_modes(self):
        """
        Compute the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values for the libration point
        """
        try:
            c2_hp = hp(self._cn(2))
            a_hp = hp(1.0)
            b_hp = hp(2.0) - c2_hp
            c_hp = hp(1.0) + c2_hp - hp(2.0) * (c2_hp ** hp(2.0))
            
            discriminant_hp = (b_hp ** hp(2.0)) - hp(4.0) * a_hp * c_hp
            
            if float(discriminant_hp) < 0:
                err = f"Discriminant for linear modes is negative: {float(discriminant_hp)}. c2={float(c2_hp)}"
                logger.error(err)
                raise RuntimeError(err)

            sqrt_discriminant_hp = discriminant_hp.sqrt()
            
            eta1_hp = (-b_hp - sqrt_discriminant_hp) / (hp(2.0) * a_hp)
            eta2_hp = (-b_hp + sqrt_discriminant_hp) / (hp(2.0) * a_hp)

            # Determine which eta is positive (for lambda1) and which is negative (for omega1)
            if float(eta1_hp) > float(eta2_hp):
                lambda1_hp = eta1_hp.sqrt() if float(eta1_hp) > 0 else hp(0.0)
                omega1_hp = (-eta2_hp).sqrt() if float(eta2_hp) < 0 else hp(0.0)
            else:
                lambda1_hp = eta2_hp.sqrt() if float(eta2_hp) > 0 else hp(0.0)
                omega1_hp = (-eta1_hp).sqrt() if float(eta1_hp) < 0 else hp(0.0)
            
            # Vertical frequency
            omega2_hp = c2_hp.sqrt() if float(c2_hp) >= 0 else hp(0.0)

            return (float(lambda1_hp), float(omega1_hp), float(omega2_hp))
            
        except Exception as e:
            err = f"Failed to calculate linear modes with Number: {e}"
            logger.error(err)
            raise RuntimeError(err) from e

    def _scale_factor(self, lambda1, omega1):
        """
        Calculate the normalization factors s1 and s2 used in the normal form transformation.
        
        Parameters
        ----------
        lambda1 : float
            The hyperbolic mode value
        omega1 : float
            The elliptic mode value
            
        Returns
        -------
        s1, s2 : tuple of float
            The normalization factors for the hyperbolic and elliptic components
        """
        c2_hp = hp(self._cn(2))
        lambda1_hp = hp(lambda1)
        omega1_hp = hp(omega1)

        # Common terms
        term_lambda = (hp(4.0) + hp(3.0) * c2_hp) * (lambda1_hp ** hp(2.0))
        term_omega = (hp(4.0) + hp(3.0) * c2_hp) * (omega1_hp ** hp(2.0))
        base_term = hp(4.0) + hp(5.0) * c2_hp - hp(6.0) * (c2_hp ** hp(2.0))

        # Calculate expressions under square root
        expr1_hp = hp(2.0) * lambda1_hp * (term_lambda + base_term)
        expr2_hp = omega1_hp * (term_omega - base_term)
        
        # Validate expressions are positive
        if float(expr1_hp) < 0:
            err = f"Expression for s1 is negative (hp): {float(expr1_hp)}."
            logger.error(err)
            raise RuntimeError(err)
            
        if float(expr2_hp) < 0:
            err = f"Expression for s2 is negative (hp): {float(expr2_hp)}."
            logger.error(err)
            raise RuntimeError(err)
        
        return float(expr1_hp.sqrt()), float(expr2_hp.sqrt())

    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the 6x6 symplectic matrix C of eq. (10) that sends H_2 to
        lambda_1 x px + (omega_1/2)(y²+p_y²) + (omega_2/2)(z²+p_z²).

        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix and Cinv is its inverse
        """
        # Check cache first
        cache_key = ('normal_form_transform',)
        cached = self.cache_get(cache_key)
        if cached is not None:
            return cached
            
        # Get the numerical parameters
        lambda1, omega1, omega2 = self.linear_modes
        c2 = self._cn(2)
        s1, s2 = self._scale_factor(lambda1, omega1)
        
        # Build the 6x6 transformation matrix C numerically
        C = np.zeros((6, 6))
        
        # First row
        C[0, 0] = 2 * lambda1 / s1
        C[0, 3] = -2 * lambda1 / s1
        C[0, 4] = 2 * omega1 / s2
        
        # Second row
        C[1, 0] = (lambda1**2 - 2*c2 - 1) / s1
        C[1, 1] = (-omega1**2 - 2*c2 - 1) / s2
        C[1, 3] = (lambda1**2 - 2*c2 - 1) / s1
        
        # Third row
        C[2, 2] = 1 / np.sqrt(omega2)
        
        # Fourth row
        C[3, 0] = (lambda1**2 + 2*c2 + 1) / s1
        C[3, 1] = (-omega1**2 + 2*c2 + 1) / s2
        C[3, 3] = (lambda1**2 + 2*c2 + 1) / s1
        
        # Fifth row
        C[4, 0] = (lambda1**3 + (1 - 2*c2)*lambda1) / s1
        C[4, 3] = (-lambda1**3 - (1 - 2*c2)*lambda1) / s1
        C[4, 4] = (-omega1**3 + (1 - 2*c2)*omega1) / s2
        
        # Sixth row
        C[5, 5] = np.sqrt(omega2)
        
        # Compute the inverse
        Cinv = np.linalg.inv(C)
        
        # Cache the result
        result = (C, Cinv)
        self.cache_set(cache_key, result)
        
        return result

    def _get_linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        LinearData
            Object containing the linear data for the Libration point
        """
        # Get cached values
        lambda1, omega1, omega2 = self.linear_modes
        C, Cinv = self.normal_form_transform()
        
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


class L1Point(CollinearPoint):
    """
    L1 Libration point, located between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        """Initialize the L1 Libration point."""
        super().__init__(system)
        
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L1 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L1
        """
        # L1 is between the primaries: -mu < x < 1-mu
        primary_interval = [-self.mu + 0.01, 1 - self.mu - 0.01]

        x = self._find_position(primary_interval)
        return np.array([x, 0, 0], dtype=np.float64)

    def _compute_gamma(self) -> float:
        """
        Compute gamma for L1 point by solving the quintic polynomial equation.
        For L1, gamma is the distance from the second primary (smaller mass).
        """
        mu = self.mu
        
        # Coefficients for L1 quintic: x^5 - (3-μ)x^4 + (3-2μ)x^3 - μx^2 + 2μx - μ = 0
        coeffs = [1, -(3-mu), (3-2*mu), -mu, 2*mu, -mu]
        return self._solve_gamma_polynomial(coeffs, [0, 1])

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L1 using Jorba & Masdemont (1999), eq. (3).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 - gamma)**(n+1))
        
        return term1 * (term2 + term3)

    @property
    def idx(self) -> int:
        return 1


class L2Point(CollinearPoint):
    """
    L2 Libration point, located beyond the smaller primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        """Initialize the L2 Libration point."""
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L2 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L2
        """
        # L2 is beyond the smaller primary: x > 1-mu
        primary_interval = [1 - self.mu + 0.001, 2.0]

        x = self._find_position(primary_interval)
        return np.array([x, 0, 0], dtype=np.float64)

    def _compute_gamma(self) -> float:
        """
        Compute gamma for L2 point by solving the quintic polynomial equation.
        For L2, gamma is the distance from the second primary (smaller mass).
        """
        mu = self.mu
        
        # Coefficients for L2 quintic: x^5 + (3-μ)x^4 + (3-2μ)x^3 - μx^2 - 2μx - μ = 0
        coeffs = [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]

        return self._solve_gamma_polynomial(coeffs, [0, 1])

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L2 using Jorba & Masdemont (1999), eq. (3).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = ((-1)**n) * mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)

    @property
    def idx(self) -> int:
        return 2


class L3Point(CollinearPoint):
    """
    L3 Libration point, located beyond the larger primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        """Initialize the L3 Libration point."""
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the L3 point by finding the root of dOmega/dx.
        
        Returns
        -------
        ndarray
            3D vector [x, 0, 0] giving the position of L3
        """
        # L3 is beyond the larger primary: x < -mu
        primary_interval = [-1.5, -self.mu - 0.001]

        x = self._find_position(primary_interval)
        return np.array([x, 0, 0], dtype=np.float64)

    def _compute_gamma(self) -> float:
        """
        Compute gamma for L3 point by solving the quintic polynomial equation.
        For L3, gamma is the distance from the first primary (larger mass).
        """
        mu = self.mu
        mu1 = 1 - mu  # mass of larger primary
        
        # Coefficients for L3 quintic: x^5 + (2+μ)x^4 + (1+2μ)x^3 - μ₁x^2 - 2μ₁x - μ₁ = 0
        coeffs = [1, (2+mu), (1+2*mu), -mu1, -2*mu1, -mu1]

        return self._solve_gamma_polynomial(coeffs, [0.5, 1.5])

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L3 using Jorba & Masdemont (1999), eq. (3).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = ((-1)**n) / (gamma**3)
        term2 = (1 - mu)
        term3 = mu * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)

    @property
    def idx(self) -> int:
        return 3


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
