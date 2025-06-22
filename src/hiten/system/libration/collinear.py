r"""
hiten.system.libration.collinear
==========================

Collinear libration points :math:`L_1`, :math:`L_2` and :math:`L_3` of the circular restricted three body problem (CR3BP).

The module defines:

* :pyclass:`CollinearPoint` - an abstract helper encapsulating the geometry shared by the collinear points.
* :pyclass:`L1Point`, :pyclass:`L2Point`, :pyclass:`L3Point` - concrete equilibria located on the x-axis connecting the primaries.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.utils.config import MPMATH_DPS
from hiten.system.libration.base import LibrationPoint, LinearData
from hiten.utils.log_config import logger
from hiten.utils.precision import find_root, hp

if TYPE_CHECKING:
    from hiten.system.base import System


class CollinearPoint(LibrationPoint):
    r"""
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
        r"""Initialize a collinear Libration point."""
        if not 0 < system.mu < 0.5:
            raise ValueError(f"Mass parameter mu must be in range (0, 0.5), got {system.mu}")
        super().__init__(system)

    def _find_position(self, primary_interval: list) -> float:
        r"""
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
        r"""
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
        r"""
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
        r"""
        Sign convention (±1) used for local ↔ synodic transformations.

        Following the convention adopted in Gómez et al. (2001):

        * L1, L2  →  -1 ("lower" sign)
        * L3      →  +1 ("upper" sign)
        """
        return 1 if isinstance(self, L3Point) else -1

    @property
    def a(self) -> float:
        r"""
        Offset *a* along the x axis used in frame changes.

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
        r"""
        Compute the gamma value for this specific libration point.
        
        Returns
        -------
        float
            The gamma value calculated with high precision
        """
        pass

    @abstractmethod
    def _compute_cn(self, n: int) -> float:
        r"""
        Compute the actual value of cn(mu) without caching.
        This needs to be implemented by subclasses.
        """
        pass

    def _cn(self, n: int) -> float:
        r"""
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
        r"""
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
        r"""
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
        r"""
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
        r"""
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
        r"""
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
        r"""
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
    r"""
    L1 Libration point, located between the two primary bodies.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        r"""
        Initialize the L1 Libration point.
        """
        super().__init__(system)
        
    def _calculate_position(self) -> np.ndarray:
        r"""
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
        r"""
        Compute gamma for L1 point by solving the quintic polynomial equation.
        For L1, gamma is the distance from the second primary (smaller mass).
        """
        mu = self.mu
        
        # Coefficients for L1 quintic: x^5 - (3-μ)x^4 + (3-2μ)x^3 - μx^2 + 2μx - μ = 0
        coeffs = [1, -(3-mu), (3-2*mu), -mu, 2*mu, -mu]
        return self._solve_gamma_polynomial(coeffs, [0, 1])

    def _compute_cn(self, n: int) -> float:
        r"""
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
    r"""
    L2 Libration point, located beyond the smaller primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        r"""
        Initialize the L2 Libration point.
        """
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        r"""
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
        r"""
        Compute gamma for L2 point by solving the quintic polynomial equation.
        For L2, gamma is the distance from the second primary (smaller mass).
        """
        mu = self.mu
        
        # Coefficients for L2 quintic: x^5 + (3-μ)x^4 + (3-2μ)x^3 - μx^2 - 2μx - μ = 0
        coeffs = [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]

        return self._solve_gamma_polynomial(coeffs, [0, 1])

    def _compute_cn(self, n: int) -> float:
        r"""
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
    r"""
    L3 Libration point, located beyond the larger primary body.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    """
    
    def __init__(self, system: "System"):
        r"""
        Initialize the L3 Libration point.
        """
        super().__init__(system)
    
    def _calculate_position(self) -> np.ndarray:
        r"""
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
        r"""
        Compute gamma for L3 point by solving the quintic polynomial equation.
        For L3, gamma is the distance from the first primary (larger mass).
        """
        mu = self.mu
        mu1 = 1 - mu  # mass of larger primary
        
        # Coefficients for L3 quintic: x^5 + (2+μ)x^4 + (1+2μ)x^3 - μ₁x^2 - 2μ₁x - μ₁ = 0
        coeffs = [1, (2+mu), (1+2*mu), -mu1, -2*mu1, -mu1]

        return self._solve_gamma_polynomial(coeffs, [0.5, 1.5])

    def _compute_cn(self, n: int) -> float:
        r"""
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
