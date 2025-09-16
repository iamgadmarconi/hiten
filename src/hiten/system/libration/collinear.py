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

from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.utils.config import MPMATH_DPS
from hiten.system.libration.base import LibrationPoint, LinearData
from hiten.utils.log_config import logger
from hiten.algorithms.utils.precision import find_root, hp

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
        if not 0 < system.mu < 0.5:
            raise ValueError(f"Mass parameter mu must be in range (0, 0.5), got {system.mu}")
        super().__init__(system)

    @property
    def gamma(self) -> float:
        """
        Get the distance ratio gamma for the libration point, calculated
        with high precision.

        Gamma is defined as the distance from the libration point to the nearest primary,
        normalized by the distance between the primaries.
        - For L1 and L2, gamma = |x_L - (1-mu)|
        - For L3, gamma = |x_L + mu| 
        (Note: This is equivalent to the root of the specific polynomial for each point).

        Returns
        -------
        float
            The gamma value calculated with high precision (dimensionless).
        """
        cached = self.cache_get(('gamma',))
        if cached is not None:
            return cached

        gamma = self._compute_gamma()
        logger.info(f"Gamma for {type(self).__name__} = {gamma}")
        
        return self.cache_set(('gamma',), gamma)

    @property
    def sign(self) -> int:
        """
        Sign convention (+-1) used for local <-> synodic transformations.

        Following the convention adopted in Gomez et al. (2001):

        * L1, L2  ->  -1 ("lower" sign)
        * L3      ->  +1 ("upper" sign)
        
        Returns
        -------
        int
            The sign convention for coordinate transformations.
        """
        return 1 if isinstance(self, L3Point) else -1

    @property
    def a(self) -> float:
        """
        Offset a along the x axis used in frame changes.

        The relation x_L = mu + a links the equilibrium x coordinate in
        synodic coordinates (x_L) with the mass parameter mu.  Using the
        distance gamma (self.gamma) to the closest primary we obtain:

            a = -1 + gamma   (L1)
            a = -1 - gamma   (L2)
            a =  gamma       (L3)
            
        Returns
        -------
        float
            The offset value a (dimensionless).
            
        Raises
        ------
        AttributeError
            If the offset is undefined for this point type.
        """
        if isinstance(self, L1Point):
            return -1 + self.gamma
        elif isinstance(self, L2Point):
            return -1 - self.gamma
        elif isinstance(self, L3Point):
            return self.gamma
        else:
            raise AttributeError("Offset 'a' undefined for this point type.")

    @property
    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values in nondimensional units.
        """
        cached = self.cache_get(('linear_modes',))
        if cached is not None:
            return cached
            
        result = self._compute_linear_modes()
        self.cache_set(('linear_modes',), result)
        return result

    @property
    @abstractmethod
    def _position_search_interval(self) -> list:
        """
        Defines the search interval for finding the x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
        """
        pass

    @property
    @abstractmethod
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        Defines the quintic polynomial for gamma calculation.
        
        Returns
        -------
        tuple
            (coefficients, search_range) where coefficients is a list of
            polynomial coefficients and search_range is (min_gamma, max_gamma).
        """
        pass

    def _find_position(self, primary_interval: list) -> float:
        """
        Find the x-coordinate of a collinear point using retry logic.
        
        Parameters
        ----------
        primary_interval : list
            Initial interval [a, b] to search for the root in nondimensional units.
            
        Returns
        -------
        float
            x-coordinate of the libration point in nondimensional units.
            
        Raises
        ------
        RuntimeError
            If both primary and fallback searches fail.
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
            Polynomial coefficients from highest to lowest degree.
        gamma_range : tuple
            (min_gamma, max_gamma) valid range for this point type.
            
        Returns
        -------
        float
            The gamma value for this libration point (dimensionless).
            
        Raises
        ------
        RuntimeError
            If polynomial root finding fails or no valid root is found.
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

    @abstractmethod
    def _compute_gamma(self) -> float:
        """
        Compute the gamma value for this specific libration point.
        
        Returns
        -------
        float
            The gamma value calculated with high precision (dimensionless).
        """
        pass

    @abstractmethod
    def _compute_cn(self, n: int) -> float:
        """
        Compute the actual value of cn(mu) without caching.
        This needs to be implemented by subclasses.
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        pass

    def _cn(self, n: int) -> float:
        """
        Get the cn coefficient with caching.
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
            
        Raises
        ------
        ValueError
            If n is negative.
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
            x-coordinate in the rotating frame (nondimensional units).
        
        Returns
        -------
        float
            Value of dOmega/dx at the given x-coordinate (nondimensional units).
            
        Raises
        ------
        ValueError
            If x-coordinate is too close to primary masses.
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

    def _J_hess_H2(self) -> np.ndarray:
        """
        Compute the 6x6 symplectic matrix for the quadratic Hamiltonian H2.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            The symplectic matrix J for the quadratic Hamiltonian.
        """
        c2 = self._cn(2)
        omega2 = np.sqrt(c2)

        J_planar = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [2.0 * c2, 0.0, 0.0, 1.0],
            [0.0, -c2, -1.0, 0.0],
        ], dtype=np.float64)

        J_vert = np.array([[0.0, omega2], [-omega2, 0.0]], dtype=np.float64)

        J_full = np.zeros((6, 6), dtype=np.float64)
        J_full[:4, :4] = J_planar
        J_full[4:, 4:] = J_vert

        return J_full

    def _compute_linear_modes(self):
        """
        Compute the linear modes (lambda1, omega1, omega2) for the libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) in nondimensional units.
            
        Raises
        ------
        RuntimeError
            If no real eigenvalues are found or expected frequencies are missing.
        """
        J_full = self._J_hess_H2()
        c2 = self._cn(2)
        omega2_expected = np.sqrt(c2)

        eigvals, _ = np.linalg.eig(J_full)

        real_mask = np.abs(eigvals.imag) < 1e-12
        imag_mask = ~real_mask

        real_eigs = eigvals[real_mask].real
        if real_eigs.size == 0:
            raise RuntimeError("No real eigen-values found while calculating linear modes.")
        lambda1 = float(np.max(np.abs(real_eigs)))

        imag_eigs = eigvals[imag_mask]
        omegas = np.unique(np.round(np.abs(imag_eigs.imag), decimals=12))
        if omegas.size < 2:
            raise RuntimeError(f"Expected two distinct imaginary frequencies, got {omegas.size}.")

        idx_vert = int(np.argmin(np.abs(omegas - omega2_expected)))
        omega2_val = float(omegas[idx_vert])
        omega1_val = float(omegas[1 - idx_vert])  # The other one

        if omega1_val < omega2_val:
            omega1_val, omega2_val = omega2_val, omega1_val

        return (float(lambda1), float(omega1_val), float(omega2_val))

    def _scale_factor(self, lambda1, omega1):
        """
        Calculate the normalization factors s1 and s2 used in the normal form transformation.
        
        Parameters
        ----------
        lambda1 : float
            The hyperbolic mode value (nondimensional units).
        omega1 : float
            The elliptic mode value (nondimensional units).
            
        Returns
        -------
        tuple
            (s1, s2) normalization factors for the hyperbolic and elliptic components.
            
        Raises
        ------
        RuntimeError
            If the expressions for s1 or s2 are negative.
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

    def _get_linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        :class:`~hiten.system.libration.LinearData`
            Object containing the linear data for the Libration point.
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
            omega3=None,
            C=C, 
            Cinv=Cinv
        )

    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the point by finding the root of dOmega/dx
        within a search interval defined by the concrete subclass.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            Position vector [x, y, z] in nondimensional units.
        """
        x = self._find_position(self._position_search_interval)
        return np.array([x, 0, 0], dtype=np.float64)

    def _compute_gamma(self) -> float:
        """
        Compute gamma for the libration point by solving the quintic polynomial
        defined by the concrete subclass.
        
        Returns
        -------
        float
            The gamma value (dimensionless).
        """
        coeffs, search_range = self._gamma_poly_def
        return self._solve_gamma_polynomial(coeffs, search_range)

    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the 6x6 symplectic matrix C that sends H2 to normal form.

        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix and Cinv is its inverse.
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
        
        # Add a safeguard for the vertical frequency omega2 to prevent division by zero
        if abs(omega2) < 1e-12:
            logger.warning(f"Vertical frequency omega2 is very small ({omega2:.2e}). Transformation matrix may be ill-conditioned.")
            sqrt_omega2 = 1e-6  # Use a small regularizing value
        else:
            sqrt_omega2 = np.sqrt(omega2)

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
        C[2, 2] = 1 / sqrt_omega2
        
        # Fourth row
        C[3, 0] = (lambda1**2 + 2*c2 + 1) / s1
        C[3, 1] = (-omega1**2 + 2*c2 + 1) / s2
        C[3, 3] = (lambda1**2 + 2*c2 + 1) / s1
        
        # Fifth row
        C[4, 0] = (lambda1**3 + (1 - 2*c2)*lambda1) / s1
        C[4, 3] = (-lambda1**3 - (1 - 2*c2)*lambda1) / s1
        C[4, 4] = (-omega1**3 + (1 - 2*c2)*omega1) / s2
        
        # Sixth row
        C[5, 5] = sqrt_omega2
        
        # Compute the inverse
        Cinv = np.linalg.inv(C)
        
        # Cache the result
        result = (C, Cinv)
        self.cache_set(cache_key, result)
        
        return result


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

    @property
    def _position_search_interval(self) -> list:
        """
        Search interval for L1's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L1 is between the primaries: -mu < x < 1-mu.
        """
        # L1 is between the primaries: -mu < x < 1-mu
        return [-self.mu + 0.01, 1 - self.mu - 0.01]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        Quintic polynomial definition for L1's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L1 quintic polynomial.
            The polynomial is: x^5 - (3-mu)x^4 + (3-2mu)x^3 - mux^2 + 2mux - mu = 0.
        """
        mu = self.mu
        # Coefficients for L1 quintic: x^5 - (3-mu)x^4 + (3-2mu)x^3 - mux^2 + 2mux - mu = 0
        coeffs = [1, -(3 - mu), (3 - 2 * mu), -mu, 2 * mu, -mu]
        return coeffs, (0, 1)

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L1 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 - gamma)**(n+1))
        
        return term1 * (term2 + term3)


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

    @property
    def _position_search_interval(self) -> list:
        """
        Search interval for L2's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L2 is beyond the smaller primary: x > 1-mu.
        """
        # L2 is beyond the smaller primary: x > 1-mu
        return [1 - self.mu + 0.001, 2.0]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        Quintic polynomial definition for L2's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L2 quintic polynomial.
            The polynomial is: x^5 + (3-mu)x^4 + (3-2mu)x^3 - mux^2 - 2mux - mu = 0.
        """
        mu = self.mu
        # Coefficients for L2 quintic: x^5 + (3-mu)x^4 + (3-2mu)x^3 - mux^2 - 2mux - mu = 0
        coeffs = [1, (3 - mu), (3 - 2 * mu), -mu, -2 * mu, -mu]
        return coeffs, (0, 1)

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L2 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = ((-1)**n) * mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)


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

    @property
    def _position_search_interval(self) -> list:
        """
        Search interval for L3's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L3 is beyond the larger primary: x < -mu.
        """
        # L3 is beyond the larger primary: x < -mu
        return [-1.5, -self.mu - 0.001]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        Quintic polynomial definition for L3's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L3 quintic polynomial.
            The polynomial is: x^5 + (2+mu)x^4 + (1+2mu)x^3 - mu_1x^2 - 2mu_1x - mu_1 = 0.
        """
        mu = self.mu
        mu1 = 1 - mu  # mass of larger primary
        # Coefficients for L3 quintic: x^5 + (2+mu)x^4 + (1+2mu)x^3 - mu_1x^2 - 2mu_1x - mu_1 = 0
        coeffs = [1, (2 + mu), (1 + 2 * mu), -mu1, -2 * mu1, -mu1]
        return coeffs, (0.5, 1.5)

    def _compute_cn(self, n: int) -> float:
        """
        Compute cn coefficient for L3 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = ((-1)**n) / (gamma**3)
        term2 = (1 - mu)
        term3 = mu * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)
