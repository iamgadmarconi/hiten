import numba
import numpy as np
import mpmath as mp
from scipy.special import legendre


from log_config import logger
from system.libration import CollinearPoint, L1Point, L2Point, L3Point


def _legendre_coefficient(n: int, mu: float, libration_point: CollinearPoint) -> float:
    """
    Calculate the Legendre coefficient of degree n for a given value of x.

    Parameters:
        n: Degree of the Legendre polynomial
        mu: Mass parameter of the system
        libration_point: Libration point to evaluate the polynomial at

    Returns:
        Legendre polynomial of degree n evaluated at x
    """


    if isinstance(libration_point, L1Point):
        term1 = 1 / libration_point.gamma ** 3
        term2 = (1) ** n * mu
        term3 = (-1) ** n * ( (1-mu) * libration_point.gamma ** (n+1) / (1 - libration_point.gamma) ** (n+1) )

        c_n = term1 * (term2 + term3)

        return c_n
    
    elif isinstance(libration_point, L2Point):
        term1 = 1 / libration_point.gamma ** 3
        term2 = (-1) ** n * mu
        term3 = (-1) ** n * ( (1-mu) * libration_point.gamma ** (n+1) / (1 + libration_point.gamma) ** (n+1) )

        c_n = term1 * (term2 + term3)

        return c_n
    
    elif isinstance(libration_point, L3Point):
        term1 = (-1) ** n / libration_point.gamma ** 3
        term2 = 1-mu
        term3 = mu * libration_point.gamma ** (n+1) / (1 + libration_point.gamma) ** (n+1)

        c_n = term1 * (term2 + term3)

        return c_n
    
    else:
        raise ValueError(f"Invalid libration point: {libration_point}")

def _evaluate_poly(n, x, rho):
    """
    Evaluate the Legendre polynomial of degree n at x/rho, and multiply by rho^n.
    
    Parameters:
        n: Degree of the Legendre polynomial
        x: x-coordinate
        rho: Distance from origin (sqrt(x^2 + y^2 + z^2))
        
    Returns:
        rho^n * P_n(x/rho)
    """
    P_n = legendre(n)
    return rho ** n * P_n(x / rho)

def legendre_series(state: np.ndarray, n: int, mu: float, libration_point: CollinearPoint) -> float:
    """
    Calculate the Legendre series expansion of a given function at a specific libration point.

    Parameters:
        state: State of the body in canonical (rotating) frame
        n: Degree of the Legendre polynomial
        mu: Mass parameter of the system
        libration_point: Libration point to evaluate the series at

    Returns:
        Legendre series expansion of the function at the given libration point
    """

    if n < 2:
        raise ValueError("`n` must be greater than or equal to 2")
    
    series = 0

    x, y, z = state[:3]

    rho = np.sqrt(x**2 + y**2 + z**2)

    for i in range(2, n):
        series += _legendre_coefficient(i, mu, libration_point) * _evaluate_poly(i, x, rho)

    return series
