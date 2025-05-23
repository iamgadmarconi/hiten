"""
Precision utilities for arbitrary precision arithmetic.

This module provides utilities for handling high precision calculations
using mpmath when needed, with fallback to standard precision.
"""

import numpy as np
import mpmath as mp
from typing import Union, Callable, Any
from config import USE_ARBITRARY_PRECISION, MPMATH_DPS, NUMPY_DTYPE_REAL, NUMPY_DTYPE_COMPLEX
from utils.log_config import logger


def with_precision(precision: int = None):
    """
    Context manager for setting mpmath precision.
    
    Parameters
    ----------
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
    """
    if precision is None:
        precision = MPMATH_DPS
    return mp.workdps(precision)


def high_precision_division(numerator: float, denominator: float, precision: int = None) -> float:
    """
    Perform high precision division if enabled, otherwise standard division.
    
    Parameters
    ----------
    numerator : float
        Numerator value
    denominator : float  
        Denominator value
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
        
    Returns
    -------
    float
        Result of division with appropriate precision
    """
    if not USE_ARBITRARY_PRECISION:
        return numerator / denominator
        
    if precision is None:
        precision = MPMATH_DPS
        
    with mp.workdps(precision):
        mp_num = mp.mpf(numerator)
        mp_den = mp.mpf(denominator)
        result = mp_num / mp_den
        return float(result)


def high_precision_sqrt(value: float, precision: int = None) -> float:
    """
    Compute square root with high precision if enabled.
    
    Parameters
    ----------
    value : float
        Value to take square root of
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
        
    Returns
    -------
    float
        Square root with appropriate precision
    """
    if not USE_ARBITRARY_PRECISION:
        return np.sqrt(value)
        
    if precision is None:
        precision = MPMATH_DPS
        
    with mp.workdps(precision):
        mp_val = mp.mpf(value)
        result = mp.sqrt(mp_val)
        return float(result)


def high_precision_power(base: float, exponent: float, precision: int = None) -> float:
    """
    Compute power with high precision if enabled.
    
    Parameters
    ----------
    base : float
        Base value
    exponent : float
        Exponent value
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
        
    Returns
    -------
    float
        Result with appropriate precision
    """
    if not USE_ARBITRARY_PRECISION:
        return base ** exponent
        
    if precision is None:
        precision = MPMATH_DPS
        
    with mp.workdps(precision):
        mp_base = mp.mpf(base)
        mp_exp = mp.mpf(exponent)
        result = mp_base ** mp_exp
        return float(result)


def high_precision_findroot(func: Callable, x0: Union[float, list], precision: int = None) -> float:
    """
    Find root with high precision using mpmath.
    
    Parameters
    ----------
    func : callable
        Function to find root of
    x0 : float or list
        Initial guess or bracket
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
        
    Returns
    -------
    float
        Root with high precision
    """
    if precision is None:
        precision = MPMATH_DPS
        
    with mp.workdps(precision):
        root = mp.findroot(func, x0)
        return float(root)


def get_numpy_dtype_real() -> np.dtype:
    """Get the configured numpy real dtype."""
    return np.dtype(NUMPY_DTYPE_REAL)


def get_numpy_dtype_complex() -> np.dtype:
    """Get the configured numpy complex dtype.""" 
    return np.dtype(NUMPY_DTYPE_COMPLEX)


def log_precision_info():
    """Log current precision settings."""
    if USE_ARBITRARY_PRECISION:
        logger.info(f"Using arbitrary precision with {MPMATH_DPS} decimal places")
    else:
        logger.info(f"Using standard precision (real: {NUMPY_DTYPE_REAL}, complex: {NUMPY_DTYPE_COMPLEX})") 