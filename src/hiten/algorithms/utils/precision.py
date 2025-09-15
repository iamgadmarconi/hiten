"""High-precision arithmetic utilities for numerical computations.

This module provides comprehensive high-precision arithmetic capabilities for
numerical computations in the circular restricted three-body problem. It supports
both standard floating-point precision and arbitrary precision arithmetic using
the mpmath library when enabled.

The module uses mpmath for arbitrary precision arithmetic when enabled,
falling back to standard NumPy operations for performance when high precision
is not required.

Notes
-----
The precision level is controlled by the USE_ARBITRARY_PRECISION configuration
flag. When disabled, the module falls back to standard floating-point arithmetic
for better performance.
"""

from typing import Callable, Union

import mpmath as mp
import numpy as np

from hiten.algorithms.utils.config import USE_ARBITRARY_PRECISION


class _Number:
    """
    A number class that supports high-precision arithmetic operations.
    
    This class wraps numeric values and provides operator overloading for
    natural mathematical syntax while maintaining high precision when enabled.
    It supports both arbitrary precision arithmetic using mpmath and standard
    floating-point arithmetic for performance.
    
    Parameters
    ----------
    value : float, int, str, or _Number
        The numeric value to wrap
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Attributes
    ----------
    value : mpmath.mpf or float
        The wrapped numeric value
    precision : int
        The precision level in decimal places
        
    Notes
    -----
    The class automatically handles precision management and provides
    seamless integration with standard Python numeric operations.
    """
    
    def __init__(self, value: Union[float, int, str, '_Number'], precision: int = None):
        self.precision = precision if precision is not None else 100
        
        if isinstance(value, _Number):
            self.value = value.value
            self.precision = max(self.precision, value.precision)
        elif USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                self.value = mp.mpf(value)
        else:
            self.value = float(value)
    
    def _ensure_precision_number(self, other) -> '_Number':
        """Convert other operand to _Number if needed."""
        if not isinstance(other, _Number):
            return _Number(other, self.precision)
        return other
    
    def _binary_operation(self, other, operation):
        """Perform a binary operation with proper precision handling."""
        other = self._ensure_precision_number(other)
        max_precision = max(self.precision, other.precision)
        
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(max_precision):
                if operation == 'add':
                    result_value = self.value + other.value
                elif operation == 'sub':
                    result_value = self.value - other.value
                elif operation == 'mul':
                    result_value = self.value * other.value
                elif operation == 'truediv':
                    result_value = self.value / other.value
                elif operation == 'pow':
                    result_value = self.value ** other.value
                elif operation == 'mod':
                    result_value = self.value % other.value
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
        else:
            if operation == 'add':
                result_value = float(self.value) + float(other.value)
            elif operation == 'sub':
                result_value = float(self.value) - float(other.value)
            elif operation == 'mul':
                result_value = float(self.value) * float(other.value)
            elif operation == 'truediv':
                result_value = float(self.value) / float(other.value)
            elif operation == 'pow':
                result_value = float(self.value) ** float(other.value)
            elif operation == 'mod':
                result_value = float(self.value) % float(other.value)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        return _Number(result_value, max_precision)
    
    def __add__(self, other):
        return self._binary_operation(other, 'add')
    
    def __radd__(self, other):
        return _Number(other, self.precision).__add__(self)
    
    def __sub__(self, other):
        return self._binary_operation(other, 'sub')
    
    def __rsub__(self, other):
        return _Number(other, self.precision).__sub__(self)
    
    def __mul__(self, other):
        return self._binary_operation(other, 'mul')
    
    def __rmul__(self, other):
        return _Number(other, self.precision).__mul__(self)
    
    def __truediv__(self, other):
        return self._binary_operation(other, 'truediv')
    
    def __rtruediv__(self, other):
        return _Number(other, self.precision).__truediv__(self)
    
    def __pow__(self, other):
        return self._binary_operation(other, 'pow')
    
    def __rpow__(self, other):
        return _Number(other, self.precision).__pow__(self)
    
    def __mod__(self, other):
        return self._binary_operation(other, 'mod')
    
    def __rmod__(self, other):
        return _Number(other, self.precision).__mod__(self)
    
    def __neg__(self):
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = -self.value
        else:
            result_value = -float(self.value)
        return _Number(result_value, self.precision)
    
    def __abs__(self):
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = abs(self.value)
        else:
            result_value = abs(float(self.value))
        return _Number(result_value, self.precision)
    
    def __eq__(self, other):
        other = self._ensure_precision_number(other)
        return float(self.value) == float(other.value)
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __lt__(self, other):
        other = self._ensure_precision_number(other)
        return float(self.value) < float(other.value)
    
    def __le__(self, other):
        other = self._ensure_precision_number(other)
        return float(self.value) <= float(other.value)
    
    def __gt__(self, other):
        other = self._ensure_precision_number(other)
        return float(self.value) > float(other.value)
    
    def __ge__(self, other):
        other = self._ensure_precision_number(other)
        return float(self.value) >= float(other.value)
    
    def sqrt(self):
        """Compute square root with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.sqrt(self.value)
        else:
            result_value = np.sqrt(float(self.value))
        return _Number(result_value, self.precision)
    
    def sin(self):
        """Compute sine with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.sin(self.value)
        else:
            result_value = np.sin(float(self.value))
        return _Number(result_value, self.precision)
    
    def cos(self):
        """Compute cosine with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.cos(self.value)
        else:
            result_value = np.cos(float(self.value))
        return _Number(result_value, self.precision)
    
    def exp(self):
        """Compute exponential with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.exp(self.value)
        else:
            result_value = np.exp(float(self.value))
        return _Number(result_value, self.precision)
    
    def log(self, base=None):
        """Compute logarithm with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                if base is None:
                    result_value = mp.log(self.value)
                else:
                    result_value = mp.log(self.value) / mp.log(base)
        else:
            if base is None:
                result_value = np.log(float(self.value))
            else:
                result_value = np.log(float(self.value)) / np.log(float(base))
        return _Number(result_value, self.precision)
    
    def __float__(self):
        return float(self.value)
    
    def __int__(self):
        return int(float(self.value))
    
    def __str__(self):
        return str(float(self.value))
    
    def __repr__(self):
        return f"_Number({float(self.value)}, precision={self.precision})"


def hp(value: Union[float, int, str], precision: int = None) -> _Number:
    """
    Create a high-precision number instance.
    
    Convenience factory function for creating high precision numbers.
    This is the recommended way to create _Number instances for better
    readability and consistency.
    
    Parameters
    ----------
    value : float, int, or str
        The numeric value to wrap
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    _Number
        High precision number instance
        
    Examples
    --------
    >>> a = hp(2.5)
    >>> b = hp(3.0)
    >>> result = (a ** b) / hp(7.0)
    >>> print(result)
    2.6785714285714286
    """
    return _Number(value, precision)


def with_precision(precision: int = None):
    """
    Context manager for setting mpmath precision.
    
    This function provides a context manager for temporarily setting the
    precision level for mpmath operations. It automatically restores the
    previous precision level when exiting the context.
    
    Parameters
    ----------
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    context manager
        Context manager that sets the precision level
        
    Examples
    --------
    >>> with with_precision(50):
    ...     result = hp(2.0).sqrt()
    ...     print(result)
    """
    return mp.workdps(precision)


def divide(numerator: float, denominator: float, precision: int = None) -> float:
    """
    Perform high precision division if enabled, otherwise standard division.
    
    This function provides high-precision division when arbitrary precision
    arithmetic is enabled, falling back to standard division for performance
    when high precision is not required.
    
    Parameters
    ----------
    numerator : float
        Numerator value
    denominator : float  
        Denominator value
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    float
        Result of division with appropriate precision
        
    Notes
    -----
    The precision level is controlled by the USE_ARBITRARY_PRECISION
    configuration flag. When disabled, standard floating-point division
    is used for better performance.
    """
    if not USE_ARBITRARY_PRECISION:
        return numerator / denominator
        
    with mp.workdps(precision):
        mp_num = mp.mpf(numerator)
        mp_den = mp.mpf(denominator)
        result = mp_num / mp_den
        return float(result)

def sqrt(value: float, precision: int = None) -> float:
    """
    Compute square root with high precision if enabled.
    
    This function provides high-precision square root computation when arbitrary
    precision arithmetic is enabled, falling back to NumPy's sqrt function for
    performance when high precision is not required.
    
    Parameters
    ----------
    value : float
        Value to take square root of
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    float
        Square root with appropriate precision
        
    Notes
    -----
    The precision level is controlled by the USE_ARBITRARY_PRECISION
    configuration flag. When disabled, NumPy's sqrt function is used
    for better performance.
    """
    if not USE_ARBITRARY_PRECISION:
        return np.sqrt(value)

    with mp.workdps(precision):
        mp_val = mp.mpf(value)
        result = mp.sqrt(mp_val)
        return float(result)


def power(base: float, exponent: float, precision: int = None) -> float:
    """
    Compute power with high precision if enabled.
    
    This function provides high-precision power computation when arbitrary
    precision arithmetic is enabled, falling back to standard Python power
    operation for performance when high precision is not required.
    
    Parameters
    ----------
    base : float
        Base value
    exponent : float
        Exponent value
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    float
        Result with appropriate precision
        
    Notes
    -----
    The precision level is controlled by the USE_ARBITRARY_PRECISION
    configuration flag. When disabled, standard Python power operation
    is used for better performance.
    """
    if not USE_ARBITRARY_PRECISION:
        return base ** exponent

    with mp.workdps(precision):
        mp_base = mp.mpf(base)
        mp_exp = mp.mpf(exponent)
        result = mp_base ** mp_exp
        return float(result)

def find_root(func: Callable, x0: Union[float, list], precision: int = None) -> float:
    """
    Find root with high precision using mpmath.
    
    This function provides high-precision root finding using mpmath's
    findroot function. It supports both single initial guesses and
    bracketing intervals for robust root finding.
    
    Parameters
    ----------
    func : callable
        Function to find root of. Must accept a single float argument
        and return a float.
    x0 : float or list
        Initial guess or bracket [a, b] for the root
    precision : int, optional
        Number of decimal places. If None, uses 100.
        
    Returns
    -------
    float
        Root with high precision
        
    Notes
    -----
    This function always uses mpmath for root finding, regardless of the
    USE_ARBITRARY_PRECISION setting, as it provides more robust algorithms
    than standard root finding methods.
    """
    with mp.workdps(precision):
        root = mp.findroot(func, x0)
        return float(root)
