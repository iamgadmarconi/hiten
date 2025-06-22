from typing import Callable, Union

import mpmath as mp
import numpy as np

from hiten.utils.config import USE_ARBITRARY_PRECISION


class Number:
    """
    A number class that supports high-precision arithmetic operations.
    
    This class wraps numeric values and provides operator overloading for
    natural mathematical syntax while maintaining high precision when enabled.
    """
    
    def __init__(self, value: Union[float, int, str, 'Number'], precision: int = None):
        """
        Initialize a high precision number.
        
        Parameters
        ----------
        value : float, int, str, or Number
            The numeric value to wrap
        precision : int, optional
            Number of decimal places. If None, uses 100.
        """
        self.precision = precision if precision is not None else 100
        
        if isinstance(value, Number):
            self.value = value.value
            self.precision = max(self.precision, value.precision)
        elif USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                self.value = mp.mpf(value)
        else:
            self.value = float(value)
    
    def _ensure_precision_number(self, other) -> 'Number':
        """Convert other operand to Number if needed."""
        if not isinstance(other, Number):
            return Number(other, self.precision)
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
            # Standard precision fallback
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
        
        return Number(result_value, max_precision)
    
    # Arithmetic operators
    def __add__(self, other):
        return self._binary_operation(other, 'add')
    
    def __radd__(self, other):
        return Number(other, self.precision).__add__(self)
    
    def __sub__(self, other):
        return self._binary_operation(other, 'sub')
    
    def __rsub__(self, other):
        return Number(other, self.precision).__sub__(self)
    
    def __mul__(self, other):
        return self._binary_operation(other, 'mul')
    
    def __rmul__(self, other):
        return Number(other, self.precision).__mul__(self)
    
    def __truediv__(self, other):
        return self._binary_operation(other, 'truediv')
    
    def __rtruediv__(self, other):
        return Number(other, self.precision).__truediv__(self)
    
    def __pow__(self, other):
        return self._binary_operation(other, 'pow')
    
    def __rpow__(self, other):
        return Number(other, self.precision).__pow__(self)
    
    def __mod__(self, other):
        return self._binary_operation(other, 'mod')
    
    def __rmod__(self, other):
        return Number(other, self.precision).__mod__(self)
    
    # Unary operators
    def __neg__(self):
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = -self.value
        else:
            result_value = -float(self.value)
        return Number(result_value, self.precision)
    
    def __abs__(self):
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = abs(self.value)
        else:
            result_value = abs(float(self.value))
        return Number(result_value, self.precision)
    
    # Comparison operators
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
    
    # Mathematical functions
    def sqrt(self):
        """Compute square root with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.sqrt(self.value)
        else:
            result_value = np.sqrt(float(self.value))
        return Number(result_value, self.precision)
    
    def sin(self):
        """Compute sine with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.sin(self.value)
        else:
            result_value = np.sin(float(self.value))
        return Number(result_value, self.precision)
    
    def cos(self):
        """Compute cosine with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.cos(self.value)
        else:
            result_value = np.cos(float(self.value))
        return Number(result_value, self.precision)
    
    def exp(self):
        """Compute exponential with high precision."""
        if USE_ARBITRARY_PRECISION:
            with mp.workdps(self.precision):
                result_value = mp.exp(self.value)
        else:
            result_value = np.exp(float(self.value))
        return Number(result_value, self.precision)
    
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
        return Number(result_value, self.precision)
    
    # Conversion methods
    def __float__(self):
        """Convert to standard Python float."""
        return float(self.value)
    
    def __int__(self):
        """Convert to standard Python int."""
        return int(float(self.value))
    
    def __str__(self):
        """String representation."""
        return str(float(self.value))
    
    def __repr__(self):
        """Detailed string representation."""
        return f"Number({float(self.value)}, precision={self.precision})"


# Factory function for convenience
def hp(value: Union[float, int, str], precision: int = None) -> Number:
    """
    Create a Number instance.
    
    Convenience factory function for creating high precision numbers.
    
    Parameters
    ----------
    value : float, int, or str
        The numeric value
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
        
    Returns
    -------
    Number
        High precision number instance
        
    Examples
    --------
    >>> a = hp(2.5)
    >>> b = hp(3.0)
    >>> result = (a ** b) / hp(7.0)
    """
    return Number(value, precision)


def with_precision(precision: int = None):
    """
    Context manager for setting mpmath precision.
    
    Parameters
    ----------
    precision : int, optional
        Number of decimal places. If None, uses MPMATH_DPS from config.
    """
    return mp.workdps(precision)


def divide(numerator: float, denominator: float, precision: int = None) -> float:
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
        
    with mp.workdps(precision):
        mp_num = mp.mpf(numerator)
        mp_den = mp.mpf(denominator)
        result = mp_num / mp_den
        return float(result)

def sqrt(value: float, precision: int = None) -> float:
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

    with mp.workdps(precision):
        mp_val = mp.mpf(value)
        result = mp.sqrt(mp_val)
        return float(result)


def power(base: float, exponent: float, precision: int = None) -> float:
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

    with mp.workdps(precision):
        mp_base = mp.mpf(base)
        mp_exp = mp.mpf(exponent)
        result = mp_base ** mp_exp
        return float(result)

def find_root(func: Callable, x0: Union[float, list], precision: int = None) -> float:
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
    with mp.workdps(precision):
        root = mp.findroot(func, x0)
        return float(root)
