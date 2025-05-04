import numpy as np
import symengine as se
from collections import defaultdict
from typing import Dict, Tuple, List, Union, Iterator
import numba as nb


# ------ Numba-accelerated helper functions ------

def _dict_to_arrays(coeffs: Dict[Tuple[int, ...], np.complex128], n_vars: int):
    """
    Convert a coefficient dictionary to arrays for Numba processing.
    
    Args:
        coeffs: Dictionary mapping exponent tuples to coefficients
        n_vars: Number of variables
        
    Returns:
        Tuple of (keys_array, coeffs_array)
        - keys_array: 2D array of shape (n_terms, n_vars) with exponents
        - coeffs_array: 1D array of complex coefficients
    """
    n_terms = len(coeffs)
    
    if n_terms == 0:
        # Return empty arrays with correct shapes
        return np.zeros((0, n_vars), dtype=np.int32), np.zeros(0, dtype=np.complex128)
    
    # Initialize arrays
    keys_array = np.zeros((n_terms, n_vars), dtype=np.int32)
    coeffs_array = np.zeros(n_terms, dtype=np.complex128)
    
    # Fill arrays
    for i, (exps, coeff) in enumerate(coeffs.items()):
        keys_array[i, :] = exps
        coeffs_array[i] = coeff
    
    return keys_array, coeffs_array


def _arrays_to_dict(keys_array: np.ndarray, coeffs_array: np.ndarray, tol: float = 1e-16):
    """
    Convert arrays back to a coefficient dictionary, handling potential duplicate keys.
    
    Args:
        keys_array: 2D array of exponents (n_terms, n_vars)
        coeffs_array: 1D array of coefficients
        tol: Tolerance for filtering near-zero coefficients
        
    Returns:
        Dictionary mapping exponent tuples to coefficients
    """
    result = {}
    
    for i in range(keys_array.shape[0]):
        key = tuple(keys_array[i, :])
        val = coeffs_array[i]
        
        if abs(val) >= tol:  # Filter near-zero coefficients
            if key in result:
                result[key] += val
            else:
                result[key] = val
    
    # Final pass to remove any small terms that might result from aggregation
    return {k: v for k, v in result.items() if abs(v) >= tol}


# ------ Numba JIT-compiled functions ------

@nb.njit(fastmath=True, cache=True)
def _numba_mul(keys1: np.ndarray, coeffs1: np.ndarray, keys2: np.ndarray, coeffs2: np.ndarray):
    """
    Multiply two polynomials represented as arrays using Numba acceleration.
    
    Args:
        keys1: 2D array of exponents for first polynomial (n_terms1, n_vars)
        coeffs1: 1D array of coefficients for first polynomial
        keys2: 2D array of exponents for second polynomial (n_terms2, n_vars)
        coeffs2: 1D array of coefficients for second polynomial
        
    Returns:
        Tuple of (result_keys, result_coeffs)
    """
    # Sizes of inputs
    n_terms1, n_vars = keys1.shape
    n_terms2 = keys2.shape[0]
    max_result_terms = n_terms1 * n_terms2
    
    # Pre-allocate result arrays (maximum possible size)
    result_keys = np.zeros((max_result_terms, n_vars), dtype=np.int32)
    result_coeffs = np.zeros(max_result_terms, dtype=np.complex128)
    
    # Multiply terms
    term_idx = 0
    for i in range(n_terms1):
        for j in range(n_terms2):
            # Multiply coefficients
            result_coeffs[term_idx] = coeffs1[i] * coeffs2[j]
            
            # Add exponents
            for k in range(n_vars):
                result_keys[term_idx, k] = keys1[i, k] + keys2[j, k]
                
            term_idx += 1
    
    return result_keys[:term_idx], result_coeffs[:term_idx]


@nb.njit(fastmath=True, cache=True)
def _numba_derivative(keys: np.ndarray, coeffs: np.ndarray, var_index: int):
    """
    Compute the derivative with respect to a variable using Numba acceleration.
    
    Args:
        keys: 2D array of exponents (n_terms, n_vars)
        coeffs: 1D array of coefficients
        var_index: Index of the variable to differentiate by
        
    Returns:
        Tuple of (result_keys, result_coeffs)
    """
    n_terms, n_vars = keys.shape
    
    # Pre-allocate result arrays (maximum size = n_terms)
    result_keys = np.zeros((n_terms, n_vars), dtype=np.int32)
    result_coeffs = np.zeros(n_terms, dtype=np.complex128)
    
    term_idx = 0
    for i in range(n_terms):
        # Only process terms containing the variable
        if keys[i, var_index] > 0:
            # Copy all exponents
            for k in range(n_vars):
                result_keys[term_idx, k] = keys[i, k]
            
            # Decrement exponent for the variable
            result_keys[term_idx, var_index] -= 1
            
            # Compute new coefficient
            result_coeffs[term_idx] = coeffs[i] * keys[i, var_index]
            
            term_idx += 1
    
    return result_keys[:term_idx], result_coeffs[:term_idx]


@nb.njit(fastmath=True, cache=True)
def _numba_poisson(keys_F: np.ndarray, coeffs_F: np.ndarray, keys_G: np.ndarray, 
                  coeffs_G: np.ndarray, n_dof: int):
    """
    Compute the Poisson bracket {F, G} using Numba acceleration.
    
    Args:
        keys_F: 2D array of exponents for F (n_terms_F, n_vars)
        coeffs_F: 1D array of coefficients for F
        keys_G: 2D array of exponents for G (n_terms_G, n_vars)
        coeffs_G: 1D array of coefficients for G
        n_dof: Number of degrees of freedom (n_vars/2)
        
    Returns:
        Tuple of (result_keys, result_coeffs)
    """
    n_vars = keys_F.shape[1]
    
    # Arrays to store intermediate results from each term in the sum
    all_keys = []
    all_coeffs = []
    
    # For each pair (q_i, p_i)
    for i in range(n_dof):
        q_idx = i
        p_idx = i + n_dof
        
        # Compute dF/dq_i
        dF_dq_keys, dF_dq_coeffs = _numba_derivative(keys_F, coeffs_F, q_idx)
        
        # Compute dF/dp_i
        dF_dp_keys, dF_dp_coeffs = _numba_derivative(keys_F, coeffs_F, p_idx)
        
        # Compute dG/dq_i
        dG_dq_keys, dG_dq_coeffs = _numba_derivative(keys_G, coeffs_G, q_idx)
        
        # Compute dG/dp_i
        dG_dp_keys, dG_dp_coeffs = _numba_derivative(keys_G, coeffs_G, p_idx)
        
        # Compute dF/dq_i * dG/dp_i
        term1_keys, term1_coeffs = _numba_mul(dF_dq_keys, dF_dq_coeffs, dG_dp_keys, dG_dp_coeffs)
        all_keys.append(term1_keys)
        all_coeffs.append(term1_coeffs)
        
        # Compute dF/dp_i * dG/dq_i (with negative sign)
        term2_keys, term2_coeffs = _numba_mul(dF_dp_keys, dF_dp_coeffs, dG_dq_keys, dG_dq_coeffs)
        all_keys.append(term2_keys)
        # Negate the coefficients for the second term
        all_coeffs.append(-term2_coeffs)
    
    # Merge all terms
    # Calculate total number of terms with a loop instead of a generator expression
    total_terms = 0
    for i in range(len(all_coeffs)):
        total_terms += len(all_coeffs[i])
    
    if total_terms == 0:
        return np.zeros((0, n_vars), dtype=np.int32), np.zeros(0, dtype=np.complex128)
    
    # Allocate arrays for the merged result
    merged_keys = np.zeros((total_terms, n_vars), dtype=np.int32)
    merged_coeffs = np.zeros(total_terms, dtype=np.complex128)
    
    # Fill the arrays
    idx = 0
    for i in range(len(all_keys)):
        n_terms = len(all_coeffs[i])
        if n_terms > 0:
            merged_keys[idx:idx+n_terms] = all_keys[i]
            merged_coeffs[idx:idx+n_terms] = all_coeffs[i]
            idx += n_terms
    
    return merged_keys, merged_coeffs


class DictPolynomial:
    """
    Polynomial stored as a dictionary of coefficients indexed by exponent tuples.
    Designed for faster numerical computation compared to symbolic SymEngine expressions.
    
    Attributes:
        variables: List of SymEngine symbols representing the variables
        coeffs: Dictionary mapping exponent tuples to coefficients
        n_vars: Number of variables
        n_dof: Number of degrees of freedom (n_vars/2 for canonical systems)
        _var_to_index: Dictionary mapping variables to their indices for quick lookups
    """
    
    def __init__(self, variables: List[se.Symbol], coeffs: Dict[Tuple[int, ...], Union[complex, float, int]] = None):
        """
        Initialize a polynomial with variables and coefficient dictionary.
        
        Args:
            variables: List of SymEngine symbols representing the variables
            coeffs: Dictionary mapping exponent tuples to coefficients
        """
        self.variables = list(variables)
        self.n_vars = len(variables)
        self.n_dof = self.n_vars // 2  # For canonical systems
        
        # Create variable to index mapping for quick lookups
        self._var_to_index = {var: idx for idx, var in enumerate(variables)}
        
        # Initialize coefficients, filtering out near-zero values
        self.coeffs = {}
        if coeffs:
            for key, value in coeffs.items():
                if len(key) != self.n_vars:
                    raise ValueError(f"Exponent tuple length {len(key)} must match number of variables {self.n_vars}")
                if abs(value) >= 1e-16:  # Filter out very small values
                    self.coeffs[key] = np.complex128(value)
    
    def __len__(self) -> int:
        """Return the number of terms in the polynomial."""
        return len(self.coeffs)
    
    def __str__(self) -> str:
        """Return a readable string representation of the polynomial."""
        if not self.coeffs:
            return "0"
        
        terms = []
        for exponents, coeff in sorted(self.coeffs.items(), 
                                       key=lambda x: sum(x[0]), reverse=True):
            # Format the coefficient
            if abs(coeff.imag) < 1e-10:  # Real coefficient
                coeff_str = f"{coeff.real:.10g}"
                # Remove trailing zeros and decimal point if integer
                if '.' in coeff_str:
                    coeff_str = coeff_str.rstrip('0').rstrip('.')
            else:  # Complex coefficient
                coeff_str = f"({coeff.real:.6g}{coeff.imag:+.6g}j)"
            
            # Build the term
            if all(exp == 0 for exp in exponents):  # Constant term
                terms.append(coeff_str)
            else:
                var_parts = []
                for i, exp in enumerate(exponents):
                    if exp > 0:
                        var_name = str(self.variables[i])
                        if exp == 1:
                            var_parts.append(var_name)
                        else:
                            var_parts.append(f"{var_name}**{exp}")
                
                # Join variable parts with multiplication symbol
                var_str = "*".join(var_parts)
                
                # Add coefficient to term
                if coeff_str == "1" and var_str:
                    terms.append(var_str)
                elif coeff_str == "-1" and var_str:
                    terms.append(f"-{var_str}")
                else:
                    terms.append(f"{coeff_str}*{var_str}" if var_str else coeff_str)
        
        return " + ".join(terms).replace(" + -", " - ")
    
    def __repr__(self) -> str:
        """Return a string representation for debugging."""
        return f"DictPolynomial({len(self.variables)} vars, {len(self.coeffs)} terms)"
    
    def copy(self) -> 'DictPolynomial':
        """Return a deep copy of this polynomial."""
        return DictPolynomial(self.variables, dict(self.coeffs))
    
    @classmethod
    def zero(cls, variables: List[se.Symbol]) -> 'DictPolynomial':
        """Create a zero polynomial with the given variables."""
        return cls(variables, {})
    
    @classmethod
    def from_coeffs(cls, variables: List[se.Symbol], 
                    coeffs: Dict[Tuple[int, ...], Union[complex, float, int]]) -> 'DictPolynomial':
        """
        Create a polynomial from a coefficient dictionary.
        
        Args:
            variables: List of SymEngine symbols for the variables
            coeffs: Dictionary mapping exponent tuples to coefficients
        """
        return cls(variables, coeffs)
    
    def total_degree(self) -> int:
        """
        Return the maximum total degree of all terms in the polynomial.
        
        Returns:
            The maximum sum of exponents, or 0 for constants, or -1 for zero polynomial.
        """
        if not self.coeffs:
            return -1  # Zero polynomial
        
        return max((sum(exps) for exps in self.coeffs.keys()), default=0)
    
    def truncate(self, max_deg: int) -> 'DictPolynomial':
        """
        Return a new polynomial with terms of degree > max_deg removed.
        
        Args:
            max_deg: Maximum degree to keep
            
        Returns:
            A new polynomial with truncated terms
        """
        new_coeffs = {
            exps: coeff for exps, coeff in self.coeffs.items() 
            if sum(exps) <= max_deg
        }
        return DictPolynomial(self.variables, new_coeffs)
    
    def get_homogeneous_part(self, degree: int) -> 'DictPolynomial':
        """
        Return a new polynomial containing only terms of the specified degree.
        
        Args:
            degree: The degree to extract
            
        Returns:
            A new polynomial with only terms of the specified degree
        """
        new_coeffs = {
            exps: coeff for exps, coeff in self.coeffs.items() 
            if sum(exps) == degree
        }
        return DictPolynomial(self.variables, new_coeffs)
    
    def iter_terms(self) -> Iterator[Tuple[Tuple[int, ...], np.complex128]]:
        """Yield (exponent_tuple, coefficient) pairs for each term."""
        for exps, coeff in self.coeffs.items():
            yield exps, coeff
    
    def __add__(self, other) -> 'DictPolynomial':
        """
        Add two polynomials or add a scalar to a polynomial.
        
        Args:
            other: Another polynomial or a scalar
            
        Returns:
            A new polynomial representing the sum
        """
        if isinstance(other, DictPolynomial):
            if self.variables != other.variables:
                raise ValueError("Cannot add polynomials with different variables")
            
            result = defaultdict(np.complex128)
            
            # Add terms from self
            for exps, coeff in self.coeffs.items():
                result[exps] += coeff
            
            # Add terms from other
            for exps, coeff in other.coeffs.items():
                result[exps] += coeff
            
            # Filter out near-zero terms
            final_coeffs = {
                exps: coeff for exps, coeff in result.items() 
                if abs(coeff) >= 1e-16
            }
            
            return DictPolynomial(self.variables, final_coeffs)
        
        elif isinstance(other, (int, float, complex)):
            # Add a scalar - this affects only the constant term
            zero_exp = tuple(0 for _ in range(self.n_vars))
            new_coeffs = dict(self.coeffs)
            new_coeffs[zero_exp] = new_coeffs.get(zero_exp, 0) + np.complex128(other)
            
            # Remove if resulting coefficient is near zero
            if abs(new_coeffs[zero_exp]) < 1e-16:
                new_coeffs.pop(zero_exp)
                
            return DictPolynomial(self.variables, new_coeffs)
        
        return NotImplemented
    
    def __radd__(self, other) -> 'DictPolynomial':
        """Add a scalar to a polynomial (right-hand operation)."""
        return self.__add__(other)
    
    def __sub__(self, other) -> 'DictPolynomial':
        """
        Subtract another polynomial or scalar from this polynomial.
        
        Args:
            other: Another polynomial or a scalar
            
        Returns:
            A new polynomial representing the difference
        """
        if isinstance(other, DictPolynomial):
            if self.variables != other.variables:
                raise ValueError("Cannot subtract polynomials with different variables")
            
            result = defaultdict(np.complex128)
            
            # Add terms from self
            for exps, coeff in self.coeffs.items():
                result[exps] += coeff
            
            # Subtract terms from other
            for exps, coeff in other.coeffs.items():
                result[exps] -= coeff
            
            # Filter out near-zero terms
            final_coeffs = {
                exps: coeff for exps, coeff in result.items() 
                if abs(coeff) >= 1e-16
            }
            
            return DictPolynomial(self.variables, final_coeffs)
        
        elif isinstance(other, (int, float, complex)):
            # Subtract a scalar - this affects only the constant term
            zero_exp = tuple(0 for _ in range(self.n_vars))
            new_coeffs = dict(self.coeffs)
            new_coeffs[zero_exp] = new_coeffs.get(zero_exp, 0) - np.complex128(other)
            
            # Remove if resulting coefficient is near zero
            if abs(new_coeffs[zero_exp]) < 1e-16:
                new_coeffs.pop(zero_exp)
                
            return DictPolynomial(self.variables, new_coeffs)
        
        return NotImplemented
    
    def __rsub__(self, other) -> 'DictPolynomial':
        """Subtract this polynomial from a scalar (right-hand operation)."""
        if isinstance(other, (int, float, complex)):
            # other - self = -(self - other)
            return -self.__sub__(other)
        return NotImplemented
    
    def __neg__(self) -> 'DictPolynomial':
        """Return the negation of this polynomial."""
        return DictPolynomial(self.variables, {
            exps: -coeff for exps, coeff in self.coeffs.items()
        })
    
    def __mul__(self, other) -> 'DictPolynomial':
        """
        Multiply this polynomial by another polynomial or a scalar.
        
        Args:
            other: Another polynomial or a scalar
            
        Returns:
            A new polynomial representing the product
        """
        if isinstance(other, DictPolynomial):
            if self.variables != other.variables:
                raise ValueError("Cannot multiply polynomials with different variables")
            
            # Use Numba acceleration
            if not self.coeffs or not other.coeffs:
                return DictPolynomial.zero(self.variables)
                
            # Convert dictionaries to arrays
            keys1, coeffs1 = _dict_to_arrays(self.coeffs, self.n_vars)
            keys2, coeffs2 = _dict_to_arrays(other.coeffs, self.n_vars)
            
            # Perform multiplication using Numba
            result_keys, result_coeffs = _numba_mul(keys1, coeffs1, keys2, coeffs2)
            
            # Convert back to dictionary
            result_dict = _arrays_to_dict(result_keys, result_coeffs)
            
            return DictPolynomial(self.variables, result_dict)
        
        elif isinstance(other, (int, float, complex)):
            # Scalar multiplication
            other_complex = np.complex128(other)
            
            # Short-circuit if multiplying by zero
            if abs(other_complex) < 1e-16:
                return DictPolynomial.zero(self.variables)
            
            # Multiply all coefficients by the scalar
            new_coeffs = {
                exps: coeff * other_complex for exps, coeff in self.coeffs.items() 
                if abs(coeff * other_complex) >= 1e-16
            }
            
            return DictPolynomial(self.variables, new_coeffs)
        
        return NotImplemented
    
    def __rmul__(self, other) -> 'DictPolynomial':
        """Multiply a scalar by this polynomial (right-hand operation)."""
        return self.__mul__(other)
    
    def __pow__(self, exponent: int) -> 'DictPolynomial':
        """
        Raise this polynomial to an integer power.
        
        Args:
            exponent: Non-negative integer exponent
            
        Returns:
            A new polynomial representing self^exponent
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")
        
        if exponent == 0:
            # Return the constant polynomial 1
            zero_exp = tuple(0 for _ in range(self.n_vars))
            return DictPolynomial(self.variables, {zero_exp: np.complex128(1.0)})
        
        if exponent == 1:
            return self.copy()
        
        # For higher powers, use binary exponentiation for efficiency
        result = self.copy()
        temp = self.copy()
        exponent -= 1
        
        while exponent > 0:
            if exponent % 2 == 1:
                result = result * temp
            temp = temp * temp
            exponent //= 2
        
        return result

    def __eq__(self, other) -> bool:
        """Check if this polynomial is equal to another polynomial."""
        if not isinstance(other, DictPolynomial):
            return False
        return self.variables == other.variables and self.coeffs == other.coeffs

    def derivative(self, var_index: int) -> 'DictPolynomial':
        """
        Compute the partial derivative with respect to the variable at given index.
        
        Args:
            var_index: Index of the variable to differentiate by
            
        Returns:
            A new polynomial representing the derivative
        """
        if var_index < 0 or var_index >= self.n_vars:
            raise ValueError(f"Variable index {var_index} out of range [0, {self.n_vars-1}]")
        
        # Use Numba acceleration
        if not self.coeffs:
            return DictPolynomial.zero(self.variables)
            
        # Convert dictionary to arrays
        keys, coeffs = _dict_to_arrays(self.coeffs, self.n_vars)
        
        # Perform differentiation using Numba
        result_keys, result_coeffs = _numba_derivative(keys, coeffs, var_index)
        
        # Convert back to dictionary
        result_dict = _arrays_to_dict(result_keys, result_coeffs)
        
        return DictPolynomial(self.variables, result_dict)
    
    def gradient(self) -> List['DictPolynomial']:
        """
        Compute the gradient of the polynomial.
        
        Returns:
            List of polynomials representing partial derivatives with respect to each variable
        """
        return [self.derivative(i) for i in range(self.n_vars)]
    
    def poisson(self, other: 'DictPolynomial') -> 'DictPolynomial':
        """
        Compute the Poisson bracket {self, other}.
        
        The Poisson bracket is defined as:
        {F, G} = sum_i (dF/dqi * dG/dpi - dF/dpi * dG/dqi)
        
        Args:
            other: Another polynomial
            
        Returns:
            A new polynomial representing {self, other}
        """
        if self.variables != other.variables:
            raise ValueError("Polynomials must have the same variables for Poisson bracket")
        
        # Use Numba acceleration
        if not self.coeffs or not other.coeffs:
            return DictPolynomial.zero(self.variables)
            
        # Convert dictionaries to arrays
        keys_F, coeffs_F = _dict_to_arrays(self.coeffs, self.n_vars)
        keys_G, coeffs_G = _dict_to_arrays(other.coeffs, self.n_vars)
        
        # Perform Poisson bracket calculation using Numba
        result_keys, result_coeffs = _numba_poisson(keys_F, coeffs_F, keys_G, coeffs_G, self.n_dof)
        
        # Convert back to dictionary
        result_dict = _arrays_to_dict(result_keys, result_coeffs)
        
        return DictPolynomial(self.variables, result_dict)
    
    def to_symengine(self) -> se.Basic:
        """
        Convert this polynomial to a SymEngine expression.
        
        Returns:
            A SymEngine expression equivalent to this polynomial
        """
        if not self.coeffs:
            return se.Integer(0)
        
        terms = []
        
        for exps, coeff in self.coeffs.items():
            # Handle the coefficient
            if abs(coeff.imag) < 1e-14:  # Real coefficient
                term = se.sympify(float(coeff.real))
            else:  # Complex coefficient
                # SymEngine doesn't directly support complex numbers, use I
                term = se.sympify(float(coeff.real)) + se.sympify(float(coeff.imag)) * se.I
            
            # Multiply by variables raised to powers
            for i, exp in enumerate(exps):
                if exp > 0:
                    term *= self.variables[i] ** exp
            
            terms.append(term)
        
        # Sum all terms
        result = se.Add(*terms)
        return result
    
    @classmethod
    def from_symengine(cls, variables: List[se.Symbol], expr: se.Basic) -> 'DictPolynomial':
        """
        Create a polynomial from a SymEngine expression.
        
        Args:
            variables: List of SymEngine symbols
            expr: SymEngine expression to convert
            
        Returns:
            A new DictPolynomial representing the expression
        """
        # First clean numerical artifacts and expand the expression
        from .core import _clean_numerical_artifacts  # Import here to avoid circular imports
        import sympy as sp
        
        # Clean numerical artifacts
        sp_expr = sp.sympify(expr)
        cleaned_expr = _clean_numerical_artifacts(sp_expr)
        expr = se.sympify(cleaned_expr)
        
        # Expand the expression
        expanded_expr = expr.expand()
        
        # Create variable to index mapping
        var_to_idx = {var: i for i, var in enumerate(variables)}
        n_vars = len(variables)
        
        # Initialize the coefficient dictionary
        coeffs = {}
        
        # Helper function to process one term
        def process_term(term):
            # Handle constant term
            if term.is_Number:
                zero_exp = tuple([0] * n_vars)
                return zero_exp, complex(float(term))
            
            # Extract coefficient and factors
            coeff = se.Integer(1)
            non_numeric_factors = []
            
            if term.is_Mul:
                for factor in term.args:
                    if factor.is_Number:
                        coeff *= factor
                    else:
                        non_numeric_factors.append(factor)
            else:
                non_numeric_factors = [term]
            
            # Initialize exponents
            exps = [0] * n_vars
            
            # Process each non-numeric factor
            for factor in non_numeric_factors:
                # Handle powers
                if factor.is_Pow:
                    base, exp = factor.args
                    if base in var_to_idx and exp.is_Integer:
                        idx = var_to_idx[base]
                        exps[idx] += int(exp)
                # Handle variables without explicit power
                elif factor in var_to_idx:
                    idx = var_to_idx[factor]
                    exps[idx] += 1
                # Log warning for unrecognized factors
                else:
                    print(f"Warning: Unrecognized factor {factor} in term {term}")
            
            return tuple(exps), complex(float(coeff))
        
        # Process each term in the expanded expression
        if expanded_expr.is_Add:
            for term in expanded_expr.args:
                exps, coeff_val = process_term(term)
                if abs(coeff_val) >= 1e-16:  # Filter near-zero terms
                    coeffs[exps] = coeffs.get(exps, 0) + np.complex128(coeff_val)
        else:
            # Single term
            exps, coeff_val = process_term(expanded_expr)
            if abs(coeff_val) >= 1e-16:  # Filter near-zero terms
                coeffs[exps] = np.complex128(coeff_val)
        
        # Filter out any remaining near-zero terms (that may result from aggregation)
        filtered_coeffs = {
            exps: coeff for exps, coeff in coeffs.items() 
            if abs(coeff) >= 1e-16
        }
        
        return cls(variables, filtered_coeffs)



