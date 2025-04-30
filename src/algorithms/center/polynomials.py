import symengine as se
import numpy as np # For isclose later if needed, though less critical with symbolic
import math      # Add math import for factorial and isclose
from collections import defaultdict # For efficient term collection in equality check
from functools import lru_cache


class Polynomial:
    """
    An improved Polynomial class using symengine as the backend for symbolic manipulation.
    
    Provides methods needed for Hamiltonian mechanics algorithms, optimized to avoid 
    unnecessary expansions where possible.
    """
    _symbol_cache = {} # Cache symbols for efficiency

    def __init__(self, expression, n_vars=6, variables=None):
        """
        Initializes the Polynomial using a symengine expression.

        Parameters
        ----------
        expression : symengine.Expr or str or number
            The symengine expression, a string to parse, or a number.
        n_vars : int
            Number of variables (must be even for PB).
        variables : list, optional
            A list of symengine symbols in canonical order [q1, p1, q2, p2,...].
            If None, they will be generated automatically as x0, x1,...
            Note that methods like poisson_bracket assume this canonical ordering.
        
        Raises
        ------
        ValueError
            If n_vars is not a positive even integer or if variables length doesn't match n_vars.
        TypeError
            If expression is of unsupported type.
        """
        if n_vars <= 0 or n_vars % 2!= 0:
            raise ValueError("n_vars must be a positive even integer.")
        self.n_vars = n_vars

        # Generate or use provided variables
        if variables:
            if len(variables)!= n_vars:
                raise ValueError(f"Provided variables list length ({len(variables)})!= n_vars ({n_vars})")
            self.variables = variables
            # Ensure cache consistency if custom variables are provided
            var_key = tuple(str(v) for v in variables)
            if var_key not in Polynomial._symbol_cache:
                Polynomial._symbol_cache[var_key] = variables
        else:
            # Generate standard variable names and use them as the cache key
            standard_names = tuple(f'x{i}' for i in range(n_vars))
            var_key = standard_names
            if var_key not in Polynomial._symbol_cache:
                Polynomial._symbol_cache[var_key] = se.symbols(list(standard_names))
            self.variables = Polynomial._symbol_cache[var_key]

        # Store the internal symengine expression
        if isinstance(expression, se.Expr):
            self.expr = expression
        elif isinstance(expression, (int, float, complex)):
            self.expr = se.sympify(expression)
        elif isinstance(expression, str):
            # String initialization using substitution.
            # NOTE: Assumes variable names in the string are 'x0', 'x1', etc.
            # Consider creating expressions programmatically for better robustness/efficiency.
            try:
                parsed_expr = se.sympify(expression)
            except (SyntaxError, TypeError, RuntimeError) as e:
                raise ValueError(f"Could not parse expression string: '{expression}'. Error: {e}")

            substitutions = {}
            # Identify symbols like 'x0', 'x1' created by sympify
            # and map them to the canonical symbols stored in self.variables
            string_symbols = {str(s): s for s in parsed_expr.free_symbols if isinstance(s, se.Symbol)}

            for i, canonical_var in enumerate(self.variables):
                var_name = f'x{i}'
                if var_name in string_symbols and string_symbols[var_name]!= canonical_var:
                    substitutions[string_symbols[var_name]] = canonical_var

            if substitutions:
                self.expr = parsed_expr.subs(substitutions)
            else:
                self.expr = parsed_expr
        else:
            raise TypeError(f"Unsupported expression type for initialization: {type(expression)}")

    @classmethod
    def zero(cls, n_vars=6, variables=None):
        """
        Creates a zero polynomial.
        
        Parameters
        ----------
        n_vars : int, optional
            Number of variables. Default is 6.
        variables : list, optional
            A list of symengine symbols in canonical order. Default is None.
            
        Returns
        -------
        Polynomial
            A polynomial with value 0.
        """
        return cls(0, n_vars=n_vars, variables=variables)

    @classmethod
    def one(cls, n_vars=6, variables=None):
        """
        Creates a one polynomial.
        
        Parameters
        ----------
        n_vars : int, optional
            Number of variables. Default is 6.
        variables : list, optional
            A list of symengine symbols in canonical order. Default is None.
            
        Returns
        -------
        Polynomial
            A polynomial with value 1.
        """
        return cls(1, n_vars=n_vars, variables=variables)

    def __repr__(self):
        # Use standard variable names in repr for consistency if possible
        std_vars_key = tuple(f'x{i}' for i in range(self.n_vars))
        std_vars = Polynomial._symbol_cache.get(std_vars_key, self.variables)
        # Temporarily substitute canonical vars with standard names for repr if different
        sub_dict_repr = {can_var: std_var for can_var, std_var in zip(self.variables, std_vars) if can_var!= std_var}
        repr_expr = self.expr.subs(sub_dict_repr) if sub_dict_repr else self.expr
        return f"Polynomial('{str(repr_expr)}', n_vars={self.n_vars})"

    def __str__(self):
        return str(self.expr)

    # --- Arithmetic Operations (unchanged, rely on symengine) ---
    def __add__(self, other):
        """Addition: Handles Polynomial + Polynomial or Polynomial + scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars!= other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            return Polynomial(self.expr + other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(self.expr + other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction: Handles Polynomial - Polynomial or Polynomial - scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars!= other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            return Polynomial(self.expr - other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(self.expr - other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(other_expr - self.expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiplication: Handles Polynomial * Polynomial or Polynomial * scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars!= other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            return Polynomial(self.expr * other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(self.expr * other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        """Negation: -Polynomial"""
        return Polynomial(-self.expr, self.n_vars, self.variables)

    @lru_cache(maxsize=None)
    def _d(expr, var):
        return se.diff(expr, var)

    # --- Core Symbolic Operations ---

    def differentiate(self, var_index):
        """
        Differentiates the polynomial with respect to a given variable index.

        Parameters
        ----------
        var_index : int
            The 0-based index of the variable in self.variables.
            Assumes canonical ordering [q1, p1,...] if using indices.
        
        Returns
        -------
        Polynomial
            The derivative of the polynomial with respect to the specified variable.
            
        Raises
        ------
        ValueError
            If var_index is out of range.
        """
        if not (0 <= var_index < self.n_vars):
            raise ValueError(f"var_index must be between 0 and {self.n_vars-1}")
        target_var = self.variables[var_index]
        derivative_expr = se.diff(self.expr, target_var)
        return Polynomial(derivative_expr, self.n_vars, self.variables)

    def poisson_bracket(self, other):
        """
        Computes the Poisson bracket {self, other}.
        
        Parameters
        ----------
        other : Polynomial or scalar
            The second argument of the Poisson bracket.
            
        Returns
        -------
        Polynomial
            The result of the Poisson bracket operation.
            
        Raises
        ------
        TypeError
            If other is not a Polynomial or numeric scalar.
        ValueError
            If polynomials have different number of variables.
        IndexError
            If there's a variable indexing error during calculation.
            
        Notes
        -----
        Assumes canonical variables ordered [q1, p1, q2, p2,...] in self.variables.
        """
        if not isinstance(other, Polynomial):
            if isinstance(other, (int, float, complex, str)):
                other_expr = se.sympify(other)
                if other_expr.is_Number:
                    return Polynomial.zero(self.n_vars, self.variables) # Bracket with constant is 0
            raise TypeError("Poisson bracket requires another Polynomial or a numeric scalar.")

        if self.n_vars!= other.n_vars:
            raise ValueError("Polynomials must have the same number of variables for Poisson bracket.")
        # n_vars must be even check done in __init__

        num_dof = self.n_vars // 2
        result_expr = se.sympify(0) # Start with zero expression

        for i in range(num_dof):
            qi_index = 2 * i
            pi_index = 2 * i + 1
            # Check if indices are valid (safety, though n_vars check should cover this)
            if pi_index >= self.n_vars:
                raise IndexError("Variable indexing error in Poisson bracket calculation.")
            qi = self.variables[qi_index]
            pi = self.variables[pi_index]

            # Compute partial derivatives using symengine.diff
            d_self_dqi = se.diff(self.expr, qi)
            d_other_dpi = se.diff(other.expr, pi)
            term1 = d_self_dqi * d_other_dpi

            d_self_dpi = se.diff(self.expr, pi)
            d_other_dqi = se.diff(other.expr, qi)
            term2 = d_self_dpi * d_other_dqi

            result_expr = result_expr + term1 - term2

        return Polynomial(result_expr, self.n_vars, self.variables)

    def substitute(self, var_map):
        """
        Substitutes variables in the expression.

        Parameters
        ----------
        var_map : dict
            A dictionary mapping symengine symbols, variable indices, or variable 
            names (str) to their substitution values (numbers, strings, symengine 
            expressions, or other Polynomials).
            
        Returns
        -------
        Polynomial
            A new polynomial with the substitutions applied.
            
        Raises
        ------
        ValueError
            If a symbol or index is not found in the polynomial variables.
        TypeError
            If key or value types in var_map are invalid.
        """
        sub_dict = {}
        for k, v in var_map.items():
            key_symbol = None
            if isinstance(k, int): # Allow substitution by index
                if 0 <= k < self.n_vars:
                    key_symbol = self.variables[k]
                else:
                    raise ValueError(f"Invalid variable index {k} for substitution.")
            elif isinstance(k, se.Symbol):
                # Ensure the symbol is one of the polynomial's variables
                if k not in self.variables:
                    raise ValueError(f"Symbol {k} not found in polynomial variables {self.variables}")
                key_symbol = k
            elif isinstance(k, str):
                # Find symbol by name
                found = False
                for sym in self.variables:
                    if str(sym) == k:
                        key_symbol = sym
                        found = True
                        break
                if not found:
                    raise ValueError(f"Symbol string '{k}' not found in polynomial variables.")
            else:
                raise TypeError(f"Invalid key type in var_map: {type(k)}")

            # Process value
            value_expr = None
            if isinstance(v, Polynomial):
                if v.n_vars!= self.n_vars:
                    # We could potentially allow substitution with polynomials of different n_vars,
                    # but it might lead to unexpected variable sets in the result.
                    # Forcing same n_vars seems safer for now.
                    print(f"Warning: Substituting with Polynomial of different n_vars ({v.n_vars}). Result retains original n_vars ({self.n_vars}).")
                value_expr = v.expr
            elif isinstance(v, (int, float, complex, str, se.Expr)):
                value_expr = se.sympify(v) # Ensure value is a symengine expression
            else:
                raise TypeError(f"Invalid value type in var_map: {type(v)}")

            sub_dict[key_symbol] = value_expr

        substituted_expr = self.expr.subs(sub_dict)
        # Return a new Polynomial, maintaining the original n_vars and variables
        # This assumes the substitution doesn't fundamentally change the variable space context.
        return Polynomial(substituted_expr, self.n_vars, self.variables)

    # --- Equality Check (Improved) ---

    def equals(self, other, tolerance=1e-12):
        """
        Checks for mathematical equality between two polynomials by comparing
        their terms and coefficients without explicit expansion.

        Parameters
        ----------
        other : Polynomial
            The polynomial to compare against.
        tolerance : float, optional
            The absolute and relative tolerance for comparing floating-point coefficients.
            Default is 1e-12.
            
        Returns
        -------
        bool
            True if the polynomials are mathematically equal within the given tolerance, 
            False otherwise.
        """
        if not isinstance(other, Polynomial):
            return False
        if self.n_vars != other.n_vars:
            return False

        # Special case for zero polynomial
        if self.expr == se.sympify(0) and other.expr == se.sympify(0):
            return True

        # Special handling for simple cases: numbers or single variables
        if self.expr.is_Number and other.expr.is_Number:
            try:
                return math.isclose(float(self.expr), float(other.expr), 
                                  rel_tol=tolerance, abs_tol=tolerance)
            except (TypeError, ValueError):
                return self.expr == other.expr

        # Use symbolic expansion to handle cases like (x+y)^2
        try:
            expanded_self = se.expand(self.expr)
            expanded_other = se.expand(other.expr)
            if expanded_self == expanded_other:
                return True
        except Exception:
            pass  # Continue with other methods if expansion fails

        # Numerical comparison approach
        # Test with sample points - fast and reliable for polynomials
        import random
        test_points = 10
        try:
            for _ in range(test_points):
                point = [random.uniform(-1, 1) for _ in range(self.n_vars)]
                val1 = self.evaluate(point)
                val2 = other.evaluate(point)
                if not math.isclose(val1, val2, rel_tol=tolerance, abs_tol=tolerance):
                    return False
            return True
        except Exception as e:
            print(f"Warning: Error during numerical comparison in equals method: {e}. Continuing with term comparison.")

        # Compare coefficients using get_terms for expanded expressions
        try:
            # Create new Polynomial objects with expanded expressions
            expanded_self_poly = Polynomial(se.expand(self.expr), self.n_vars, self.variables)
            expanded_other_poly = Polynomial(se.expand(other.expr), self.n_vars, self.variables)
            
            terms1 = dict(expanded_self_poly.get_terms())
            terms2 = dict(expanded_other_poly.get_terms())
            
            # Check if the sets of monomials are the same
            if set(terms1.keys()) != set(terms2.keys()):
                return False
                
            # Compare coefficients with tolerance
            for exp_tuple in terms1:
                coeff1 = terms1[exp_tuple]
                coeff2 = terms2[exp_tuple]
                if abs(coeff1 - coeff2) > tolerance:
                    return False
                    
            return True
        except Exception as e:
            print(f"Warning: Could not extract terms for equality check: {e}. Falling back to False.")
            return False

    def __eq__(self, other):
        return self.equals(other)

    # --- Term and Coefficient Extraction (Improved) ---

    def get_terms(self):
        """
        Yields (exponent_tuple, coefficient) pairs for the polynomial without
        explicitly calling expand().
        
        Relies on symengine's internal representation via as_coefficients_dict().

        Yields
        ------
        tuple
            (exponent_tuple, coefficient), where exponent_tuple is a tuple
            of integers representing the powers of the variables, and
            coefficient is the corresponding complex numerical coefficient.

        Raises
        ------
        AttributeError
            If as_coefficients_dict() is not available for the expression.
        NotImplementedError
            If parsing a specific term structure is not implemented.
        TypeError
            If a coefficient cannot be converted to complex.
        ValueError
            If a coefficient cannot be converted to complex.
        RuntimeError
            If term extraction fails even with expansion fallback.
        """
        term_dict = None
        try:
            # Attempt to get the dictionary directly, assuming it doesn't need prior expansion.
            # This is the key optimization.
            term_dict = self.expr.as_coefficients_dict()
        except AttributeError:
            # Handle cases where the expression might be simple (number, symbol)
            # and doesn't have as_coefficients_dict, or if the method truly requires expansion
            # (which would contradict our optimization assumption).
            if self.expr.is_Number:
                yield (tuple([0] * self.n_vars), complex(float(self.expr)))
                return
            elif self.expr.is_Symbol:
                exp = [0] * self.n_vars
                try:
                    idx = self.variables.index(self.expr)
                    exp[idx] = 1
                    yield(tuple(exp), complex(1.0))
                except ValueError:
                    # Symbol not in self.variables - treat as a constant? Or error?
                    # Yielding as constant term might be unexpected. Let's yield nothing or raise.
                    # For now, yield nothing, assuming only terms with self.variables matter.
                    pass
                return
            else:
                # If it's not Number/Symbol and as_coefficients_dict failed,
                # it might be a single Mul/Pow term or require expansion.
                # Re-raising or raising NotImplementedError is appropriate.
                # Let's try expanding as a fallback ONLY if as_coefficients_dict fails.
                # This makes the optimization attempt explicit.
                print("Warning: as_coefficients_dict() failed or not available directly. Attempting expansion as fallback for get_terms().")
                try:
                    expanded_expr = se.expand(self.expr)
                    # Check if expansion resulted in something with as_coefficients_dict
                    if hasattr(expanded_expr, 'as_coefficients_dict'):
                        term_dict = expanded_expr.as_coefficients_dict()
                    elif expanded_expr.is_Number: # Handle case where expansion simplifies to number
                        yield (tuple([0] * self.n_vars), complex(float(expanded_expr)))
                        return
                    # Add checks for Symbol, Mul, Pow if expansion results in single term
                    #... (similar logic as above for Symbol, and below for Mul/Pow parsing)
                    else:
                        # If expansion didn't help, we can't proceed easily
                        raise NotImplementedError(f"Cannot extract terms for expression type {type(self.expr)} even after expansion.")

                except Exception as e:
                    raise RuntimeError(f"Failed to get terms even with expansion fallback: {e}") from e

        if term_dict is None:
            # This should ideally not be reached if the logic above is sound.
            raise RuntimeError("Term dictionary could not be obtained.")


        # --- Parsing logic remains largely the same, but operates on term_dict ---
        variable_map = {v: i for i, v in enumerate(self.variables)} # Map symbol to index

        for term_expr, coeff_num in term_dict.items():
            exponent = [0] * self.n_vars
            term_processed = False

            if term_expr == se.sympify(1): # Constant term key is 1
                exponent = tuple([0] * self.n_vars)
                term_processed = True
            elif term_expr == se.sympify(0): # Zero term 
                # Skip zero terms or yield with zero coefficient
                continue  # Skip zero terms entirely
            elif isinstance(term_expr, se.Symbol):
                try:
                    idx = variable_map[term_expr]
                    exponent[idx] = 1
                    term_processed = True
                except KeyError:
                    # Symbol not in self.variables, skip this term
                    # (Could be a symbolic coefficient part, but as_coefficients_dict should handle that)
                    print(f"Warning: Skipping term with unknown symbol {term_expr} in get_terms().")
                    continue
            elif isinstance(term_expr, se.Pow):
                base, exp_val = term_expr.args
                if isinstance(base, se.Symbol) and exp_val.is_Integer:
                    try:
                        idx = variable_map[base]
                        exponent[idx] = int(exp_val)
                        term_processed = True
                    except KeyError:
                        print(f"Warning: Skipping term with unknown symbol base {base} in get_terms().")
                        continue
                else:
                    # Handle non-symbol base or non-integer exponent if necessary,
                    # but typically polynomial terms have symbol bases and integer exponents.
                    print(f"Warning: Skipping non-standard Pow term {term_expr} in get_terms().")
                    continue
            elif isinstance(term_expr, se.Mul):
                valid_term = True
                temp_exponent = [0] * self.n_vars # Use temp to handle errors during factor processing
                for factor in term_expr.args:
                    if isinstance(factor, se.Symbol):
                        try:
                            idx = variable_map[factor]
                            temp_exponent[idx] += 1
                        except KeyError: valid_term = False; break
                    elif isinstance(factor, se.Pow):
                        base, exp_val = factor.args
                        if isinstance(base, se.Symbol) and exp_val.is_Integer:
                            try:
                                idx = variable_map[base]
                                temp_exponent[idx] += int(exp_val)
                            except KeyError: valid_term = False; break
                        else: valid_term = False; break # Non-standard Pow factor
                    else:
                        # Factor is not Symbol or Pow (e.g., a numeric factor in the term key?)
                        # This shouldn't happen if as_coefficients_dict separates numeric coefficients.
                        valid_term = False; break
                if valid_term:
                    exponent = temp_exponent
                    term_processed = True
                else:
                    print(f"Warning: Skipping term with unexpected factor structure {term_expr} in get_terms().")
                    continue
            # else: # Handle other potential term_expr types if necessary

            if not term_processed:
                print(f"Warning: Skipping unrecognized term structure {term_expr} (type: {type(term_expr)}) in get_terms().")
                continue

            # Yield the result
            try:
                # Ensure coefficient is numeric complex
                coeff = complex(float(coeff_num))
                yield tuple(exponent), coeff
            except (TypeError, ValueError) as e:
                # This might happen if coeff_num is symbolic (e.g., 'a' in a*x).
                # The current class assumes numeric coefficients for get_terms/get_coefficient.
                # If symbolic coefficients are needed, the return type and subsequent methods
                # (like equals) would need adjustment.
                print(f"Warning: Could not convert coefficient '{coeff_num}' to numeric complex in get_terms(): {e}. Skipping term.")
                # Alternatively, could yield the symbolic coefficient: yield tuple(exponent), coeff_num
                continue


    def get_coefficient(self, exponent_tuple):
        """
        Extracts the numerical coefficient of the monomial corresponding to exponent_tuple.
        
        Uses the potentially optimized get_terms() method.

        Parameters
        ----------
        exponent_tuple : tuple
            A tuple of non-negative integers representing the powers of the 
            variables [x0, x1,...]. Length must match self.n_vars.

        Returns
        -------
        complex
            The numerical coefficient of the specified monomial, or 0.0 if the term 
            does not exist.
            
        Raises
        ------
        ValueError
            If exponent tuple length doesn't match n_vars or if exponents are negative.
        RuntimeError
            If coefficient extraction fails due to error in get_terms.
        """
        if len(exponent_tuple)!= self.n_vars:
            raise ValueError("Exponent tuple length must match n_vars.")
        if not all(isinstance(p, int) and p >= 0 for p in exponent_tuple):
            raise ValueError("Exponents must be non-negative integers.")

        try:
            # Iterate through terms provided by the (hopefully) efficient get_terms
            for exp_t, coeff in self.get_terms():
                if exp_t == exponent_tuple:
                    # get_terms should already yield complex coefficients
                    return coeff
        except Exception as e:
            # Handle potential errors during term iteration
            print(f"Error during get_coefficient -> get_terms iteration: {e}")
            # Depending on desired behavior, could re-raise or return 0
            raise RuntimeError(f"Failed to extract coefficient due to error in get_terms: {e}") from e


        # If the loop completes without finding the term
        return complex(0.0)

    # --- Optional Helper Methods ---

    def total_degree(self):
        if self.expr == se.sympify(0):  # Special case for zero polynomial
            return -1
        return max(sum(mon) for mon, _ in self.get_terms())

    def degree_by_var(self, i):
        if self.expr == se.sympify(0):  # Special case for zero polynomial
            return -1
        return max(mon[i] for mon, _ in self.get_terms())

    def truncate(self, max_degree):
        """
        Truncates the polynomial, keeping only terms with total degree less than
        or equal to max_degree.

        Parameters
        ----------
        max_degree : int
            The maximum total degree to keep. Must be non-negative.

        Returns
        -------
        Polynomial
            A new Polynomial instance containing only the terms up to the specified 
            maximum degree.
            
        Raises
        ------
        ValueError
            If max_degree is not a non-negative integer.
        RuntimeError
            If truncation fails due to error in get_terms.
        """
        if not isinstance(max_degree, int) or max_degree < 0:
            raise ValueError("max_degree must be a non-negative integer.")

        # Special case for zero polynomial
        if self.expr == se.sympify(0):
            return Polynomial.zero(self.n_vars, self.variables)
            
        # Special cases: constant term only for max_degree=0
        if max_degree == 0:
            # Extract the constant term
            try:
                const_exp = tuple([0] * self.n_vars)
                coeff = self.get_coefficient(const_exp)
                if abs(coeff) < 1e-12:  # No constant term
                    return Polynomial.zero(self.n_vars, self.variables)
                else:
                    # Use string format to avoid complex notation
                    return Polynomial(str(int(coeff) if coeff.imag == 0 and coeff.real == int(coeff.real) 
                                        else float(coeff.real)), n_vars=self.n_vars, variables=self.variables)
            except Exception as e:
                print(f"Warning: Error extracting constant term: {e}. Using fallback method.")
                # Fall through to regular method
            
        # If polynomial is already of degree <= max_degree, return a copy
        try:
            if self.total_degree() <= max_degree:
                return Polynomial(self.expr, self.n_vars, self.variables)
        except Exception:
            # If total_degree fails, continue with truncation
            pass

        # For numerical comparison, expand the expression first
        expanded_expr = se.expand(self.expr)
        try:
            # Create a new polynomial with the expanded expression for easier term extraction
            expanded_poly = Polynomial(expanded_expr, self.n_vars, self.variables)
            
            # Build the truncated polynomial manually - this approach avoids complex notation
            terms = []
            for exp_tuple, coeff in expanded_poly.get_terms():
                term_degree = sum(exp_tuple)
                if term_degree <= max_degree and abs(coeff) > 1e-12:
                    # Format coefficient to avoid complex notation
                    coeff_str = str(int(coeff.real) if coeff.imag == 0 and coeff.real == int(coeff.real) 
                                   else float(coeff.real))
                    if coeff_str == "1" and sum(exp_tuple) > 0:
                        coeff_str = ""  # Skip "1*" for variables
                    elif coeff_str == "-1" and sum(exp_tuple) > 0:
                        coeff_str = "-"  # Use just "-" for negative vars
                    elif coeff_str != "0":
                        coeff_str += "*"  # Add multiplication symbol
                        
                    # Skip the term if coefficient is zero
                    if coeff_str == "0" or coeff_str == "0.0*":
                        continue
                        
                    # Build the variables part
                    var_parts = []
                    for i, power in enumerate(exp_tuple):
                        if power == 0:
                            continue
                        elif power == 1:
                            var_parts.append(f"x{i}")
                        else:
                            var_parts.append(f"x{i}**{power}")
                    
                    # Combine coefficient and variables
                    if var_parts:
                        if coeff_str in ["", "-"]:
                            term = f"{coeff_str}{var_parts[0]}"
                            for var in var_parts[1:]:
                                term += f"*{var}"
                        else:
                            term = f"{coeff_str}{var_parts[0]}"
                            for var in var_parts[1:]:
                                term += f"*{var}"
                    else:
                        # Constant term
                        term = coeff_str.rstrip("*")
                    
                    terms.append(term)
            
            # Special case for zero result (no terms)
            if not terms:
                return Polynomial.zero(self.n_vars, self.variables)
                
            # Build the polynomial expression
            result_str = " + ".join(terms).replace(" + -", " - ")
            return Polynomial(result_str, self.n_vars, self.variables)

        except Exception as e:
            print(f"Error during truncate: {e}")
            # Fallback method if the more precise approach fails
            new_expr = se.sympify(0)
            for exp_tuple, coeff in expanded_poly.get_terms():
                term_degree = sum(exp_tuple)
                if term_degree <= max_degree:
                    # Reconstruct the term
                    term = se.sympify(1)
                    for i, power in enumerate(exp_tuple):
                        if power > 0:
                            term *= self.variables[i]**power
                    term *= coeff
                    new_expr += term

            return Polynomial(new_expr, self.n_vars, self.variables)

    def evaluate(self, values, use_lambda=True):
        """
        Evaluates the polynomial at the given values.
        
        Parameters
        ----------
        values : list
            List of values to substitute for variables [x0, x1, ...].
            Length must match n_vars.
        use_lambda : bool, optional
            Whether to use symengine's lambdify for faster evaluation.
            If False, uses symbolic substitution. Default is True.
            
        Returns
        -------
        float
            The value of the polynomial at the given point.
            
        Raises
        ------
        ValueError
            If length of values doesn't match n_vars.
        """
        if len(values) != self.n_vars:
            raise ValueError("length mismatch")
        
        # Special case for zero polynomial
        if self.expr == se.sympify(0):
            return 0.0
            
        # Special case for constant polynomials
        if self.expr.is_Number:
            return float(self.expr)
            
        # Special case for single variable
        if self.expr.is_Symbol:
            var_idx = self.variables.index(self.expr)
            return float(values[var_idx])
            
        if use_lambda:
            try:
                # First, try direct numerical evaluation without lambdify
                subs_dict = {var: se.Float(val) for var, val in zip(self.variables, values)}
                direct_result = self.expr.subs(subs_dict)
                if direct_result.is_Number:
                    return float(direct_result)
                
                # If that fails, try lambdify
                if not hasattr(self, "_lamb"):
                    try:
                        # Use sympify to handle different expression formats
                        from symengine import lambdify
                        expr = se.sympify(self.expr)
                        self._lamb = lambdify(self.variables, expr)
                        self._lamb_returns_array = False
                    except Exception:
                        print("Warning: Failed to create standard lambdify, trying with array approach")
                        from symengine import lambdify
                        self._lamb = lambdify(self.variables, [self.expr])
                        self._lamb_returns_array = True
                
                result = self._lamb(*values)
                if hasattr(self, "_lamb_returns_array") and self._lamb_returns_array:
                    if hasattr(result, "__len__"):
                        return float(result[0])
                    else:
                        return float(result)
                else:
                    return float(result)
            except Exception as e:
                print(f"Warning: Lambda evaluation failed: {e}. Falling back to substitution.")
                use_lambda = False
        
        # Use direct substitution without evalf
        try:
            subs_dict = {var: se.Float(val) for var, val in zip(self.variables, values)}
            result = self.expr.subs(subs_dict)
            return float(result)
        except Exception as e:
            print(f"Warning: Substitution evaluation failed: {e}")
            # Last resort: try with sympify and numeric evaluation
            numeric_expr = self.expr
            for i, var in enumerate(self.variables):
                numeric_expr = numeric_expr.subs(var, se.Float(values[i]))
            return float(numeric_expr)

    def as_float(self, prec=53):
        conv = {c: se.Float(c.evalf(), prec)
                for c in self.expr.atoms(se.Number) if not c.is_Integer}
        return Polynomial(self.expr.xreplace(conv), self.n_vars, self.variables)
        
    def gradient(self, point=None):
        """
        Computes the gradient (vector of partial derivatives) of the polynomial.
        
        Parameters
        ----------
        point : array-like, optional
            If provided, evaluates the gradient at this point.
            Should be a sequence of length n_vars.
        
        Returns
        -------
        numpy.ndarray
            A vector of partial derivatives, evaluated at the given point if provided.
            
        Raises
        ------
        ValueError
            If point is provided but has incorrect length.
        """
        # Compute partial derivatives with respect to each variable
        derivatives = [self.differentiate(i) for i in range(self.n_vars)]
        
        if point is None:
            # Return the symbolic gradient as a list of Polynomials
            return derivatives
        else:
            # Validate point length
            if len(point) != self.n_vars:
                raise ValueError(f"Point must have length {self.n_vars}, got {len(point)}")
                
            # Evaluate each derivative at the given point
            return np.array([derivative.evaluate(point) for derivative in derivatives])
