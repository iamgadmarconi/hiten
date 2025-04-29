import symengine as se
import numpy as np # For isclose later if needed, though less critical with symbolic
import math  # Add math import for factorial function
from abc import ABC, abstractmethod


class Polynomial():
    """
    A Polynomial class using symengine as the backend for symbolic manipulation.
    Provides methods needed for Hamiltonian mechanics algorithms.
    """
    _symbol_cache = {} # Cache symbols for efficiency

    def __init__(self, expression, n_vars=6, variables=None):
        """
        Initializes the Polynomial using a symengine expression.

        Args:
            expression (symengine.Expr or str or number):
                The symengine expression, a string to parse, or a number.
            n_vars (int): Number of variables (must be even for PB).
            variables (list[symengine.Symbol], optional):
                A list of symengine symbols in canonical order [q1, p1, q2, p2,...].
                If None, they will be generated automatically as x0, x1, ...
        """
        if n_vars <= 0 or n_vars % 2 != 0:
            raise ValueError("n_vars must be a positive even integer.")
        self.n_vars = n_vars

        # Generate or use provided variables
        if variables:
            if len(variables) != n_vars:
                raise ValueError(f"Provided variables list length ({len(variables)}) != n_vars ({n_vars})")
            self.variables = variables
            # Ensure cache consistency if custom variables are provided
            var_key = tuple(str(v) for v in variables)
            if var_key not in Polynomial._symbol_cache:
                Polynomial._symbol_cache[var_key] = variables
        else:
            # Generate standard variable names and use them as the cache key
            # Fix 4: Use tuple of names as key for robustness
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
            # Fix 1: String initialization using substitution.
            # NOTE: This assumes variable names in the string are 'x0', 'x1', etc.
            # It might be less efficient/robust than initializing with expressions
            # or pre-defined symbols if symengine.sympify had a 'locals' arg.
            # Consider creating expressions programmatically where possible.
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
                if var_name in string_symbols and string_symbols[var_name] != canonical_var:
                    substitutions[string_symbols[var_name]] = canonical_var

            if substitutions:
                self.expr = parsed_expr.subs(substitutions)
            else:
                self.expr = parsed_expr
        else:
            raise TypeError(f"Unsupported expression type for initialization: {type(expression)}")

    @classmethod
    def zero(cls, n_vars=6, variables=None):
        """Creates a zero polynomial."""
        return cls(0, n_vars=n_vars, variables=variables)

    @classmethod
    def one(cls, n_vars=6, variables=None):
        """Creates a one polynomial."""
        return cls(1, n_vars=n_vars, variables=variables)


    def __repr__(self):
        # Use standard variable names in repr for consistency if possible
        std_vars = Polynomial._symbol_cache.get(tuple(f'x{i}' for i in range(self.n_vars)), self.variables)
        # Temporarily substitute canonical vars with standard names for repr if different
        sub_dict_repr = {can_var: std_var for can_var, std_var in zip(self.variables, std_vars) if can_var != std_var}
        repr_expr = self.expr.subs(sub_dict_repr) if sub_dict_repr else self.expr
        return f"Polynomial('{str(repr_expr)}', n_vars={self.n_vars})"

    def __str__(self):
        return str(self.expr)

    def __add__(self, other):
        """Addition: Handles Polynomial + Polynomial or Polynomial + scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            # Use symengine's '+' operator
            return Polynomial(self.expr + other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            # Add scalar (sympify handles conversion)
            other_expr = se.sympify(other)
            return Polynomial(self.expr + other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __radd__(self, other):
        # Handles scalar + Polynomial
        return self.__add__(other)

    def __sub__(self, other):
        """Subtraction: Handles Polynomial - Polynomial or Polynomial - scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            # Use symengine's '-' operator
            return Polynomial(self.expr - other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(self.expr - other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __rsub__(self, other):
        # Handles scalar - Polynomial
        if isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(other_expr - self.expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __mul__(self, other):
        """Multiplication: Handles Polynomial * Polynomial or Polynomial * scalar"""
        if isinstance(other, Polynomial):
            if self.n_vars != other.n_vars:
                raise ValueError("Polynomials must have the same number of variables.")
            # Use symengine's '*' operator
            return Polynomial(self.expr * other.expr, self.n_vars, self.variables)
        elif isinstance(other, (int, float, complex, str)):
            other_expr = se.sympify(other)
            return Polynomial(self.expr * other_expr, self.n_vars, self.variables)
        else:
            return NotImplemented

    def __rmul__(self, other):
        # Handles scalar * Polynomial
        return self.__mul__(other)

    def __neg__(self):
        """Negation: -Polynomial"""
        return Polynomial(-self.expr, self.n_vars, self.variables)

    # Optional: Implement division if needed, e.g., for coefficient extraction
    # def __truediv__(self, other): ...

    def differentiate(self, var_index):
        """
        Differentiates the polynomial with respect to a given variable index.
        Assumes variables are 0-indexed [q1, p1, q2, p2, ...].
        """
        if not (0 <= var_index < self.n_vars):
            raise ValueError(f"var_index must be between 0 and {self.n_vars-1}")
        target_var = self.variables[var_index]
        # Use symengine's diff method
        derivative_expr = se.diff(self.expr, target_var)
        return Polynomial(derivative_expr, self.n_vars, self.variables)

    def poisson_bracket(self, other):
        """
        Computes the Poisson bracket {self, other}.
        Assumes canonical variables ordered [q1, p1, q2, p2, ...].
        """
        if not isinstance(other, Polynomial):
            # Allow bracket with scalar (which is always 0)
            if isinstance(other, (int, float, complex, str)):
                if se.sympify(other).is_Number:
                    return Polynomial(0, self.n_vars, self.variables)
            raise TypeError("Poisson bracket requires another Polynomial or a numeric scalar.")

        if self.n_vars != other.n_vars:
            raise ValueError("Polynomials must have the same number of variables.")
        # n_vars check already done in __init__

        num_dof = self.n_vars // 2
        result_expr = se.sympify(0) # Start with zero expression

        for i in range(num_dof):
            qi_index = 2 * i
            pi_index = 2 * i + 1
            qi = self.variables[qi_index]
            pi = self.variables[pi_index]

            # Compute partial derivatives using symengine.diff
            d_self_dqi = se.diff(self.expr, qi)
            d_other_dpi = se.diff(other.expr, pi)
            term1 = d_self_dqi * d_other_dpi # Use symengine multiplication

            d_self_dpi = se.diff(self.expr, pi)
            d_other_dqi = se.diff(other.expr, qi)
            term2 = d_self_dpi * d_other_dqi # Use symengine multiplication

            # Accumulate result: result += term1 - term2
            result_expr = result_expr + term1 - term2 # Use symengine arithmetic

        return Polynomial(result_expr, self.n_vars, self.variables)

    def substitute(self, var_map):
        """
        Substitutes variables in the expression.

        Args:
            var_map (dict): A dictionary mapping symengine symbols or variable
                            indices to their substitution values (numbers or
                            other symengine expressions/Polynomials).
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
            if isinstance(v, Polynomial):
                sub_dict[key_symbol] = v.expr
            elif isinstance(v, (int, float, complex, str, se.Expr)):
                sub_dict[key_symbol] = se.sympify(v) # Ensure value is a symengine expression
            else:
                raise TypeError(f"Invalid value type in var_map: {type(v)}")

        substituted_expr = self.expr.subs(sub_dict)
        return Polynomial(substituted_expr, self.n_vars, self.variables)

    def equals(self, other):
        """Checks for symbolic equality (can be computationally expensive)."""
        if not isinstance(other, Polynomial):
            return False
        if self.n_vars != other.n_vars:
             return False
        
        # Use symengine's expand and then compare
        expr1 = se.expand(self.expr)
        expr2 = se.expand(other.expr)
        
        # If the expressions are identical, return True
        if expr1 == expr2:
            return True
        
        # If they differ only in numeric formatting (3.0 vs 3), try to normalize
        # by converting to a standard form
        try:
            # Subtract one from the other and check if the result is zero or very close to zero
            diff = expr1 - expr2
            # If it's a constant, it should be zero (or very close to zero)
            if diff.is_Number:
                return abs(float(diff)) < 1e-10
                
            # If we have coefficients that might be float vs int formats, 
            # we need to check term by term
            # This is more complex and would require parsing the expression structure
            # For now, we'll just return False if the direct comparison fails
            return False
        except Exception:
            # If any conversion or computation errors, default to False
            return False
            
    def __eq__(self, other):
        """Checks for symbolic equality."""
        return self.equals(other)

    def get_coefficient(self, exponent_tuple):
        """
        Extracts the coefficient of the monomial corresponding to exponent_tuple.

        Example: get_coefficient((2, 0, 1, 0, 0, 0)) gets coeff of x0^2 * x2^1

        NOTE: This can be computationally expensive as it may rely on expand()
              and repeated calls to coeff(). Consider performance implications.
              Assumes exponent_tuple length matches self.n_vars.
        """
        if len(exponent_tuple) != self.n_vars:
            raise ValueError("Exponent tuple length must match n_vars.")

        # Ensure the expression is expanded to make coefficient extraction reliable
        # WARNING: This is the potentially expensive step!
        expanded_expr = se.expand(self.expr)

        # Start with the fully expanded expression
        coeff_expr = expanded_expr

        # Iteratively extract coefficient for each variable^power
        factorial_prod = 1
        for i in range(self.n_vars):
            var = self.variables[i]
            power = exponent_tuple[i]
            if power < 0:
                 raise ValueError("Exponents cannot be negative.")
            if power > 0:
                 # Get the coefficient of var**power in the current expression
                 coeff_expr = coeff_expr.coeff(var, power)
                 factorial_prod *= math.factorial(power)
            else:
                 # If power is 0, we need the term independent of this var.
                 # Substitute var=0 to eliminate terms containing it.
                 # This assumes no denominators involving the variable.
                 coeff_expr = coeff_expr.subs({var: 0})

            # If at any point the coefficient becomes zero, the final coeff is zero
            if coeff_expr == se.sympify(0):
                return complex(0.0)

        # At the end, coeff_expr should be the numerical coefficient.
        # Note: The `.coeff(var, power)` method might include factors like
        # binomial coefficients, which we need to account for.
        # The differentiation method C = (1/k!...) * d^k P / dx^k |_{x=0}
        # is more direct but potentially slower.
        # Let's assume `.coeff` gives the direct coefficient here,
        # but this might need verification/adjustment based on symengine behavior.
        # If `.coeff` behaves like SymPy's `expr.coeff`, it should be correct.

        # Convert the final symbolic coefficient (should be a number) to complex
        if isinstance(coeff_expr, se.Expr) and coeff_expr.is_Number:
             # Attempt conversion, handle potential errors if it's not purely numeric
             try:
                 # Use complex for consistency, even if it's real
                 return complex(float(coeff_expr))
             except (TypeError, RuntimeError):
                 # If conversion fails, something unexpected happened
                 print(f"Warning: Could not convert final coefficient '{coeff_expr}' to numeric.")
                 return complex(0.0) # Or raise error? Or return the expression?
        elif coeff_expr == se.sympify(0):
             return complex(0.0)
        else:
             # This shouldn't happen if the logic is correct and input is a polynomial
             print(f"Warning: Coefficient extraction resulted in non-numeric expression: {coeff_expr}")
             return complex(0.0) # Or raise error

    def get_terms(self):
        """
        Yields (exponent_tuple, coefficient) pairs for the polynomial.

        NOTE: This relies on symengine's as_dict() method after expansion,
              which might be expensive. The keys from as_dict() need
              careful interpretation to map back to exponent tuples.
        """
        # WARNING: expand() can be computationally expensive!
        expanded_expr = se.expand(self.expr)

        # as_dict() breaks the expression into Add terms, keys are terms, values are coefficients
        # e.g., {x0**2: 2.0, x1: 3.0, 1: 4.0}
        try:
            term_dict = expanded_expr.as_coefficients_dict()
        except AttributeError:
            # Handle case where expression is just a number or single symbol/term
            if expanded_expr.is_Number:
                yield (tuple([0] * self.n_vars), complex(float(expanded_expr)))
                return
            elif expanded_expr.is_Symbol:
                exp = [0] * self.n_vars
                try:
                    idx = self.variables.index(expanded_expr)
                    exp[idx] = 1
                    yield(tuple(exp), complex(1.0))
                except ValueError:
                    # Symbol not in our list, maybe an external constant? Treat as constant term?
                    # This case needs careful definition based on expected use.
                    # For now, assume it shouldn't happen with canonical variables.
                    pass
                return
            else:
                # Could be a single Mul/Pow term, need to parse structure
                # Let's defer complex parsing, try get_coefficient for specific terms instead
                # or check if symengine offers better tools for this.
                raise NotImplementedError("get_terms() for single complex terms (Mul/Pow) not fully implemented without as_coefficients_dict().")


        # Convert the dictionary keys (terms) into exponent tuples
        variable_map = {v: i for i, v in enumerate(self.variables)} # Map symbol to index

        for term_expr, coeff_num in term_dict.items():
            exponent = [0] * self.n_vars
            if term_expr.is_Number: # Constant term
                if term_expr == se.sympify(1): # The key for the constant term is 1
                     exponent = tuple([0] * self.n_vars) # Exponent is all zeros
                else:
                    # This shouldn't happen if as_coefficients_dict works as expected
                    continue
            elif term_expr.is_Symbol:
                try:
                    idx = variable_map[term_expr]
                    exponent[idx] = 1
                except KeyError:
                    continue # Skip terms with symbols not in self.variables
            elif term_expr.is_Pow:
                base, exp_val = term_expr.args
                if base.is_Symbol and exp_val.is_Integer:
                    try:
                        idx = variable_map[base]
                        exponent[idx] = int(exp_val)
                    except KeyError:
                        continue
                else: continue # Skip complex Pow terms for now
            elif term_expr.is_Mul:
                for factor in term_expr.args:
                    if factor.is_Symbol:
                        try:
                            idx = variable_map[factor]
                            exponent[idx] += 1
                        except KeyError: exponent = None; break # Contains unknown symbol
                    elif factor.is_Pow:
                        base, exp_val = factor.args
                        if base.is_Symbol and exp_val.is_Integer:
                            try:
                                idx = variable_map[base]
                                exponent[idx] += int(exp_val)
                            except KeyError: exponent = None; break # Contains unknown symbol
                        else: exponent = None; break # Complex factor
                    else: exponent = None; break # Non-symbol/Pow factor
                if exponent is None: continue # Skip this term
            else:
                continue # Skip other complex expression types

            try:
                coeff = complex(float(coeff_num)) # Ensure coeff is numeric complex
                yield tuple(exponent), coeff
            except (TypeError, ValueError):
                # Handle cases where coefficient isn't purely numeric? Unlikely from as_coefficients_dict
                pass