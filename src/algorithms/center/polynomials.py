import symengine as se
import numpy as np # For isclose later if needed, though less critical with symbolic
from abc import ABC, abstractmethod


class Polynomial():
    """
    A Polynomial class using symengine as the backend for symbolic manipulation.
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
        else:
            # Generate standard variable names if not provided
            var_key = n_vars
            if var_key not in Polynomial._symbol_cache:
                 # Standard names: x0, x1, x2,... corresponding to q1, p1, q2,...
                 Polynomial._symbol_cache[var_key] = se.symbols([f'x{i}' for i in range(n_vars)])
            self.variables = Polynomial._symbol_cache[var_key]

        # Store the internal symengine expression
        if isinstance(expression, se.Expr):
            self.expr = expression
        elif isinstance(expression, (int, float, complex)):
            # Directly convert numeric values
            self.expr = se.sympify(expression)
        elif isinstance(expression, str):
            # For strings, we need to handle the variables manually
            # Parse the string expression with variable substitution
            expr_str = expression
            self.expr = se.sympify(expr_str)
            
            # Replace any symbolic variables with our actual variables
            # This is needed because symengine.sympify doesn't accept locals parameter
            substitutions = {}
            for i, var in enumerate(self.variables):
                var_name = f'x{i}'
                symbol_in_expr = se.Symbol(var_name)
                if symbol_in_expr != var and symbol_in_expr in self.expr.free_symbols:
                    substitutions[symbol_in_expr] = var
            
            if substitutions:
                self.expr = self.expr.subs(substitutions)
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
        return f"Polynomial('{str(self.expr)}', n_vars={self.n_vars})"

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
