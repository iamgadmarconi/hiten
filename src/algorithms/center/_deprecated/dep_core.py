from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, MutableMapping, Tuple, TYPE_CHECKING

import math
from math import factorial
import symengine as se
import numpy as np


class Polynomial:
    """
    Minimal, robust wrapper around a SymEngine expression.

    * Default variables   :  x0, x1, …  (lower-case) so the unit-tests work.
    * Optional `variables`:  list of SymEngine symbols in canonical order
                             [q1, p1, q2, p2, …].  Length must equal `n_vars`.
    """

    # ------------------------------------------------------------------ #
    # simple cache so we don't re-create the same symbol objects
    # key   = tuple of symbol *names*  ('x0','x1',…)
    # value = list[Symbol]
    _symbol_cache: dict[tuple[str, ...], list[se.Symbol]] = {}

    # ------------------------------------------------------------------ #
    def __init__(self, expr, n_vars: int = 6, variables: list[se.Symbol] | None = None):
        if n_vars <= 0 or n_vars % 2:
            raise ValueError("n_vars must be a positive, even integer")

        # ---- choose / create variable list --------------------------------
        if variables is None:
            key = tuple(f"x{i}" for i in range(n_vars))      # ('x0','x1',...)
            if key not in Polynomial._symbol_cache:
                Polynomial._symbol_cache[key] = se.symbols(list(key))
            self.variables = Polynomial._symbol_cache[key]
        else:
            if len(variables) != n_vars:
                raise ValueError("len(variables) must equal n_vars")
            # normalise to tuple of names so 'X' and Symbol('X') hash the same
            key = tuple(str(s) for s in variables)
            Polynomial._symbol_cache.setdefault(key, variables)
            self.variables = Polynomial._symbol_cache[key]

        self.n_vars = n_vars

        # ---- parse / store expression -------------------------------------
        if isinstance(expr, se.Basic):
            self.expr = expr
        elif isinstance(expr, (int, float, complex)):
            self.expr = se.sympify(expr)
        elif isinstance(expr, str):
            parsed = se.sympify(expr)
            # substitute any stray 'x0', 'x1'… symbols with the canonical ones
            sub_map = {se.Symbol(f"x{i}"): self.variables[i] for i in range(n_vars)}
            self.expr = parsed.xreplace(sub_map)
        else:
            raise TypeError(f"Unsupported expression type: {type(expr)}")

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

    def is_zero(self):
        return self.expr == se.sympify(0)

    def is_one(self):
        return self.expr == se.sympify(1)
    
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

    @staticmethod
    @lru_cache(maxsize=10_000)
    def _calculate_pb_expr(expr1, expr2, variables):
        """
        Static helper method to calculate the Poisson bracket expression.
        This method is cached for performance.
        
        Parameters
        ----------
        expr1 : symengine.Expr
            First expression for Poisson bracket.
        expr2 : symengine.Expr
            Second expression for Poisson bracket.
        variables : tuple
            Tuple of symengine symbols in canonical order [q1, p1, q2, p2,...].
            
        Returns
        -------
        symengine.Expr
            The expression resulting from Poisson bracket calculation.
        """
        n_vars = len(variables)
        num_dof = n_vars // 2
        result_expr = se.sympify(0)  # Start with zero expression

        for i in range(num_dof):
            qi_index = 2 * i
            pi_index = 2 * i + 1
            qi = variables[qi_index]
            pi = variables[pi_index]

            # Compute partial derivatives
            d_expr1_dqi = se.diff(expr1, qi)
            d_expr2_dpi = se.diff(expr2, pi)
            term1 = d_expr1_dqi * d_expr2_dpi

            d_expr1_dpi = se.diff(expr1, pi)
            d_expr2_dqi = se.diff(expr2, qi)
            term2 = d_expr1_dpi * d_expr2_dqi

            result_expr = result_expr + term1 - term2

        return result_expr

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

        if self.expr.is_Number or other.expr.is_Number:
            return Polynomial.zero(self.n_vars, self.variables)

        result_expr = Polynomial._calculate_pb_expr(self.expr, other.expr,
                                                    tuple(self.variables))

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
        Yield (exponent_tuple, coefficient) pairs.

        Works for *any* SymEngine expression after se.expand():  
        constants, a lone monomial, or an Add of monomials.

        Assumes self.variables is ordered [X, Y, Z, PX, PY, PZ] (6 vars).
        """
        var2idx = {v: i for i, v in enumerate(self.variables)}

        def _one_term(expr):
            # constant
            if expr.is_Number:
                yield (tuple([0] * self.n_vars), complex(float(expr)))
                return

            # generic monomial  -> use as_powers_dict()
            powers = expr.as_powers_dict()           # {Symbol: exponent}
            coeff  = expr / se.Mul(*[b**e for b, e in powers.items()])

            exps = [0] * self.n_vars
            for base, exp in powers.items():
                if base not in var2idx:
                    # Instead of silently dropping the term, raise an error
                    # so that the issue is more visible
                    raise ValueError(f"Symbol {base} not found in variables {self.variables}")
                exps[var2idx[base]] = int(exp)
            yield tuple(exps), complex(float(coeff))

        # expand once; if the result is an Add, walk its .args
        expanded = se.expand(self.expr)
        if isinstance(expanded, se.Add):
            for term in expanded.args:
                yield from _one_term(term)
        else:
            yield from _one_term(expanded)

    def get_coefficient(self, exponent_tuple):
        """
        Return the numeric coefficient of the monomial with exponents
        `exponent_tuple`.  Works for constants (all zeros) as well.

        This version does *not* rely on get_terms(); instead it queries
        SymEngine's internal coefficient map after a single expand().
        """
        if len(exponent_tuple) != self.n_vars:
            raise ValueError("Exponent tuple length must match n_vars.")
        if any(e < 0 for e in exponent_tuple):
            raise ValueError("Exponents must be non-negative integers.")

        expanded = se.expand(self.expr)
        coeff_dict = expanded.as_coefficients_dict()

        # Build the monomial key   var0**e0 * var1**e1 * …
        monomial = se.Integer(1)
        for var, exp in zip(self.variables, exponent_tuple):
            if exp:
                monomial *= var**exp

        coeff = coeff_dict.get(monomial, se.Integer(0))
        return complex(float(coeff))

    def total_degree(self):
        if self.expr == se.sympify(0):  # Special case for zero polynomial
            return -1
        if self.expr.is_Number:
            return 0                 # constant

        # 2) try normal term walk
        terms = list(self.get_terms())
        if terms:                    # at least one monomial found
            return max(sum(mon) for mon, _ in terms)

        # 3) fallback: treat *expr* itself as a single monomial
        #    => total degree = sum of exponents in .as_powers_dict()
        return sum(self.expr.as_powers_dict().values())

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
                        self._lamb = lambdify(self.variables, expr, backend="llvm")
                        self._lamb_returns_array = False
                    except Exception:
                        print("Warning: Failed to create standard lambdify, trying with array approach")
                        from symengine import lambdify
                        self._lamb = lambdify(self.variables, [self.expr], backend="llvm")
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


class FormalSeries(MutableMapping[int, "Polynomial"]):
    """
    Sparse homogeneous power series.
    """

    def __init__(self, mapping: Dict[int, "Polynomial"] | None = None):
        self._data: Dict[int, Polynomial] = dict(mapping or {})
        if mapping:
            for k, v in mapping.items():
                self[k] = v  # uses __setitem__ validation

    # --- mutable‑mapping protocol -------------------------------------------
    def __getitem__(self, key: int):
        return self._data[key]

    def __setitem__(self, key: int, value: "Polynomial"):
        if not isinstance(key, int) or key < 0:
            raise KeyError("Degree must be a non‑negative integer.")
        self._data[key] = value

    def __delitem__(self, key: int):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def degrees(self) -> List[int]:
        return sorted(self._data.keys())

    def truncate(self, max_degree: int) -> "FormalSeries":
        """Return a shallow copy containing only terms ≤ max_degree."""
        return FormalSeries({k: v for k, v in self._data.items() if k <= max_degree})

    @staticmethod
    def poisson_pair(series1: "FormalSeries", series2: "FormalSeries", degree: int):
        """Return the homogeneous degree-`degree` part of {S1, S2}."""
        # Leibniz: look at all pairs k + j - 2 == degree (since PB adds degrees‑2)
        res = None
        for k, pk in series1._data.items():
            j = degree + 2 - k
            if j in series2._data:
                term = pk.poisson_bracket(series2[j])
                if res is None:
                    res = term
                else:
                    res += term
        # Return None if res is None or if it's a zero polynomial
        return None if res is None or res.expr == se.sympify(0) else res

    def lie_transform(self, chi: Polynomial, k_max: int) -> "FormalSeries":
        """Return *new* series  e^{L_χ} H  truncated at total degree k_max.

        Implements the recursive cascade described in Jorba (1999), Algorithm
        *traham* - each iteration re-uses earlier ad-powers so the total cost is
        O(k_max^2) but ~100× faster than naïve because χ is small.
        """
        out = FormalSeries(self._data.copy())
        chi_deg = chi.total_degree()
        if chi.is_zero():
            return out

        # ad_power[d] holds (ad_χ)^r (H_{d}) for current r
        ad_power: Dict[int, Polynomial] = {}

        # first-order bracket
        for d, poly in self._data.items():
            tgt = d + chi_deg - 2
            if tgt <= k_max:
                ad_power[tgt] = ad_power.get(tgt, Polynomial.zero(poly.n_vars, poly.variables)) + chi.poisson_bracket(poly)
                out[tgt] = out.get(tgt, Polynomial.zero(poly.n_vars, poly.variables)) + ad_power[tgt]

        # higher-order brackets (r ≥ 2)
        r = 2
        while True:
            coef = 1.0 / factorial(r)
            new_ad: Dict[int, Polynomial] = {}
            pushed = False
            for d, poly in ad_power.items():
                tgt = d + chi_deg - 2
                if tgt > k_max:
                    continue
                term = chi.poisson_bracket(poly)
                if term.is_zero():
                    continue
                new_ad[tgt] = term
                out[tgt] = out.get(tgt, Polynomial.zero(term.n_vars, term.variables)) + coef * term
                pushed = True
            if not pushed:
                break  # nothing new added – convergence
            ad_power = new_ad
            r += 1
        return out

    def __str__(self):
        terms = ", ".join(f"deg{d}" for d in self.degrees())
        return f"FormalSeries({terms})"

    __repr__ = __str__


@dataclass(slots=True)
class Hamiltonian:
    series: FormalSeries
    mu: float
    coords: str = "synodic"  # 'synodic', 'normal', 'center'
    order: int = field(init=False)

    def __post_init__(self):
        self.order = max(self.series.degrees()) if len(self.series) else 0

    # ---------------------------------------------------------------------
    def quadratic(self):
        return self.series[2]

    # ---------------------------------------------------------------------
    def evaluate(self, x: np.ndarray, float_only: bool = True) -> float:
        """Evaluate ∑ H_k(x).

        Parameters
        ----------
        x : np.ndarray length = n_vars
        float_only : if True, cast result to float (double‑precision).
        """
        val = sum(p.evaluate(x) for p in self.series.values())
        return float(val) if float_only else val

    # ---------------------------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.add.reduce([p.gradient(x) for p in self.series.values()])

    def symplectic_dot(self, grad: np.ndarray) -> np.ndarray:
        """Convert \nabla H = (∂H/∂q, ∂H/∂p) into the Hamiltonian vector field.
        Works with any even-dimensional gradient vector."""
        if grad.size % 2 != 0:
            raise ValueError("symplectic_dot expects an even-length gradient vector.")
        n = grad.size // 2
        dq = grad[n:]
        dp = -grad[:n]
        return np.concatenate((dq, dp))

    # ---------------------------------------------------------------------
    def vector_field(self, x: np.ndarray) -> np.ndarray:
        return self.symplectic_dot(self.gradient(x))

    # ---------------------------------------------------------------------
    def poisson(self, other: "Hamiltonian") -> "Hamiltonian":
        max_deg = self.order + other.order - 2
        data: Dict[int, "Polynomial"] = {}
        for d in range(2, max_deg + 1):
            val = FormalSeries.poisson_pair(self.series, other.series, d)
            if val is not None:
                data[d] = val
        return Hamiltonian(FormalSeries(data), self.mu, self.coords)

    # ---------------------------------------------------------------------
    def change_variables(self, transform) -> "Hamiltonian":
        """Return a new Hamiltonian H∘T where `transform` is a callable that
        maps a Polynomial to another Polynomial (e.g. linear C⁻¹ map, Lie
        transform, centre-manifold injection…)."""
        new_series = FormalSeries({k: transform(p) for k, p in self.series.items()})
        return Hamiltonian(new_series, self.mu, self.coords)

    # ---------------------------------------------------------------------
    # I/O helpers ----------------------------------------------------------
    def to_hdf(self, path: str):
        import h5py, pickle
        with h5py.File(path, "w") as f:
            f.attrs["mu"] = self.mu
            f.attrs["coords"] = self.coords
            grp = f.create_group("series")
            for d, poly in self.series._data.items():
                # Use pickle for binary serialization
                binary_data = pickle.dumps(poly.expr, protocol=5)
                grp.create_dataset(str(d), data=np.void(binary_data))

    @staticmethod
    def from_hdf(path: str) -> "Hamiltonian":
        import h5py, pickle

        with h5py.File(path, "r") as f:
            mu = float(f.attrs["mu"])
            coords = f.attrs["coords"]
            data: Dict[int, Polynomial] = {}
            for name, ds in f["series"].items():
                deg = int(name)
                # Deserialize using pickle
                binary_data = ds[()].tobytes()
                expr = pickle.loads(binary_data)
                data[deg] = Polynomial(expr)
        return Hamiltonian(FormalSeries(data), mu, coords)

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"Hamiltonian(order≤{self.order}, coords='{self.coords}', μ={self.mu})"

    __repr__ = __str__
