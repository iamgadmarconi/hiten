import math
import symengine as se
from functools import lru_cache
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Monomial:
    coeff: se.Basic
    kq: tuple[int]
    kp: tuple[int]
    sym: se.Basic        # the full q^k p^k expression

    def exponents_qp_split(self):
        return self.kq, self.kp


class Polynomial:
    """
    Thin wrapper around a SymEngine expression.
    """

    def __init__(self, variables: list[se.Symbol], expression: se.Basic):
        self.variables = variables
        self.expression = expression
        self._expansion = None
        self._by_degree_cache = None
        self._grad_cache = None
        self._monomials_cache = None

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return str(self.expression)

    def copy(self) -> 'Polynomial':
        """
        Create a deep copy of the polynomial with fresh caches.
        
        Returns
        -------
        Polynomial
            A new polynomial instance with the same variables and expression
            
        Notes
        -----
        This creates a new instance with the same variables and expression,
        but with all caches reset. This ensures that modifications to the
        copy don't affect the original polynomial.
        """
        return Polynomial(self.variables, self.expression)

    def coefficient(self, variable: se.Symbol, order: int) -> se.Basic:
        return self.expression.coeff(variable, order)

    def __add__(self, other):
        # Support Polynomial, SymEngine expression, and Python numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for addition.")
            expr = self.expression + other.expression
        elif isinstance(other, se.Basic):
            expr = self.expression + other
        elif isinstance(other, (int, float, complex)):
            expr = self.expression + se.sympify(other)
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        # Support Polynomial, SymEngine expression, and Python numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for subtraction.")
            expr = self.expression - other.expression
        elif isinstance(other, se.Basic):
            expr = self.expression - other
        elif isinstance(other, (int, float, complex)):
            expr = self.expression - se.sympify(other)
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)

    def __rsub__(self, other):
        # Right-hand subtraction for SymEngine expressions and numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for subtraction.")
            expr = other.expression - self.expression
        elif isinstance(other, se.Basic):
            expr = other - self.expression
        elif isinstance(other, (int, float, complex)):
            expr = se.sympify(other) - self.expression
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)

    def __mul__(self, other):
        # Support Polynomial, SymEngine expression, and Python numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for multiplication.")
            expr = self.expression * other.expression
        elif isinstance(other, se.Basic):
            expr = self.expression * other
        elif isinstance(other, (int, float, complex)):
            expr = self.expression * se.sympify(other)
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Support Polynomial, SymEngine expression, and Python numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for division.")
            expr = self.expression / other.expression
        elif isinstance(other, se.Basic):
            expr = self.expression / other
        elif isinstance(other, (int, float, complex)):
            expr = self.expression / se.sympify(other)
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)

    def __rtruediv__(self, other):
        # Right-hand division for SymEngine expressions and numeric types
        if isinstance(other, Polynomial):
            if self.variables != other.variables:
                raise ValueError("Polynomial variables must match for division.")
            expr = other.expression / self.expression
        elif isinstance(other, se.Basic):
            expr = other / self.expression
        elif isinstance(other, (int, float, complex)):
            expr = se.sympify(other) / self.expression
        else:
            return NotImplemented
        return Polynomial(self.variables, expr)
    
    def __neg__(self):
        return Polynomial(self.variables, -self.expression)

    def __eq__(self, other):
        # Support equality with Polynomial, SymEngine expression, and numeric types
        if isinstance(other, Polynomial):
            # If both expressions evaluate to zero, consider them equal regardless of variables
            if self.expression == 0 and other.expression == 0:
                return True
            return self.variables == other.variables and self.expression == other.expression
        elif isinstance(other, se.Basic):
            return self.expression == other
        elif isinstance(other, (int, float, complex)):
            return self.expression == se.sympify(other)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, int):
            return Polynomial(self.variables, self.expression ** other)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.expression)

    @property
    def expansion(self) -> 'Polynomial':
        if not self._expansion:
            expanded = self._expand()
            expanded._expansion = expanded
            self._expansion = expanded
        return self._expansion

    def _expand(self) -> 'Polynomial':
        # Handle case when expression is a Python primitive type
        if isinstance(self.expression, (int, float)):
            return Polynomial(self.variables, se.sympify(self.expression))
        return Polynomial(self.variables, self.expression.expand())

    def sexpand(self, variable: se.Symbol, degree: int) -> 'Polynomial':
        return Polynomial(self.variables, se.series(self.expression, variable, degree))

    @staticmethod
    def _deg_in_var(term: se.Basic, var: se.Symbol) -> int:
        """Degree of *var* in a single monomial *term* (term must NOT be an Add)."""
        if term.is_Number:                     #  constant
            return 0
        if term == var:                        #  just x
            return 1
        if term.is_Symbol:                     #  some other symbol
            return 0
        if term.is_Pow:                        #  x**n , n integer
            base, exp = term.args
            if base == var and exp.is_integer:
                return int(exp)
            return 0
        if term.is_Mul:                        #  product of factors
            return sum(Polynomial._deg_in_var(f, var) for f in term.args)
        raise ValueError(f"Non-polynomial term encountered: {term}")

    @staticmethod
    def _total_deg_term(term: se.Basic, vars_: list[se.Symbol]) -> int:
        """Total degree of a single monomial."""
        return sum(Polynomial._deg_in_var(term, v) for v in vars_)

    def variable_degree(self, var: se.Symbol) -> int:
        """max_k  such that  var**k  divides some term."""
        expr = self.expansion.expression
        if expr.is_Add:
            return max(self._deg_in_var(t, var) for t in expr.args)
        return self._deg_in_var(expr, var)

    def total_degree(self) -> int:
        """
        Maximum total degree (sum of exponents) among all monomials.
        """
        expr = self.expansion.expression
        if expr.is_Add:
            return max(self._total_deg_term(t, self.variables) for t in expr.args)
        return self._total_deg_term(expr, self.variables)

    def truncate_by_var(self, var: se.Symbol, max_deg: int) -> "Polynomial":
        """
        Keep only terms where degree(var) <= max_deg.
        Useful when you reduce the Hamiltonian order-by-order.
        """
        # Handle case when expression is a Python primitive type
        if isinstance(self.expression, (int, float)):
            return Polynomial(self.variables, se.sympify(self.expression))
            
        expr = self.expansion.expression
        if expr.is_Add:
            kept = [t for t in expr.args if self._deg_in_var(t, var) <= max_deg]
            new_expr = se.Add(*kept) if kept else se.Integer(0)
        else:
            new_expr = expr if self._deg_in_var(expr, var) <= max_deg else se.Integer(0)
        return Polynomial(self.variables, new_expr)

    def truncate(self, max_deg: int) -> "Polynomial":
        """
        Remove every monomial whose **total degree** exceeds `max_deg`.
        """
        # Handle case when expression is a Python primitive type
        if isinstance(self.expression, (int, float)):
            return Polynomial(self.variables, se.sympify(self.expression))
            
        expr = self.expansion.expression
        if expr.is_Add:
            kept = [t for t in expr.args
                    if self._total_deg_term(t, self.variables) <= max_deg]
            new_expr = se.Add(*kept) if kept else se.Integer(0)
        else:
            new_expr = expr if self._total_deg_term(expr, self.variables) <= max_deg else se.Integer(0)
        
        # Clear all caches to ensure they are rebuilt with the new truncated expression
        self._expansion = None
        self._by_degree_cache = None
        self._grad_cache = None
        self._monomials_cache = None
        self.expression = new_expr

        return Polynomial(self.variables, new_expr)

    def evaluate(self, subs_dict: dict[se.Symbol, se.Basic]) -> se.Basic:
        return self.expression.subs(subs_dict)

    def derivative(self, variable) -> 'Polynomial':
        """
        Compute the derivative of this polynomial with respect to the given variable.
        
        Parameters:
            variable: Can be a Polynomial, SymEngine symbol, or numeric type
            
        Returns:
            A new Polynomial representing the derivative
        """
        # Handle case when expression is a Python number (int, float)
        if isinstance(self.expression, (int, float)):
            return Polynomial(self.variables, se.sympify(0))
        
        # Support Polynomial, SymEngine expression, and Python numeric types
        if isinstance(variable, Polynomial):
            if self.variables != variable.variables:
                raise ValueError("Polynomial variables must match for differentiation.")
            return Polynomial(self.variables, self.expression.diff(variable.expression))
        elif isinstance(variable, se.Basic):
            derived = self.expression.diff(variable)
            # Ensure we return exactly zero when the derivative is zero
            return Polynomial(self.variables, derived if derived else se.sympify(0))
        elif isinstance(variable, (int, float, complex)):
            # Differentiating with respect to a constant always gives zero
            return Polynomial(self.variables, se.sympify(0))
        else:
            return NotImplemented

    def _gradient(self):
        """
        Returns (dF/dq, dF/dp) lists.  Computed once, reused by every
        Poisson bracket that involves this Polynomial.
        """
        if self._grad_cache is None:
            n_vars = len(self.variables)
            n_dof = n_vars // 2
            q_syms = self.variables[:n_dof]
            p_syms = self.variables[n_dof:]
            
            # Make sure we're working with expanded expressions for more efficient differentiation
            expr = self.expansion.expression
            
            # For sparse polynomials, term-by-term differentiation may be faster
            if expr.is_Add and len(expr.args) < n_dof:
                dF_dq = []
                dF_dp = []
                for i in range(n_dof):
                    q_derivative = sum((term.diff(q_syms[i]) for term in expr.args if term.has(q_syms[i])), se.Integer(0))
                    p_derivative = sum((term.diff(p_syms[i]) for term in expr.args if term.has(p_syms[i])), se.Integer(0))
                    dF_dq.append(q_derivative)
                    dF_dp.append(p_derivative)
            else:
                # Standard differentiation for dense expressions
                dF_dq = [expr.diff(q) for q in q_syms]
                dF_dp = [expr.diff(p) for p in p_syms]
            
            self._grad_cache = (dF_dq, dF_dp)
        return self._grad_cache

    def gradient(self):
        """
        Computes the gradient of the polynomial with respect to all q and p variables.
        
        Returns:
            A tuple of two dictionaries: (dF_dq, dF_dp) where:
            - dF_dq maps q variables to their derivatives
            - dF_dp maps p variables to their derivatives
        """
        dF_dq_raw, dF_dp_raw = self._gradient()
        
        n_vars = len(self.variables)
        n_dof = n_vars // 2
        q_syms = self.variables[:n_dof]
        p_syms = self.variables[n_dof:]
        
        # Create dictionaries mapping variables to their derivatives
        dF_dq = {}
        dF_dp = {}
        
        for i, q in enumerate(q_syms):
            dF_dq[q] = Polynomial(self.variables, dF_dq_raw[i])
        
        for i, p in enumerate(p_syms):
            dF_dp[p] = Polynomial(self.variables, dF_dp_raw[i])
        
        return dF_dq, dF_dp

    def build_by_degree(self) -> dict[int, list[Monomial]]:
        """
        Return {deg: [Monomial, …]} for the expanded polynomial.
        
        Returns
        -------
        dict[int, list[Monomial]]
            A dictionary mapping degrees to lists of Monomial objects
            
        Notes
        -----
        This method organizes the monomials of the polynomial by their total degree.
        The total degree of a monomial is the sum of all exponents of its variables.
        """
        if self._by_degree_cache is None:
            by_deg: dict[int, list] = defaultdict(list)
            _update_by_deg(by_deg, self)
            self._by_degree_cache = by_deg
        return self._by_degree_cache

    def get_monomials(self) -> list[Monomial]:
        """
        Extract all monomials from the polynomial.
        
        Returns
        -------
        list[Monomial]
            A list of Monomial objects representing each term in the polynomial
            
        Notes
        -----
        This method expands the polynomial first to ensure all terms are separated,
        then extracts each term as a Monomial object with coefficient, exponents,
        and symbolic representation.
        
        Examples
        --------
        >>> poly = Polynomial(vars, 3*q1**2*p1 + 2*q2*p2)
        >>> monomials = poly.get_monomials()
        >>> len(monomials)
        2
        >>> [m.sym for m in monomials]
        [3*q1**2*p1, 2*q2*p2]
        """
        if self._monomials_cache is None:
            n_dof = len(self.variables) // 2
            q_vars = self.variables[:n_dof]
            p_vars = self.variables[n_dof:]
            
            # Use _monomial_key to decompose the expanded expression
            self._monomials_cache = list(_monomial_key(self.expansion.expression, q_vars, p_vars))
            
        return self._monomials_cache

    @staticmethod
    def from_monomials(variables: list[se.Symbol], monomials: list[Monomial]) -> 'Polynomial':
        """
        Create a Polynomial from a list of Monomial objects.
        
        Parameters
        ----------
        variables : list[se.Symbol]
            The variables of the polynomial
        monomials : list[Monomial]
            The monomials to include in the polynomial
            
        Returns
        -------
        Polynomial
            A new Polynomial constructed from the given monomials
            
        Notes
        -----
        This method allows direct construction of a polynomial from 
        a list of Monomial objects, which is useful when manipulating
        polynomials term by term.
        
        Examples
        --------
        >>> m1 = Monomial(coeff=3, kq=(2,0,0), kp=(1,0,0), sym=3*q1**2*p1)
        >>> m2 = Monomial(coeff=2, kq=(0,1,0), kp=(0,1,0), sym=2*q2*p2)
        >>> poly = Polynomial.from_monomials(vars, [m1, m2])
        >>> str(poly)
        '3*q1**2*p1 + 2*q2*p2'
        """
        if not monomials:
            return Polynomial(variables, se.Integer(0))
        
        # Sum all the symbolic expressions of the monomials
        expr = sum(monomial.sym for monomial in monomials)
        return Polynomial(variables, expr)

    def iter_terms(self):
        """
        Iterate through the terms of the polynomial, yielding (coefficient, monomial) pairs.
        
        Yields
        ------
        tuple
            A tuple of (coefficient, monomial) for each term in the polynomial
            
        Notes
        -----
        This method is used for manipulations where both the coefficient and
        the monomial structure (with its exponents) are needed separately.
        """
        for monomial in self.get_monomials():
            yield monomial.coeff, monomial

    def poisson(self, other: 'Polynomial') -> 'Polynomial':
        """
        Compute the Poisson bracket {self, other}.
        
        Parameters
        ----------
        other : Polynomial
            The second polynomial for the Poisson bracket
            
        Returns
        -------
        Polynomial
            A new polynomial representing {self, other}
            
        Notes
        -----
        The Poisson bracket is defined as:
        {F, G} = sum_i (dF/dqi * dG/dpi - dF/dpi * dG/dqi)
        """
        return _poisson_bracket(self, other)

    def optimized_poisson(self, other: 'Polynomial', method: str = 'auto', use_cache: bool = True) -> 'Polynomial':
        """
        Compute the Poisson bracket {self, other} using optimized methods.
        
        Parameters
        ----------
        other : Polynomial
            The second polynomial for the Poisson bracket
        method : str, optional
            The method to use for computation:
            - 'auto': Automatically select the best method
            - 'standard': Use the standard implementation
            - 'term_by_term': Use term-by-term differentiation
        use_cache : bool, optional
            Whether to use memoization for repeated calculations
            
        Returns
        -------
        Polynomial
            A new polynomial representing {self, other}
            
        Notes
        -----
        Different methods may be more efficient for different types of polynomials:
        - Standard method: Good for dense polynomials with few variables
        - Term-by-term: Better for sparse polynomials with many variables
        - Memoization: Beneficial when the same brackets are computed repeatedly
        """
        if use_cache:
            # Convert to hashable expressions for caching
            hash_F = Polynomial(self.variables, self.expression)
            hash_G = Polynomial(other.variables, other.expression)
            return Polynomial.memoized_poisson(hash_F, hash_G)
        
        if method == 'term_by_term':
            return _poisson_bracket_term_by_term(self, other)
        elif method == 'standard':
            return _poisson_bracket(self, other)
        else:  # 'auto'
            # Simple heuristic: use term-by-term for sparse polynomials
            F_monomials = len(self.expansion.get_monomials())
            G_monomials = len(other.expansion.get_monomials())
            n_vars = len(self.variables) // 2
            
            # If both polynomials are sparse relative to the number of variables
            if F_monomials * G_monomials < 4 * n_vars**2:
                return _poisson_bracket_term_by_term(self, other)
            else:
                return _poisson_bracket(self, other)

    @staticmethod
    @lru_cache(maxsize=128)
    def memoized_poisson(F: 'Polynomial', G: 'Polynomial') -> 'Polynomial':
        """
        Compute the Poisson bracket with result caching.
        
        This method caches the result of Poisson brackets between 
        polynomial pairs, which can significantly speed up calculations
        when the same brackets are computed multiple times.
        
        Parameters
        ----------
        F, G : Polynomial
            The polynomials for which to compute {F, G}
            
        Returns
        -------
        Polynomial
            The Poisson bracket result
        """
        return _poisson_bracket(F, G)

    def subs(self, subs_dict: dict[se.Symbol, se.Basic]) -> 'Polynomial':
        """
        Substitute variables in the polynomial using a dictionary.
        
        Parameters
        ----------
        subs_dict : dict[se.Symbol, se.Basic]
            A dictionary mapping variables to their substitutions
            
        Returns
        -------
        Polynomial
            A new polynomial with the substitutions applied
        """
        old_expr = se.sympify(self.expression)
        new_expr = old_expr.subs(subs_dict)
        return Polynomial(self.variables, new_expr)


def _poisson_bracket(F: Polynomial, G: Polynomial) -> Polynomial:
    """
    Cached PB that reuses nabla_F and nabla_G when already computed.
    """
    assert F.variables == G.variables, "Variables must match for Poisson bracket"
    dF_dq, dF_dp = F._gradient()
    dG_dq, dG_dp = G._gradient()

    expr = sum(dF_dq[i] * dG_dp[i] - dF_dp[i] * dG_dq[i]
                for i in range(len(dF_dq)))
    return Polynomial(F.variables, expr)


def _poisson_bracket_term_by_term(F: Polynomial, G: Polynomial) -> Polynomial:
    """
    Compute Poisson bracket by differentiating term-by-term.
    
    This implementation may be more efficient for sparse polynomials
    where many terms have zero derivatives.
    
    Parameters
    ----------
    F, G : Polynomial
        The polynomials for which to compute {F, G}
        
    Returns
    -------
    Polynomial
        The Poisson bracket result
    """
    assert F.variables == G.variables, "Variables must match for Poisson bracket"
    n_vars = len(F.variables)
    n_dof = n_vars // 2
    
    # Get expanded polynomials
    F_exp = F.expansion
    G_exp = G.expansion
    
    # Get monomials for term-by-term differentiation
    F_monomials = F_exp.get_monomials()
    G_monomials = G_exp.get_monomials()
    
    result_terms = []
    
    # Process each pair of monomials
    for F_mono in F_monomials:
        F_term = Polynomial(F.variables, F_mono.sym)
        dF_dq, dF_dp = F_term._gradient()
        
        for G_mono in G_monomials:
            G_term = Polynomial(G.variables, G_mono.sym)
            dG_dq, dG_dp = G_term._gradient()
            
            # Compute the bracket for this pair of terms
            term_expr = sum(dF_dq[i] * dG_dp[i] - dF_dp[i] * dG_dq[i]
                          for i in range(n_dof))
            
            if term_expr != 0:  # Only add non-zero terms
                result_terms.append(term_expr)
    
    # Sum all resulting terms
    if not result_terms:
        return Polynomial(F.variables, se.Integer(0))
    
    return Polynomial(F.variables, sum(result_terms))


def _split_coeff_and_factors(term: se.Basic):
    """
    Split a symbolic expression term into its numeric coefficient and symbolic factors.

    Parameters
    ----------
    term : se.Basic
        A SymEngine expression term (can be Number, Symbol, Mul, Pow, etc.)

    Returns
    -------
    tuple
        A tuple (coefficient, symbolic_factors) where:
        - coefficient is a SymEngine number
        - symbolic_factors is a tuple of symbolic expressions
        
    Notes
    -----
    This function handles three cases:
    - For numeric terms (e.g., 7), returns (term, ())
    - For Mul terms (e.g., 3*q1**2*p2), splits out numeric parts as coefficient
    - For other symbolic terms, returns (1, (term,))
    
    Examples
    --------
    >>> _split_coeff_and_factors(se.Integer(7))
    (7, ())
    >>> _split_coeff_and_factors(q1)
    (1, (q1,))
    >>> _split_coeff_and_factors(3*q1**2*p2)
    (3, (q1**2, p2))
    """
    if term.is_Number:                  # e.g.  7
        return term, ()

    if term.is_Mul:                     # e.g.  3*q1**2*p2
        coeff = se.Integer(1)
        symbolic = []
        for fac in term.args:
            if fac.is_Number:
                coeff *= fac
            else:
                symbolic.append(fac)
        return coeff, tuple(symbolic)

    # every other object counts as "symbolic" factor with coeff 1
    return se.Integer(1), (term,)


def _update_by_deg(by_deg: dict[int, list], poly: Polynomial) -> None:
    """
    Insert all monomials of a polynomial into a dictionary organized by total degree.
    
    Parameters
    ----------
    by_deg : dict[int, list]
        A dictionary mapping degrees to lists of monomials
    poly : Polynomial
        The polynomial whose monomials will be added to the dictionary
        
    Returns
    -------
    None
        The function modifies the by_deg dictionary in place
        
    Notes
    -----
    Each monomial is stored in the dictionary as a Monomial object containing:
    - coeff: the numeric coefficient
    - kq: tuple of exponents for position variables (q)
    - kp: tuple of exponents for momentum variables (p)
    - sym: the full q^k p^k expression
    
    The key in the dictionary is the total degree (sum of all exponents)
    """
    n_dof  = len(poly.variables) // 2
    q_vars = poly.variables[:n_dof]
    p_vars = poly.variables[n_dof:]

    expanded_expr = poly.expansion.expression

    for monomial in _monomial_key(expanded_expr, q_vars, p_vars): # Use expanded_expr here
        deg = sum(monomial.kq) + sum(monomial.kp)
        by_deg[deg].append(monomial)

def _monomial_key(expr: se.Basic,
                q_vars: list[se.Symbol],
                p_vars: list[se.Symbol]):
    """
    Decompose an expression into its constituent monomials with separated q and p variable exponents.
    
    Parameters
    ----------
    expr : se.Basic
        An **expanded** SymEngine expression to decompose
    q_vars : list[se.Symbol]
        List of position variables (q)
    p_vars : list[se.Symbol]
        List of momentum variables (p)
        
    Yields
    ------
    Monomial
        A Monomial instance for each term in the expression, containing:
        - coeff: numeric coefficient
        - kq: tuple of exponents for each q variable
        - kp: tuple of exponents for each p variable
        - sym: the full q^k p^k expression
        
    Notes
    -----
    This function:
    1. Maps variables to their indices
    2. Iterates through terms in the expression
    3. Decomposes each term into coefficient and factors
    4. Processes each factor to extract variable exponents
    5. Yields Monomial instances
    
    Any symbolic terms not in q_vars or p_vars are treated as parameters
    and incorporated into the coefficient.
    
    Examples
    --------
    >>> list(_monomial_key(3*q1**2*p1 + 2*q2*p2, [q1,q2,q3], [p1,p2,p3]))
    [Monomial(coeff=3, kq=(2,0,0), kp=(1,0,0), sym=3*q1**2*p1), 
     Monomial(coeff=2, kq=(0,1,0), kp=(0,1,0), sym=2*q2*p2)]
    """
    n = len(q_vars)
    q2idx = {v: i for i, v in enumerate(q_vars)}     #  0 … n-1
    p2idx = {v: i for i, v in enumerate(p_vars)}     #  0 … n-1

    terms = expr.args if expr.is_Add else (expr,)

    for term in terms:
        coeff, factors = _split_coeff_and_factors(term)
        kq = [0]*n
        kp = [0]*n

        for fac in factors:                          # fac = x   or   x**e
            base, exp = fac.as_base_exp() if fac.is_Pow else (fac, 1)
            exp = int(exp)

            if base in q2idx:                        # q-family
                kq[q2idx[base]] += exp
            elif base in p2idx:                      # p-family
                kp[p2idx[base]] += exp
            else:                                    # parameter (µ, λ, …)
                coeff *= base**exp

        kq_tuple = tuple(kq)
        kp_tuple = tuple(kp)
        
        # Create the symbolic expression for the monomial
        sym_expr = coeff * _monomial_from_key(kq_tuple, kp_tuple, q_vars, p_vars)
        
        yield Monomial(coeff=coeff, kq=kq_tuple, kp=kp_tuple, sym=sym_expr)

def _monomial_from_key(kq, kp, q_syms, p_syms):
    """
    Reconstruct a symbolic monomial from exponent tuples for q and p variables.
    
    Parameters
    ----------
    kq : tuple
        Tuple of exponents for position variables (q1, q2, q3)
    kp : tuple
        Tuple of exponents for momentum variables (p1, p2, p3)
        
    Returns
    -------
    se.Basic
        A SymEngine expression representing the monomial with the given exponents
        
    Notes
    -----
    This function is the inverse of _monomial_key. It takes the exponent representation
    of a monomial and reconstructs the symbolic expression by raising each variable
    to its corresponding exponent.
    
    This is particularly useful in normal form calculations where terms are processed
    by degree and then need to be reconstructed into symbolic expressions.
    
    Examples
    --------
    >>> _monomial_from_key((2, 1, 0), (0, 0, 3))
    q1**2*q2*p3**3
    
    >>> _monomial_from_key((1, 0, 0), (1, 0, 0))
    q1*p1
    """
    expr = se.Integer(1)
    for i,e in enumerate(kq): expr *= q_syms[i]**e
    for i,e in enumerate(kp): expr *= p_syms[i]**e
    return expr

def _dot_product(v1: tuple[int], v2: list[se.Basic]) -> se.Basic:
    """Computes the dot product of an integer tuple and a list of SymEngine expressions."""
    # Assuming v1 and v2 have compatible dimensions
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension for dot product.")
    
    result = se.Integer(0)
    for i in range(len(v1)):
        result += se.sympify(v1[i]) * v2[i] # Ensure integer is sympified for multiplication
    return result
