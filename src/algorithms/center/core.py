import symengine as se
from functools import lru_cache
from collections import defaultdict


class Polynomial:
    """
    Thin wrapper around a SymEngine expression.
    """

    def __init__(self, variables: list[se.Symbol], expression: se.Basic):
        self.variables = variables
        self.expression = expression
        self._expansion = None
        self._grad_cache = None

    def __str__(self):
        return str(self.expression)

    def __repr__(self):
        return str(self.expression)

    def coefficient(self, variable: se.Symbol, order: int) -> se.Basic:
        return self.expression.coeff(variable, order)

    def __add__(self, other):
        return Polynomial(self.variables, self.expression + other.expression)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Polynomial(self.variables, self.expression - other.expression)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        return Polynomial(self.variables, self.expression * other.expression)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Polynomial(self.variables, self.expression / other.expression)

    def __rtruediv__(self, other):
        return self.__truediv__(other)
    
    def __neg__(self):
        return Polynomial(self.variables, -self.expression)

    def __eq__(self, other):
        return self.expression == other.expression

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
        return Polynomial(self.variables, self.expression.expand())

    def series_expand(self, variable: se.Symbol, degree: int) -> 'Polynomial':
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
        expr = self.expansion.expression
        if expr.is_Add:
            kept = [t for t in expr.args
                    if self._total_deg_term(t, self.variables) <= max_deg]
            new_expr = se.Add(*kept) if kept else se.Integer(0)
        else:
            new_expr = expr if self._total_deg_term(expr, self.variables) <= max_deg else se.Integer(0)
        return Polynomial(self.variables, new_expr)

    def evaluate(self, subs_dict: dict[se.Symbol, se.Basic]) -> se.Basic:
        return self.expression.subs(subs_dict)

    def derivative(self, variable: se.Symbol) -> 'Polynomial':
        return Polynomial(self.variables, self.expression.diff(variable))

    def _gradient(self):
        """
        Returns (dF/dq, dF/dp) lists.  Computed once, reused by every
        Poisson bracket that involves this Polynomial.
        """
        if self._grad_cache is None:
            n_vars = len(self.variables)
            n_dof = n_vars // 2
            q_syms = self.variables[::2]
            p_syms = self.variables[1::2]
            dF_dq = [self.expression.diff(q) for q in q_syms]
            dF_dp = [self.expression.diff(p) for p in p_syms]
            self._grad_cache = (dF_dq, dF_dp)
        return self._grad_cache


def _poisson_bracket(F: Polynomial, G: Polynomial) -> Polynomial:
    """
    Cached PB that reuses nabla_F and nabla_G when already computed.
    """
    dF_dq, dF_dp = F._gradient()
    dG_dq, dG_dp = G._gradient()

    expr = sum(dF_dq[i] * dG_dp[i] - dF_dp[i] * dG_dq[i]
               for i in range(len(dF_dq)))
    return Polynomial(F.variables, expr)


def _split_coeff_and_factors(term: se.Basic):
    """
    Return (numeric_coefficient , tuple_of_symbolic_factors).

    Works for SymEngine Expr; never uses as_coeff_* which SymEngine lacks.
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
    Insert all monomials of `poly` (already expanded) into the bucket map.
    """
    n_dof  = len(poly.variables) // 2
    q_vars = poly.variables[:n_dof]
    p_vars = poly.variables[n_dof:]

    for coeff, kq, kp in monomial_key(poly.expression, q_vars, p_vars):
        deg = sum(kq) + sum(kp)
        by_deg[deg].append((coeff, kq, kp))


def monomial_key(expr: se.Basic, q_vars: list[se.Symbol], p_vars: list[se.Symbol]):
    """
    Yield triples (coeff , kq , kp) for every monomial in *expr*.
    expr is assumed to be already expanded
    """
    n = len(q_vars)
    var2idx = {v: i            for i, v in enumerate(q_vars)}
    var2idx.update({v: i - n   for i, v in enumerate(p_vars, n)})

    terms = expr.args if expr.is_Add else (expr,)

    for term in terms:
        coeff, factors = _split_coeff_and_factors(term)

        kq = [0]*n
        kp = [0]*n

        # decompose the symbolic part
        for fac in factors:
            base, exp = fac.as_base_exp() if fac.is_Pow else (fac, 1)
            exp = int(exp)
            idx = var2idx.get(base)
            if idx is None:                 # parameter (µ, λ, …) stays in coeff
                coeff *= base**exp
                continue
            if idx >= 0:                    # q-family
                kq[idx] += exp
            else:                           # p-family
                kp[idx+n] += exp

        yield coeff, tuple(kq), tuple(kp)

def monomial_from_key(kq, kp):
    q1, q2, q3 = se.Symbol('q1'), se.Symbol('q2'), se.Symbol('q3')
    p1, p2, p3 = se.Symbol('p1'), se.Symbol('p2'), se.Symbol('p3')
    q_syms = [q1,q2,q3]
    p_syms = [p1,p2,p3]
    expr = se.Integer(1)
    for i,e in enumerate(kq): expr *= q_syms[i]**e
    for i,e in enumerate(kp): expr *= p_syms[i]**e
    return expr

def _lie_transform(H: Polynomial, max_degree: int = 6) -> Polynomial:
    """
    Normal-form Lie series up to 'max_degree' with incremental caches.
    """
    H_current = H.expansion
    by_deg = defaultdict(list)
    _update_by_deg(by_deg, H_current)   # ← build once

    # start at order 3 (quadratic part already normalised)
    for n in range(3, max_degree + 1):
        G_terms = by_deg.get(n, [])     # re-use cache, no rebuild
        if not G_terms:
            continue

        # assemble S_n ---------------------------------------------------
        S_expr = sum(coeff * monomial_from_key(kq, kp)
                     for coeff, kq, kp in G_terms
                     if kq[0] != kp[0])           # kill mixed hyperbolic terms

        if not S_expr:
            continue
        S = Polynomial(H.variables, S_expr)  # Correct parameter order

        # canonical transformation: H <- e^{L_S} H
        H_update   = _poisson_bracket(S, H_current).truncate(max_degree).expansion
        H_current  = (H_current + H_update).truncate(max_degree).expansion

        # only *new* terms need inserting into the cache
        _update_by_deg(by_deg, H_update)

    return H_current
