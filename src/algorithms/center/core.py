import symengine as se
from symengine import Symbol, series, I
from functools import lru_cache
from collections import defaultdict


class Polynomial:
    """
    Thin wrapper around a SymEngine expression.
    """

    def __init__(self, variables: list[se.Symbol], expression: se.Basic):
        self.variables = variables
        self.expression = expression

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
        expr = self.expression.expand()
        if expr.is_Add:
            return max(self._deg_in_var(t, var) for t in expr.args)
        return self._deg_in_var(expr, var)

    def total_degree(self) -> int:
        """
        Maximum total degree (sum of exponents) among all monomials.
        """
        expr = self.expression.expand()
        if expr.is_Add:
            return max(self._total_deg_term(t, self.variables) for t in expr.args)
        return self._total_deg_term(expr, self.variables)

    def truncate(self, var: se.Symbol, max_deg: int) -> "Polynomial":
        """
        Keep only terms where degree(var) <= max_deg.
        Useful when you reduce the Hamiltonian order-by-order.
        """
        expr = self.expression.expand()
        if expr.is_Add:
            kept = [t for t in expr.args if self._deg_in_var(t, var) <= max_deg]
            new_expr = se.Add(*kept) if kept else se.Integer(0)
        else:
            new_expr = expr if self._deg_in_var(expr, var) <= max_deg else se.Integer(0)
        return Polynomial(self.variables, new_expr)

    def evaluate(self, subs_dict: dict[se.Symbol, se.Basic]) -> se.Basic:
        return self.expression.subs(subs_dict)

    def derivative(self, variable: se.Symbol) -> 'Polynomial':
        return Polynomial(self.variables, self.expression.diff(variable))

    def expand(self) -> 'Polynomial':
        return Polynomial(self.variables, self.expression.expand())

    def series_expand(self, variable: se.Symbol, degree: int) -> 'Polynomial':
        return Polynomial(self.variables, series(self.expression, variable, degree))


@lru_cache(maxsize=10000)
def _poisson_bracket(poly1: Polynomial, poly2: Polynomial, variables: tuple[se.Symbol, ...]) -> Polynomial:
        n_vars = len(variables)
        num_dof = n_vars // 2
        result_expr = se.sympify(0)  # Start with zero expression

        for i in range(num_dof):
            qi_index = 2 * i
            pi_index = 2 * i + 1
            qi = variables[qi_index]
            pi = variables[pi_index]

            # Compute partial derivatives
            d_expr1_dqi = poly1.derivative(qi)
            d_expr2_dpi = poly2.derivative(pi)
            term1 = d_expr1_dqi.expression * d_expr2_dpi.expression

            d_expr1_dpi = poly1.derivative(pi)
            d_expr2_dqi = poly2.derivative(qi)
            term2 = d_expr1_dpi.expression * d_expr2_dqi.expression

            result_expr = result_expr + term1 - term2

        return Polynomial(list(variables), result_expr)

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

    # every other object counts as “symbolic” factor with coeff 1
    return se.Integer(1), (term,)
# -------------------------------------------------------------------------


def monomial_key(expr: se.Basic,
                 q_vars: list[se.Symbol],
                 p_vars: list[se.Symbol]):
    """
    Yield triples (coeff , kq , kp) for every monomial in *expr*.

    * q_vars – ordered list  [q1,q2,q3]
    * p_vars – ordered list  [p1,p2,p3]
    """
    n = len(q_vars)
    var2idx = {v: i            for i, v in enumerate(q_vars)}
    var2idx.update({v: i - n   for i, v in enumerate(p_vars, n)})

    expr = expr.expand()
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

@lru_cache(maxsize=10000)
def _lie_transform(H: Polynomial, max_degree: int) -> Polynomial:
    """
    Lie-series reduction of `H` up to total order `max_degree`.
    Assumes H.variables == [q1,q2,q3,p1,p2,p3] in that order and that
    H2 is already diagonal (eq. 13 in the paper).
    """
    q1,q2,q3,p1,p2,p3 = H.variables
    lam = se.Symbol('lam1')
    w1  = se.Symbol('w1')
    w2  = se.Symbol('w2')
    eta = (lam, se.I*w1, se.I*w2)

    # split H into dict degree -> Expr
    by_deg = defaultdict(lambda: se.Integer(0))
    for coeff, kq, kp in monomial_key(H.expression,
                                        q_vars=[q1,q2,q3],
                                        p_vars=[p1,p2,p3]):
        deg = sum(kq)+sum(kp)
        by_deg[deg] += coeff * (
            q1**kq[0]*q2**kq[1]*q3**kq[2] *
            p1**kp[0]*p2**kp[1]*p3**kp[2])

    H_current = H.expression
    for n in range(3, max_degree+1):
        Hn = by_deg[n]
        if Hn == 0:
            continue

        # --- build G_n
        G_terms = []
        for coeff, kq, kp in monomial_key(H_current,
                                        q_vars=[q1,q2,q3],
                                        p_vars=[p1,p2,p3]):
            if kq[0] != kp[0]:          # elimination condition
                denom = sum((kp[i]-kq[i])*eta[i] for i in range(3))
                G_terms.append((-coeff/denom) *
                                q1**kq[0]*q2**kq[1]*q3**kq[2] *
                                p1**kp[0]*p2**kp[1]*p3**kp[2])
        if not G_terms:
            continue
        G_n = se.Add(*G_terms)
        Gpoly = Polynomial(H.variables, G_n)

        # --- Lie transform truncated to order max_degree
        def truncate(expr):
            out = se.Integer(0)
            for c, kq, kp in monomial_key(expr,
                                        q_vars=[q1,q2,q3],
                                        p_vars=[p1,p2,p3]):
                if sum(kq)+sum(kp) <= max_degree:
                    out += c*(q1**kq[0]*q2**kq[1]*q3**kq[2] *
                                p1**kp[0]*p2**kp[1]*p3**kp[2])
            return out

        L_G = _poisson_bracket  # cached PB you already wrote

        delta = L_G(Polynomial(H.variables, H_current),
                    Gpoly, tuple(H.variables)).expression
        # second-order commutator is already ≥ n+1 so enough to add first term
        H_current = truncate(H_current + delta)

        # update degree dictionary for next rounds
        by_deg.clear()
        for c, kq, kp in monomial_key(H_current,
                                        q_vars=[q1,q2,q3],
                                        p_vars=[p1,p2,p3]):
            by_deg[sum(kq)+sum(kp)] += c*(q1**kq[0]*q2**kq[1]*q3**kq[2] *
                                            p1**kp[0]*p2**kp[1]*p3**kp[2])

    return Polynomial(H.variables, H_current)



q1 = Symbol('q1')
q2 = Symbol('q2')
q3 = Symbol('q3')
p1 = Symbol('p1')
p2 = Symbol('p2')
p3 = Symbol('p3')
l1 = Symbol('l1')
l2 = Symbol('l2')
w1 = Symbol('w1')
w2 = Symbol('w2')

H_total = l1 * q1 * p1 + I * w1 * q2 * p2 + I * w2 * q3 * p3
poly_H = Polynomial([q1,q2,q3,p1,p2,p3], H_total).expand()
print(poly_H)
H_bar  = _lie_transform(poly_H, max_degree=6)  # say up to sextic terms
print(H_bar)
