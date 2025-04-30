import symengine as se
from symengine import Symbol
from functools import lru_cache


class Polynomial:


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

    def evaluate(self, subs_dict: dict[se.Symbol, se.Basic]) -> se.Basic:
        return self.expression.subs(subs_dict)

    def derivative(self, variable: se.Symbol) -> 'Polynomial':
        return Polynomial(self.variables, self.expression.diff(variable))

    def expand(self):
        return self.expression.expand()

    def series_expand(self):
        pass


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


def _lie_bracket(poly1: Polynomial, poly2: Polynomial, variables: tuple[se.Symbol, ...]) -> Polynomial:
    return _poisson_bracket(poly1, poly2, variables)


x = Symbol('x')
y = Symbol('y')
z = Symbol('z')
px = Symbol('px')
py = Symbol('py')
pz = Symbol('pz')

variables = [x,y,z,px,py,pz]

expression1 = (x+y)**2+y

expression2 = x*y

expression3 = 2

poly1 = Polynomial(variables, expression1)

poly2 = Polynomial(variables, expression2)

print(poly1 == poly2)

poly3 = Polynomial(variables, expression3)

poly4 = poly1 + poly2 / poly3

print(poly4)

print(poly4.expand())

s_dict = {x: 1, y: 2, z: 3, px: 4, py: 5, pz: 6}

print(poly4.evaluate(s_dict))

print(poly4.derivative(x))

poly5 = _poisson_bracket(poly1, poly2, tuple(variables))

print(poly5)


