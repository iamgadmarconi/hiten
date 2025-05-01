import pytest
import numpy as np
import symengine as se
from collections import defaultdict


from algorithms.center.core import Polynomial, _poisson_bracket, _split_coeff_and_factors, _update_by_deg


# --- Pytest Fixtures ---

q1, q2, q3 = se.symbols('q1 q2 q3')
p1, p2, p3 = se.symbols('p1 p2 p3')

@pytest.fixture
def vars():
    return [q1, q2, q3, p1, p2, p3]

@pytest.fixture
def var_q1(vars):
    return vars[0]

@pytest.fixture
def var_p1(vars):
    return vars[3]

@pytest.fixture
def var_q2(vars):
    return vars[1]

@pytest.fixture
def var_p2(vars):
    return vars[4]

@pytest.fixture
def var_q3(vars):
    return vars[2]

@pytest.fixture
def var_p3(vars):
    return vars[5]

@pytest.fixture
def P_zero(vars):
    return Polynomial(vars, 0)

@pytest.fixture
def P_one(vars):
    return Polynomial(vars, 1)

@pytest.fixture
def P_q1(vars):
    return Polynomial(vars, q1)

@pytest.fixture
def P_p1(vars):
    return Polynomial(vars, p1)

@pytest.fixture
def P_q2(vars):
    return Polynomial(vars, q2)

@pytest.fixture
def P_p2(vars):
    return Polynomial(vars, p2)

@pytest.fixture
def P_q3(vars):
    return Polynomial(vars, q3)

@pytest.fixture
def P_p3(vars):
    return Polynomial(vars, p3)

@pytest.fixture
def P_qp_sum(P_q1, P_p1):
    return P_q1 + P_p1

@pytest.fixture
def P_complex(P_q1, P_p1, P_q2, P_p2, P_q3, P_p3):
    return (3*P_q1**2*P_p1 + 2*P_q2*P_p2 - 5* P_q3*P_p3**3).expansion 

# --- Test Functions ---

def test_initialization(vars, P_zero, P_one, P_q1):
    assert len(P_zero.variables) == len(vars)
    assert str(P_zero.expression) == '0'

    assert len(P_one.variables) == len(vars)
    assert str(P_one.expression) == '1'

    assert len(P_q1.variables) == len(vars)
    assert str(P_q1.expression) == 'q1'

    # Test initialization with symengine expressions
    x0 = se.Symbol('x0')
    p_with_expr = Polynomial(vars, x0 + 2)
    # The string representation may have different ordering (2 + x0 vs x0 + 2)
    assert p_with_expr.expression == x0 + 2
    # Test creation with numerical value
    p_with_num = Polynomial(vars, 42)
    assert str(p_with_num.expression) == '42'

def test_equality(P_zero, P_one, P_q1, P_p1, var_q1, var_p1):
    assert P_q1 == P_q1
    assert P_q1 != P_p1
    assert P_q1 != 0
    assert P_q1 != 1
    assert P_q1 == var_q1
    assert P_q1 != var_p1

def test_multiplication(P_zero, P_one, P_q1, P_p1, var_q1, var_p1):
    assert P_q1 * P_one == P_q1
    assert P_q1 * P_zero == P_zero
    assert P_q1 * P_p1 == P_q1 * var_p1
    assert P_q1 * 2 == 2 * P_q1
    assert 2 * P_q1 == 2 * var_q1
    assert P_q1 * P_q1 == var_q1 * var_q1
    assert P_q1 * P_p1 == var_q1 * var_p1
    assert P_p1 * se.I == var_p1 * se.I
    assert P_p1 * se.I ** 2 == -1 * var_p1

def test_addition_subtraction(P_zero, P_one, P_q1, P_p1, var_q1, var_p1):
    # Identities
    assert P_q1 + P_zero == P_q1
    assert P_q1 - P_q1 == P_zero
    assert P_q1 + P_one == P_q1 + 1
    assert P_q1 - P_one == P_q1 - 1
    assert P_q1 - P_p1 == var_q1 - var_p1

    # Commutativity
    assert P_q1 + P_p1 == P_p1 + P_q1

    assert P_q1 + 5 == var_q1 + 5
    assert 5 + P_q1 == var_q1 + 5
    assert P_q1 - 5 == var_q1 - 5
    assert 5 - P_q1 == 5 - var_q1

    # associativity
    assert (P_q1 + P_q1) + P_q1 == 3*var_q1
    assert P_q1 + (P_q1 + P_q1) == 3*var_q1

    # SymEngine expression
    expr = var_q1**2 + var_q1
    assert P_q1 + var_q1**2 == expr
    assert var_q1**2 + P_q1 == expr
    assert P_q1 - var_q1**2 == var_q1 - var_q1**2

    # Imaginary unit
    assert P_q1 + se.I == var_q1 + se.I
    assert P_q1 - se.I == var_q1 - se.I

def test_differentiation(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    # Test derivatives of constants
    assert P_zero.derivative(q1) == P_zero
    assert P_one.derivative(q1) == P_zero
    
    # Create polynomial with symbolic coefficient
    c = se.symbols('c')
    P_coeff = Polynomial(vars, c * q1 * q2)
    assert P_coeff.derivative(q1) == Polynomial(vars, c * q2)
    
    # Test derivative with respect to variable not in expression
    P_q1q2 = Polynomial(vars, q1 * q2)
    assert P_q1q2.derivative(q3) == P_zero
    assert P_q1q2.derivative(p3) == P_zero
    assert P_q1.derivative(q1) == P_one
    assert P_q1.derivative(p1) == P_zero
    assert P_p1.derivative(q1) == P_zero
    assert P_p1.derivative(p1) == P_one
    
    # Test squared terms
    P_q1_squared = P_q1 * P_q1
    assert P_q1_squared.derivative(q1) == 2 * P_q1
    
    # Test product of variables
    P_qp = P_q1 * P_p1
    assert P_qp.derivative(q1) == P_p1
    assert P_qp.derivative(p1) == P_q1

    # Test second derivative
    assert P_q1.derivative(q1).derivative(q1) == 0
    
    # Test derivatives of cubic terms
    P_q1_cubed = P_q1 * P_q1 * P_q1
    assert P_q1_cubed.derivative(q1) == 3 * P_q1 * P_q1
    assert P_q1_cubed.derivative(q1).derivative(q1) == 6 * P_q1

    # Test mixed terms
    P_mixed = P_q1 * P_q2 + P_p1 * P_p3
    assert P_mixed.derivative(q1) == P_q2
    assert P_mixed.derivative(q2) == P_q1
    assert P_mixed.derivative(p1) == P_p3
    assert P_mixed.derivative(p3) == P_p1
    
    # Test polynomial with multiple terms
    P_complex = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    assert P_complex.derivative(q1) == 4 * P_q1 + 3 * P_p1
    assert P_complex.derivative(p1) == 3 * P_q1
    assert P_complex.derivative(p2) == -10 * P_p2

    # Test calculus rules
    f = P_q1 * P_q1 + P_p1  # f = q1^2 + p1
    g = P_q1 * P_p1         # g = q1*p1
    
    # Test product rule: (f*g)' = f'*g + f*g'
    product = f * g
    assert product.derivative(q1) == f.derivative(q1) * g + f * g.derivative(q1)
    
    # Test sum rule: (f+g)' = f' + g'
    sum_poly = f + g
    assert sum_poly.derivative(q1) == f.derivative(q1) + g.derivative(q1)


def test_gradient(P_zero, P_one, P_q1, P_p1, P_p2):
    assert P_q1.gradient()[0][q1] == P_one
    assert P_p1.gradient()[1][p1] == P_one

    # Test gradient of a more complex polynomial
    P_complex = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    dF_dq, dF_dp = P_complex.gradient()
    assert dF_dq[q1] == 4 * P_q1 + 3 * P_p1
    assert dF_dp[p1] == 3 * P_q1
    assert dF_dp[p2] == -10 * P_p2


def test_poisson_bracket(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    assert _poisson_bracket(P_q1, P_q1) == P_zero

    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3

    # Calculate Poisson brackets and their expansions
    pb1 = _poisson_bracket(f, g)
    pb2 = _poisson_bracket(g, f)
    pb3 = _poisson_bracket(f, g + h)
    pb4 = _poisson_bracket(f, g*h)
    pb5 = _poisson_bracket(g, h)
    pb6 = _poisson_bracket(h, f)
    pb7 = _poisson_bracket(f, pb5)
    pb8 = _poisson_bracket(g, pb6)
    pb9 = _poisson_bracket(h, pb1)


    # Use our improved __neg__ method which properly distributes the negative sign
    assert pb1.expansion == (-pb2).expansion

    assert pb3.expansion == _poisson_bracket(f, g).expansion + _poisson_bracket(f, h).expansion

    assert pb4.expansion == (_poisson_bracket(f, g) * h).expansion + (g * _poisson_bracket(f, h)).expansion

    assert (pb7+pb8+pb9).expansion == 0

def test_series_expansion(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    assert f.sexpand(q1, 2).expansion == P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3

def test_degree(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_p3 ** 3
    assert f.variable_degree(q1) == 1
    assert f.variable_degree(p1) == 1
    assert f.variable_degree(q2) == 1
    assert f.variable_degree(p2) == 1
    assert f.variable_degree(q3) == 0
    assert f.variable_degree(p3) == 3

def test_total_degree(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3 ** 3
    assert f.total_degree() == 4

    assert P_q1.total_degree() == 1
    assert P_zero.total_degree() == 0
    assert P_one.total_degree() == 0

def test_truncate(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3 ** 3
    assert f.truncate(3).expansion == P_q1 * P_q2 * P_p1 + P_p1 * P_p2
    assert f.truncate(2).expansion == P_p1 * P_p2
    assert f.truncate(1).expansion == 0

def test_evaluate(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3 ** 3
    assert f.evaluate({q1: 1, q2: 2, q3: 3, p1: 4, p2: 5, p3: 6}) == 1*2*4 + 4*5 - 3*6**3

def test_split_coeff_and_factors(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3, P_complex):
    f = P_complex
    terms = f.expression.args if f.expression.is_Add else (f.expression,)
    #print(terms)
    coeff, factors = _split_coeff_and_factors(f.expression)
    #print(coeff, factors)

def test_update_by_deg(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3, P_complex):
    f = P_complex
    print(f.expression)
    by_deg = defaultdict(list)
    _update_by_deg(by_deg, f)
    print(by_deg)

