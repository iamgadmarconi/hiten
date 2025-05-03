import pytest
import numpy as np
import symengine as se
from collections import defaultdict

from algorithms.center.core import (Polynomial, _poisson_bracket, 
                                    _split_coeff_and_factors, _update_by_deg, 
                                    _monomial_key, _monomial_from_key, _dot_product)
from algorithms.variables import get_vars, canonical_normal_vars


q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)

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

@pytest.fixture
def P_high_order(P_q1, P_p1, P_q2, P_p2, P_q3, P_p3):
    return (P_q1**7*P_p1 + P_q2**6*P_p2 + P_q3**5*P_p3+ P_q3**2*7).expansion


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

def test_split_coeff_and_factors():
    # Test with a number
    coeff, factors = _split_coeff_and_factors(se.Integer(5))
    assert coeff == 5
    assert factors == ()
    
    # Test with a symbol
    coeff, factors = _split_coeff_and_factors(q1)
    assert coeff == 1
    assert factors == (q1,)
    
    # Test with a product of symbols
    coeff, factors = _split_coeff_and_factors(q1 * q2)
    assert coeff == 1
    assert set(factors) == {q1, q2}  # Order might vary
    
    # Test with coefficient and symbols
    coeff, factors = _split_coeff_and_factors(3 * q1 * p1)
    assert coeff == 3
    assert set(factors) == {q1, p1}
    
    # Test with powers
    coeff, factors = _split_coeff_and_factors(2 * q1**2 * p1)
    assert coeff == 2
    assert q1**2 in factors
    assert p1 in factors
    
    # Test with imaginary unit
    coeff, factors = _split_coeff_and_factors(se.I * q1)
    assert coeff == se.I
    assert factors == (q1,)

def test_monomial_key():
    # Setup variables
    q_vars = [q1, q2, q3]
    p_vars = [p1, p2, p3]
    
    # Test with a simple monomial
    expr = q1 * p1
    result = list(_monomial_key(expr, q_vars, p_vars))
    assert len(result) == 1
    monomial = result[0]
    assert monomial.coeff == 1
    assert monomial.kq == (1,0,0)
    assert monomial.kp == (1,0,0)
    assert monomial.sym == q1*p1
    
    # Test with a sum of monomials
    expr = 3 * q1**2 * p1 + 2 * q2 * p2
    result = list(_monomial_key(expr, q_vars, p_vars))
    assert len(result) == 2
    
    # Find the specific terms
    term1 = None
    term2 = None
    for item in result:
        if item.coeff == 3:
            term1 = item
        elif item.coeff == 2:
            term2 = item
    
    assert term1 is not None
    assert term2 is not None
    
    # Should be 3*q1^2*p1
    assert term1.coeff == 3
    assert term1.kq == (2,0,0)
    assert term1.kp == (1,0,0)
    assert term1.sym == 3*q1**2*p1
    
    # Should be 2*q2*p2
    assert term2.coeff == 2
    assert term2.kq == (0,1,0)
    assert term2.kp == (0,1,0)
    assert term2.sym == 2*q2*p2
    
    # Test with powers and negative coefficients
    expr = q1**3 - 5 * p3**2 * q2
    result = list(_monomial_key(expr, q_vars, p_vars))
    assert len(result) == 2
    
    term1 = None
    term2 = None
    for item in result:
        if item.coeff == -5:
            term1 = item
        elif item.coeff == 1:
            term2 = item
    
    assert term1 is not None
    assert term2 is not None
    
    # Should be -5*q2*p3^2
    assert term1.coeff == -5
    assert term1.kq == (0,1,0)
    assert term1.kp == (0,0,2)
    assert term1.sym == -5*q2*p3**2
    
    # Should be q1^3
    assert term2.coeff == 1
    assert term2.kq == (3,0,0)
    assert term2.kp == (0,0,0)
    assert term2.sym == q1**3
    
    # Test with parameters (symbols not in q_vars or p_vars)
    mu = se.symbols('mu')
    expr = mu * q1 * p1
    result = list(_monomial_key(expr, q_vars, p_vars))
    assert len(result) == 1
    monomial = result[0]
    assert monomial.coeff == mu
    assert monomial.kq == (1,0,0)
    assert monomial.kp == (1,0,0)
    assert monomial.sym == mu*q1*p1

def test_update_by_deg(vars):
    # Test with a simple monomial
    by_deg = defaultdict(list)
    f = Polynomial(vars, q1 * p1)
    _update_by_deg(by_deg, f)
    assert len(by_deg) == 1
    assert 2 in by_deg  # Total degree is 2
    
    # Check monomial properties
    monomial = by_deg[2][0]
    assert monomial.coeff == 1
    assert monomial.kq == (1,0,0)
    assert monomial.kp == (1,0,0)
    assert monomial.sym == q1*p1
    
    # Test with multiple terms of different degrees
    by_deg = defaultdict(list)
    f = Polynomial(vars, q1**2 * p1 + q2 * p2)
    _update_by_deg(by_deg, f.expansion)
    assert len(by_deg) == 2
    assert 3 in by_deg  # q1^2*p1 has degree 3
    assert 2 in by_deg  # q2*p2 has degree 2
    
    # Find specific monomials
    monomial_deg3 = None
    monomial_deg2 = None
    
    for monomial in by_deg[3]:
        if monomial.kq == (2,0,0) and monomial.kp == (1,0,0):
            monomial_deg3 = monomial
    
    for monomial in by_deg[2]:
        if monomial.kq == (0,1,0) and monomial.kp == (0,1,0):
            monomial_deg2 = monomial
    
    assert monomial_deg3 is not None
    assert monomial_deg2 is not None
    
    # Verify specific terms
    assert monomial_deg3.coeff == 1
    assert monomial_deg3.kq == (2,0,0)
    assert monomial_deg3.kp == (1,0,0)
    assert monomial_deg3.sym == q1**2*p1
    
    assert monomial_deg2.coeff == 1
    assert monomial_deg2.kq == (0,1,0)
    assert monomial_deg2.kp == (0,1,0)
    assert monomial_deg2.sym == q2*p2
    
    # Test with numeric coefficients
    by_deg = defaultdict(list)
    f = Polynomial(vars, 3 * q1**2 * p1 + 2 * q2 * p2 - 5 * q3 * p3**3)
    _update_by_deg(by_deg, f.expansion)
    assert len(by_deg) == 3
    assert 3 in by_deg  # 3*q1^2*p1 has degree 3
    assert 2 in by_deg  # 2*q2*p2 has degree 2
    assert 4 in by_deg  # -5*q3*p3^3 has degree 4
    
    # Find specific monomials
    monomial_deg3 = None
    monomial_deg2 = None
    monomial_deg4 = None
    
    for monomial in by_deg[3]:
        if monomial.kq == (2,0,0) and monomial.kp == (1,0,0):
            monomial_deg3 = monomial
    
    for monomial in by_deg[2]:
        if monomial.kq == (0,1,0) and monomial.kp == (0,1,0):
            monomial_deg2 = monomial
    
    for monomial in by_deg[4]:
        if monomial.kq == (0,0,1) and monomial.kp == (0,0,3):
            monomial_deg4 = monomial
    
    assert monomial_deg3 is not None
    assert monomial_deg2 is not None
    assert monomial_deg4 is not None
    
    # Verify specific terms
    assert monomial_deg3.coeff == 3
    assert monomial_deg3.kq == (2,0,0)
    assert monomial_deg3.kp == (1,0,0)
    assert monomial_deg3.sym == 3*q1**2*p1
    
    assert monomial_deg2.coeff == 2
    assert monomial_deg2.kq == (0,1,0)
    assert monomial_deg2.kp == (0,1,0)
    assert monomial_deg2.sym == 2*q2*p2
    
    assert monomial_deg4.coeff == -5
    assert monomial_deg4.kq == (0,0,1)
    assert monomial_deg4.kp == (0,0,3)
    assert monomial_deg4.sym == -5*q3*p3**3

def test_build_by_degree(vars):
    f = Polynomial(vars, q1 * p1)
    by_deg = f.build_by_degree()
    expected = defaultdict(list)
    _update_by_deg(expected, f)
    assert by_deg == expected

def test_monomial_from_key():
    kq = (2, 1, 0)
    kp = (0, 0, 3)
    monomial = _monomial_from_key(kq, kp, [q1, q2, q3], [p1, p2, p3])
    assert monomial == q1**2*q2*p3**3
    
    # Test with zero exponents
    kq = (0, 0, 0)
    kp = (0, 0, 0)
    monomial = _monomial_from_key(kq, kp, [q1, q2, q3], [p1, p2, p3])
    assert monomial == 1
    
    # Test with mixed exponents
    kq = (1, 0, 2)
    kp = (3, 1, 0)
    monomial = _monomial_from_key(kq, kp, [q1, q2, q3], [p1, p2, p3])
    assert monomial == q1*q3**2*p1**3*p2

def test_monomial_methods(vars):
    # Test get_monomials
    f = Polynomial(vars, 3 * q1**2 * p1 + 2 * q2 * p2)
    monomials = f.get_monomials()
    
    assert len(monomials) == 2
    
    # Find specific monomials
    m1 = None
    m2 = None
    for m in monomials:
        if m.coeff == 3:
            m1 = m
        elif m.coeff == 2:
            m2 = m
    
    assert m1 is not None
    assert m2 is not None
    
    assert m1.coeff == 3
    assert m1.kq == (2,0,0)
    assert m1.kp == (1,0,0)
    assert m1.sym == 3*q1**2*p1
    
    assert m2.coeff == 2
    assert m2.kq == (0,1,0)
    assert m2.kp == (0,1,0)
    assert m2.sym == 2*q2*p2
    
    # Test from_monomials
    f2 = Polynomial.from_monomials(vars, [m1, m2])
    assert f.expansion.expression == f2.expansion.expression
    
    # Test with empty monomial list
    f_empty = Polynomial.from_monomials(vars, [])
    assert f_empty.expression == 0
    
    # Test with a single monomial
    f_single = Polynomial.from_monomials(vars, [m1])
    assert f_single.expression == 3*q1**2*p1
    
    # Test monomial filtering 
    # Create a polynomial with terms of different degrees
    f = Polynomial(vars, q1**3 + q1**2*p1 + q2*p2)
    monomials = f.get_monomials()
    
    # Filter monomials by criteria (e.g., degree 3)
    deg3_monomials = [m for m in monomials if sum(m.kq) + sum(m.kp) == 3]
    f_deg3 = Polynomial.from_monomials(vars, deg3_monomials)
    
    assert len(deg3_monomials) == 2
    assert f_deg3.expression == q1**3 + q1**2*p1

def test_dot_product():
    v1 = (1, 2, 3)
    v2 = [q1, q2, q3]
    result = _dot_product(v1, v2)
    assert result == 1*q1 + 2*q2 + 3*q3


