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

def test_optimized_poisson_standard(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test that optimized_poisson with 'standard' method gives the same results as the original implementation."""
    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    
    # Compare results with standard method
    pb_standard = _poisson_bracket(f, g)
    pb_optimized = f.optimized_poisson(g, method='standard', use_cache=False)
    
    assert pb_standard.expansion.expression == pb_optimized.expansion.expression
    
    # Test with simpler polynomials
    assert P_q1.optimized_poisson(P_q1, method='standard', use_cache=False) == P_zero
    
    # Test with constants
    assert P_one.optimized_poisson(P_q1, method='standard', use_cache=False) == P_zero
    
    # Test all Poisson bracket identities
    
    # 1. Antisymmetry: {f, g} = -{g, f}
    pb1 = f.optimized_poisson(g, method='standard', use_cache=False)
    pb2 = g.optimized_poisson(f, method='standard', use_cache=False)
    assert pb1.expansion.expression == (-pb2).expansion.expression
    
    # 2. Linearity: {f, g + h} = {f, g} + {f, h}
    pb3 = f.optimized_poisson(g + h, method='standard', use_cache=False)
    assert pb3.expansion.expression == (
        f.optimized_poisson(g, method='standard', use_cache=False) +
        f.optimized_poisson(h, method='standard', use_cache=False)
    ).expansion.expression
    
    # 3. Leibniz Rule: {f, g*h} = {f, g}*h + g*{f, h}
    pb4 = f.optimized_poisson(g*h, method='standard', use_cache=False)
    assert pb4.expansion.expression == (
        f.optimized_poisson(g, method='standard', use_cache=False) * h +
        g * f.optimized_poisson(h, method='standard', use_cache=False)
    ).expansion.expression
    
    # 4. Jacobi Identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
    pb5 = f.optimized_poisson(g.optimized_poisson(h, method='standard', use_cache=False), 
                           method='standard', use_cache=False)
    pb6 = g.optimized_poisson(h.optimized_poisson(f, method='standard', use_cache=False), 
                           method='standard', use_cache=False)
    pb7 = h.optimized_poisson(f.optimized_poisson(g, method='standard', use_cache=False), 
                           method='standard', use_cache=False)
    
    assert (pb5 + pb6 + pb7).expansion.expression == 0

def test_optimized_poisson_term_by_term(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test that optimized_poisson with 'term_by_term' method gives correct results."""
    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    
    # Compare results with standard method
    pb_standard = _poisson_bracket(f, g)
    pb_term_by_term = f.optimized_poisson(g, method='term_by_term', use_cache=False)
    
    assert pb_standard.expansion.expression == pb_term_by_term.expansion.expression
    
    # Test with simpler polynomials
    assert P_q1.optimized_poisson(P_q1, method='term_by_term', use_cache=False) == P_zero
    
    # Test with constants
    assert P_one.optimized_poisson(P_q1, method='term_by_term', use_cache=False) == P_zero
    
    # Test all Poisson bracket identities
    
    # 1. Antisymmetry: {f, g} = -{g, f}
    pb1 = f.optimized_poisson(g, method='term_by_term', use_cache=False)
    pb2 = g.optimized_poisson(f, method='term_by_term', use_cache=False)
    assert pb1.expansion.expression == (-pb2).expansion.expression
    
    # 2. Linearity: {f, g + h} = {f, g} + {f, h}
    pb3 = f.optimized_poisson(g + h, method='term_by_term', use_cache=False)
    assert pb3.expansion.expression == (
        f.optimized_poisson(g, method='term_by_term', use_cache=False) +
        f.optimized_poisson(h, method='term_by_term', use_cache=False)
    ).expansion.expression
    
    # 3. Leibniz Rule: {f, g*h} = {f, g}*h + g*{f, h}
    pb4 = f.optimized_poisson(g*h, method='term_by_term', use_cache=False)
    assert pb4.expansion.expression == (
        f.optimized_poisson(g, method='term_by_term', use_cache=False) * h +
        g * f.optimized_poisson(h, method='term_by_term', use_cache=False)
    ).expansion.expression
    
    # 4. Jacobi Identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
    pb5 = f.optimized_poisson(g.optimized_poisson(h, method='term_by_term', use_cache=False), 
                           method='term_by_term', use_cache=False)
    pb6 = g.optimized_poisson(h.optimized_poisson(f, method='term_by_term', use_cache=False), 
                           method='term_by_term', use_cache=False)
    pb7 = h.optimized_poisson(f.optimized_poisson(g, method='term_by_term', use_cache=False), 
                           method='term_by_term', use_cache=False)
    
    assert (pb5 + pb6 + pb7).expansion.expression == 0

def test_optimized_poisson_auto(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test that optimized_poisson with 'auto' method gives correct results."""
    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    
    # Compare results with standard method
    pb_standard = _poisson_bracket(f, g)
    pb_auto = f.optimized_poisson(g, method='auto', use_cache=False)
    
    assert pb_standard.expansion.expression == pb_auto.expansion.expression
    
    # Test with high-degree polynomial which should trigger different algorithm selections
    f_high = P_q1**5 * P_q2**3 + P_p1**4 * P_p2
    g_sparse = P_q1 * P_p1
    
    # These should still give consistent results regardless of algorithm chosen
    pb1 = f_high.optimized_poisson(g_sparse, method='standard', use_cache=False)
    pb2 = f_high.optimized_poisson(g_sparse, method='term_by_term', use_cache=False)
    pb3 = f_high.optimized_poisson(g_sparse, method='auto', use_cache=False)
    
    assert pb1.expansion.expression == pb2.expansion.expression
    assert pb1.expansion.expression == pb3.expansion.expression
    
    # Test all Poisson bracket identities
    
    # 1. Antisymmetry: {f, g} = -{g, f}
    pb1 = f.optimized_poisson(g, method='auto', use_cache=False)
    pb2 = g.optimized_poisson(f, method='auto', use_cache=False)
    assert pb1.expansion.expression == (-pb2).expansion.expression
    
    # 2. Linearity: {f, g + h} = {f, g} + {f, h}
    pb3 = f.optimized_poisson(g + h, method='auto', use_cache=False)
    assert pb3.expansion.expression == (
        f.optimized_poisson(g, method='auto', use_cache=False) +
        f.optimized_poisson(h, method='auto', use_cache=False)
    ).expansion.expression
    
    # 3. Leibniz Rule: {f, g*h} = {f, g}*h + g*{f, h}
    pb4 = f.optimized_poisson(g*h, method='auto', use_cache=False)
    assert pb4.expansion.expression == (
        f.optimized_poisson(g, method='auto', use_cache=False) * h +
        g * f.optimized_poisson(h, method='auto', use_cache=False)
    ).expansion.expression
    
    # 4. Jacobi Identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
    pb5 = f.optimized_poisson(g.optimized_poisson(h, method='auto', use_cache=False), 
                           method='auto', use_cache=False)
    pb6 = g.optimized_poisson(h.optimized_poisson(f, method='auto', use_cache=False), 
                           method='auto', use_cache=False)
    pb7 = h.optimized_poisson(f.optimized_poisson(g, method='auto', use_cache=False), 
                           method='auto', use_cache=False)
    
    assert (pb5 + pb6 + pb7).expansion.expression == 0

def test_memoized_poisson(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test that memoized_poisson gives correct results and caches calculations."""
    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    
    # Clear the LRU cache before running this test to ensure accurate results
    Polynomial.memoized_poisson.cache_clear()
    
    # First call should compute and cache the result
    pb1 = Polynomial.memoized_poisson(f, g)
    
    # Second call should use the cached result
    pb2 = Polynomial.memoized_poisson(f, g)
    
    # Both should match the standard implementation
    pb_standard = _poisson_bracket(f, g)
    
    assert pb1.expansion.expression == pb_standard.expansion.expression
    assert pb2.expansion.expression == pb_standard.expansion.expression
    
    # Check that the cache has at least one entry
    assert Polynomial.memoized_poisson.cache_info().currsize > 0
    
    # Test all Poisson bracket identities with memoized_poisson
    
    # 1. Antisymmetry: {f, g} = -{g, f}
    pb1 = Polynomial.memoized_poisson(f, g)
    pb2 = Polynomial.memoized_poisson(g, f)
    assert pb1.expansion.expression == (-pb2).expansion.expression
    
    # 2. Linearity: {f, g + h} = {f, g} + {f, h}
    pb3 = Polynomial.memoized_poisson(f, g + h)
    assert pb3.expansion.expression == (
        Polynomial.memoized_poisson(f, g) +
        Polynomial.memoized_poisson(f, h)
    ).expansion.expression
    
    # 3. Leibniz Rule: {f, g*h} = {f, g}*h + g*{f, h}
    pb4 = Polynomial.memoized_poisson(f, g*h)
    assert pb4.expansion.expression == (
        Polynomial.memoized_poisson(f, g) * h +
        g * Polynomial.memoized_poisson(f, h)
    ).expansion.expression
    
    # 4. Jacobi Identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
    pb5 = Polynomial.memoized_poisson(f, Polynomial.memoized_poisson(g, h))
    pb6 = Polynomial.memoized_poisson(g, Polynomial.memoized_poisson(h, f))
    pb7 = Polynomial.memoized_poisson(h, Polynomial.memoized_poisson(f, g))
    
    assert (pb5 + pb6 + pb7).expansion.expression == 0
    
    # Check cache hits during identity testing
    cache_info = Polynomial.memoized_poisson.cache_info()
    assert cache_info.hits > 0, "Cache should have been used during identity testing"

def test_poisson_methods_mathematical_properties(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test that all Poisson bracket methods satisfy the mathematical properties."""
    f = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    g = P_q1 * P_q2 ** 2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    h = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3
    
    # Test different methods for computing {f, g}
    methods = ['standard', 'term_by_term', 'auto']
    
    for method in methods:
        # 1. Antisymmetry: {f, g} = -{g, f}
        pb_fg = f.optimized_poisson(g, method=method, use_cache=False)
        pb_gf = g.optimized_poisson(f, method=method, use_cache=False)
        assert pb_fg.expansion.expression == (-pb_gf).expansion.expression
        
        # 2. Linearity: {f, g + h} = {f, g} + {f, h}
        pb_fgh = f.optimized_poisson(g + h, method=method, use_cache=False)
        pb_fg_plus_pb_fh = f.optimized_poisson(g, method=method, use_cache=False) + f.optimized_poisson(h, method=method, use_cache=False)
        assert pb_fgh.expansion.expression == pb_fg_plus_pb_fh.expansion.expression
        
        # 3. Leibniz Rule: {f, g*h} = {f, g}*h + g*{f, h}
        pb_fgh_product = f.optimized_poisson(g * h, method=method, use_cache=False)
        product_rule = f.optimized_poisson(g, method=method, use_cache=False) * h + g * f.optimized_poisson(h, method=method, use_cache=False)
        assert pb_fgh_product.expansion.expression == product_rule.expansion.expression
        
        # 4. Jacobi Identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0
        pb_gh = g.optimized_poisson(h, method=method, use_cache=False)
        pb_hf = h.optimized_poisson(f, method=method, use_cache=False)
        pb_fg = f.optimized_poisson(g, method=method, use_cache=False)
        
        pb_f_gh = f.optimized_poisson(pb_gh, method=method, use_cache=False)
        pb_g_hf = g.optimized_poisson(pb_hf, method=method, use_cache=False)
        pb_h_fg = h.optimized_poisson(pb_fg, method=method, use_cache=False)
        
        jacobi_sum = (pb_f_gh + pb_g_hf + pb_h_fg).expansion.expression
        assert jacobi_sum == 0

def test_optimized_poisson_performance(P_complex, P_high_order):
    """
    Test performance characteristics of the different Poisson bracket implementations.
    This is not a strict test but provides information about the relative performance.
    """
    import time
    
    # Complex polynomials for testing
    f = P_complex
    g = P_high_order
    
    # Measure time for each method
    methods = {
        'standard': lambda: f.optimized_poisson(g, method='standard', use_cache=False),
        'term_by_term': lambda: f.optimized_poisson(g, method='term_by_term', use_cache=False),
        'auto': lambda: f.optimized_poisson(g, method='auto', use_cache=False),
        'memoized': lambda: f.optimized_poisson(g, use_cache=True)
    }
    
    results = {}
    iterations = 5  # Number of iterations for each method
    
    for name, method_func in methods.items():
        # Clear cache for fair comparison
        if name == 'memoized':
            Polynomial.memoized_poisson.cache_clear()
        
        # Warm-up call
        result = method_func()
        
        # Measure time
        start_time = time.time()
        for _ in range(iterations):
            method_func()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        results[name] = avg_time
    
    # No assertions here, just print performance info
    # Results will vary by machine, so this is mainly for development reference
    print(f"\nPoisson bracket performance (avg time over {iterations} iterations):")
    for name, avg_time in results.items():
        print(f"  {name}: {avg_time:.6f} seconds")
    
    # Verify all methods give the same result
    pb_standard = f.optimized_poisson(g, method='standard', use_cache=False)
    pb_term_by_term = f.optimized_poisson(g, method='term_by_term', use_cache=False)
    pb_auto = f.optimized_poisson(g, method='auto', use_cache=False)
    pb_memoized = f.optimized_poisson(g, use_cache=True)
    
    assert pb_standard.expansion.expression == pb_term_by_term.expansion.expression
    assert pb_standard.expansion.expression == pb_auto.expansion.expression
    assert pb_standard.expansion.expression == pb_memoized.expansion.expression

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

def test_sparse_polynomial_optimization(vars):
    """
    Test specifically focused on sparse polynomials where term-by-term
    differentiation would have an advantage over standard differentiation.
    """
    import time
    
    # Create very high-dimensional but sparse polynomials
    q1, q2, q3, p1, p2, p3 = vars
    
    # Sparse polynomial with high powers but few terms
    sparse_f = Polynomial(vars, q1**10 + p1**10)
    sparse_g = Polynomial(vars, q2**10 + p2**10)
    
    # Dense polynomial with many terms
    terms = []
    for i in range(5):
        for j in range(5):
            terms.append(q1**i * q2**j * p1**(4-i) * p2**(4-j))
    dense_f = Polynomial(vars, sum(terms))
    dense_g = Polynomial(vars, sum(terms) * 2)
    
    # Define test cases
    test_cases = [
        ("sparse-sparse", sparse_f, sparse_g),
        ("sparse-dense", sparse_f, dense_g),
        ("dense-dense", dense_f, dense_g)
    ]
    
    # Number of iterations for timing
    iterations = 3
    
    for name, f, g in test_cases:
        print(f"\nTest case: {name}")
        
        # Term counts for reference
        f_terms = len(f.expansion.get_monomials())
        g_terms = len(g.expansion.get_monomials())
        print(f"  F has {f_terms} terms, G has {g_terms} terms")
        
        # Time standard method
        start_time = time.time()
        for _ in range(iterations):
            result_standard = f.optimized_poisson(g, method='standard', use_cache=False)
        standard_time = (time.time() - start_time) / iterations
        
        # Time term-by-term method
        start_time = time.time()
        for _ in range(iterations):
            result_term = f.optimized_poisson(g, method='term_by_term', use_cache=False)
        term_time = (time.time() - start_time) / iterations
        
        # Time auto method (should select the faster one)
        start_time = time.time()
        for _ in range(iterations):
            result_auto = f.optimized_poisson(g, method='auto', use_cache=False)
        auto_time = (time.time() - start_time) / iterations
        
        print(f"  Standard method: {standard_time:.6f} seconds")
        print(f"  Term-by-term method: {term_time:.6f} seconds")
        print(f"  Auto method: {auto_time:.6f} seconds")
        
        # Verify results match
        assert result_standard.expansion.expression == result_term.expansion.expression
        assert result_standard.expansion.expression == result_auto.expansion.expression
        
        # For sparse polynomials, term-by-term should be faster
        # (we don't assert this because it may not always be true depending on implementations)
        if name == "sparse-sparse":
            print(f"  Relative speed (term-by-term vs standard): {standard_time/term_time:.2f}x")
            
        # Check which method was chosen by 'auto'
        method_chosen = "term-by-term" if abs(auto_time - term_time) < abs(auto_time - standard_time) else "standard"
        print(f"  Auto method likely chose: {method_chosen}")

def test_memoization_efficiency(P_complex):
    """
    Test the efficiency gains of memoization when doing repeated calculations.
    This simulates a real-world scenario where the same Poisson brackets
    are computed multiple times during normal form calculations.
    """
    import time
    
    # Create several polynomials for testing
    f = P_complex
    g = f * 2
    h = f * 3
    j = f * 4
    k = f * 5
    
    # List of polynomial pairs to compute brackets for
    bracket_pairs = [
        (f, g), (f, h), (f, j), (f, k),
        (g, h), (g, j), (g, k),
        (h, j), (h, k),
        (j, k)
    ]
    
    # Function to run calculation with or without cache
    def run_calculation(use_cache):
        # First calculate all brackets
        results = []
        for p1, p2 in bracket_pairs:
            results.append(p1.optimized_poisson(p2, use_cache=use_cache))
        
        # Then calculate all brackets again (with some repetition)
        second_results = []
        for p1, p2 in bracket_pairs:
            second_results.append(p1.optimized_poisson(p2, use_cache=use_cache))
            
        # Calculate some brackets between the results (creates new combinations)
        for i in range(5):  # Only use a subset to keep test runtime reasonable
            for j in range(i+1, 5):
                results.append(results[i].optimized_poisson(results[j], use_cache=use_cache))
        
        return results, second_results
    
    # Clear the cache before starting
    Polynomial.memoized_poisson.cache_clear()
    
    # Time without memoization
    print("\nMemoization efficiency test:")
    start_time = time.time()
    results_no_cache, second_results_no_cache = run_calculation(use_cache=False)
    no_cache_time = time.time() - start_time
    print(f"  Without memoization: {no_cache_time:.6f} seconds")
    
    # Clear the cache
    Polynomial.memoized_poisson.cache_clear()
    
    # Time with memoization
    start_time = time.time()
    results_with_cache, second_results_with_cache = run_calculation(use_cache=True)
    with_cache_time = time.time() - start_time
    print(f"  With memoization: {with_cache_time:.6f} seconds")
    
    # Verify results match
    for i in range(len(results_no_cache)):
        assert results_no_cache[i].expansion.expression == results_with_cache[i].expansion.expression
    
    # Report cache stats
    cache_info = Polynomial.memoized_poisson.cache_info()
    print(f"  Cache hits: {cache_info.hits}")
    print(f"  Cache misses: {cache_info.misses}")
    print(f"  Cache current size: {cache_info.currsize}")
    
    # Performance improvement
    speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
    print(f"  Speedup with memoization: {speedup:.2f}x")
    
    # With sufficient repetition, we should see a significant speedup
    # This assertion may occasionally fail depending on test environment
    # so we make it very conservative
    assert speedup > 1.1, "Memoization should provide at least a 10% speedup with repeated calculations"


