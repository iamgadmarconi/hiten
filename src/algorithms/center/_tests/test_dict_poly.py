import pytest
import numpy as np
import symengine as se
import sympy as sp
from collections import defaultdict

from algorithms.center.dict_core import DictPolynomial
from algorithms.variables import get_vars, canonical_normal_vars

q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)

@pytest.fixture
def vars():
    return [q1, q2, q3, p1, p2, p3]

@pytest.fixture
def P_zero(vars):
    return DictPolynomial(vars, {})

@pytest.fixture
def P_one(vars):
    return DictPolynomial(vars, {(0, 0, 0, 0, 0, 0): 1.0})

@pytest.fixture
def P_q1(vars):
    return DictPolynomial(vars, {(1, 0, 0, 0, 0, 0): 1.0})

@pytest.fixture
def P_p1(vars):
    return DictPolynomial(vars, {(0, 0, 0, 1, 0, 0): 1.0})

@pytest.fixture
def P_q2(vars):
    return DictPolynomial(vars, {(0, 1, 0, 0, 0, 0): 1.0})

@pytest.fixture
def P_p2(vars):
    return DictPolynomial(vars, {(0, 0, 0, 0, 1, 0): 1.0})

@pytest.fixture
def P_q3(vars):
    return DictPolynomial(vars, {(0, 0, 1, 0, 0, 0): 1.0})

@pytest.fixture
def P_p3(vars):
    return DictPolynomial(vars, {(0, 0, 0, 0, 0, 1): 1.0})

@pytest.fixture
def P_qp_sum(P_q1, P_p1):
    return P_q1 + P_p1

@pytest.fixture
def P_complex(P_q1, P_p1, P_q2, P_p2, P_q3, P_p3):
    return 3*P_q1**2*P_p1 + 2*P_q2*P_p2 - 5*P_q3*P_p3**3

@pytest.fixture
def P_high_order(P_q1, P_p1, P_q2, P_p2, P_q3, P_p3):
    return P_q1**7*P_p1 + P_q2**6*P_p2 + P_q3**5*P_p3 + P_q3**2*7

def test_initialization(vars, P_zero, P_one, P_q1):
    """Test initialization of DictPolynomial objects."""
    assert len(P_zero.variables) == len(vars)
    assert len(P_zero.coeffs) == 0
    
    assert len(P_one.variables) == len(vars)
    assert P_one.coeffs.get((0, 0, 0, 0, 0, 0), 0) == 1.0
    
    assert len(P_q1.variables) == len(vars)
    assert P_q1.coeffs.get((1, 0, 0, 0, 0, 0), 0) == 1.0
    
    # Test initialization with symengine expressions
    expr = q1 + 2
    p_with_expr = DictPolynomial.from_symengine(vars, expr)
    
    # Convert both to SymPy expressions for proper comparison
    # This handles the float vs int coefficient issue
    expr_sp = sp.sympify(expr)
    p_expr_sp = sp.sympify(p_with_expr.to_symengine())
    assert sp.simplify(p_expr_sp - expr_sp) == 0
    
    # Alternatively, check the structure of the polynomial directly
    assert len(p_with_expr.coeffs) == 2  # Should have two terms
    assert abs(p_with_expr.coeffs.get((1, 0, 0, 0, 0, 0), 0) - 1.0) < 1e-10  # q1 term
    assert abs(p_with_expr.coeffs.get((0, 0, 0, 0, 0, 0), 0) - 2.0) < 1e-10  # constant term
    
    # Test creation with numerical value
    p_with_num = DictPolynomial.from_symengine(vars, 42)
    assert len(p_with_num.coeffs) == 1
    assert p_with_num.coeffs.get((0, 0, 0, 0, 0, 0), 0) == 42.0
    
    # Test copy method
    p_copy = P_q1.copy()
    assert p_copy == P_q1
    assert p_copy is not P_q1

def test_string_representation(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test string representation of polynomials"""
    assert str(P_zero) == "0"
    assert str(P_one) == "1"
    assert str(P_q1) == "q1"
    
    # Test more complex polynomial
    complex_poly = 3*P_q1**2*P_p1 + 2*P_q2*P_p2
    assert "3*q1**2*p1" in str(complex_poly)
    assert "2*q2*p2" in str(complex_poly)

def test_equality(P_zero, P_one, P_q1, P_p1):
    """Test equality comparison of DictPolynomial objects."""
    assert P_q1 == P_q1
    assert P_q1 != P_p1
    assert P_q1 != P_zero
    assert P_one != P_zero
    
    # Create a copy with the same coefficients
    P_q1_copy = DictPolynomial(P_q1.variables, dict(P_q1.coeffs))
    assert P_q1 == P_q1_copy
    
    # Test with different coefficient values
    P_q1_2 = DictPolynomial(P_q1.variables, {(1, 0, 0, 0, 0, 0): 2.0})
    assert P_q1 != P_q1_2

def test_multiplication(P_zero, P_one, P_q1, P_p1):
    """Test multiplication of DictPolynomial objects."""
    assert P_q1 * P_one == P_q1
    assert P_q1 * P_zero == P_zero
    
    # q1 * p1 = q1p1
    q1p1 = P_q1 * P_p1
    expected = DictPolynomial(P_q1.variables, {(1, 0, 0, 1, 0, 0): 1.0})
    assert q1p1 == expected
    
    # Test scalar multiplication
    assert P_q1 * 2 == 2 * P_q1
    assert 2 * P_q1 == DictPolynomial(P_q1.variables, {(1, 0, 0, 0, 0, 0): 2.0})
    
    # Test more complex multiplication
    q1_squared = P_q1 * P_q1
    assert q1_squared == DictPolynomial(P_q1.variables, {(2, 0, 0, 0, 0, 0): 1.0})
    
    # Test power operation
    q1_cubed = P_q1**3
    assert q1_cubed == P_q1 * P_q1 * P_q1
    assert q1_cubed == DictPolynomial(P_q1.variables, {(3, 0, 0, 0, 0, 0): 1.0})

def test_addition_subtraction(P_zero, P_one, P_q1, P_p1):
    """Test addition and subtraction of DictPolynomial objects."""
    # Identities
    assert P_q1 + P_zero == P_q1
    assert P_q1 - P_q1 == P_zero
    
    # q1 + p1
    q1_plus_p1 = P_q1 + P_p1
    expected = DictPolynomial(P_q1.variables, {
        (1, 0, 0, 0, 0, 0): 1.0,
        (0, 0, 0, 1, 0, 0): 1.0
    })
    assert q1_plus_p1 == expected
    
    # Commutativity
    assert P_q1 + P_p1 == P_p1 + P_q1
    
    # Scalar addition/subtraction
    q1_plus_5 = P_q1 + 5
    assert q1_plus_5 == DictPolynomial(P_q1.variables, {
        (1, 0, 0, 0, 0, 0): 1.0,
        (0, 0, 0, 0, 0, 0): 5.0
    })
    
    # Right subtraction
    five_minus_q1 = 5 - P_q1
    assert five_minus_q1 == DictPolynomial(P_q1.variables, {
        (1, 0, 0, 0, 0, 0): -1.0,
        (0, 0, 0, 0, 0, 0): 5.0
    })
    
    # Associativity
    triple_q1 = (P_q1 + P_q1) + P_q1
    assert triple_q1 == 3 * P_q1
    
    # Negation
    neg_q1 = -P_q1
    assert neg_q1 == DictPolynomial(P_q1.variables, {(1, 0, 0, 0, 0, 0): -1.0})
    assert P_q1 + neg_q1 == P_zero

def test_differentiation(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test differentiation of DictPolynomial objects."""
    # Test derivatives of constants
    assert P_zero.derivative(0) == P_zero
    assert P_one.derivative(0) == P_zero
    
    # Test derivative with respect to variable
    assert P_q1.derivative(0) == P_one
    assert P_q1.derivative(3) == P_zero  # derivative wrt p1
    assert P_p1.derivative(0) == P_zero  # derivative wrt q1
    assert P_p1.derivative(3) == P_one   # derivative wrt p1
    
    # Test squared terms
    P_q1_squared = P_q1 * P_q1
    assert P_q1_squared.derivative(0) == 2 * P_q1
    
    # Test product of variables
    P_qp = P_q1 * P_p1
    assert P_qp.derivative(0) == P_p1
    assert P_qp.derivative(3) == P_q1
    
    # Test second derivative
    assert P_q1.derivative(0).derivative(0) == P_zero
    
    # Test derivatives of cubic terms
    P_q1_cubed = P_q1 * P_q1 * P_q1
    assert P_q1_cubed.derivative(0) == 3 * P_q1 * P_q1
    assert P_q1_cubed.derivative(0).derivative(0) == 6 * P_q1
    
    # Test mixed terms
    P_mixed = P_q1 * P_q2 + P_p1 * P_p3
    assert P_mixed.derivative(0) == P_q2
    assert P_mixed.derivative(1) == P_q1
    assert P_mixed.derivative(3) == P_p3
    assert P_mixed.derivative(5) == P_p1
    
    # Test polynomial with multiple terms
    P_complex = 2 * P_q1 * P_q1 + 3 * P_q1 * P_p1 - 5 * P_p2 * P_p2
    assert P_complex.derivative(0) == 4 * P_q1 + 3 * P_p1
    assert P_complex.derivative(3) == 3 * P_q1
    assert P_complex.derivative(4) == -10 * P_p2

def test_gradient(vars, P_zero, P_one, P_q1, P_p1, P_q2, P_p2, P_p3):
    """Test gradient calculation using derivatives."""
    # Test gradient of q1
    for i in range(len(vars)):
        if i == 0:  # q1 index
            assert P_q1.derivative(i) == P_one
        else:
            assert P_q1.derivative(i) == P_zero
    
    # Test with a different polynomial directly constructed with the fixture objects
    q1_squared = P_q1 * P_q1
    q1_p1 = P_q1 * P_p1
    p2_squared = P_p2 * P_p2
    
    test_poly = q1_squared * 2 + q1_p1 * 3 - p2_squared * 5
    
    # Expected gradients
    expected_dF_dq1 = P_q1 * 4 + P_p1 * 3
    expected_dF_dp1 = P_q1 * 3
    expected_dF_dp2 = P_p2 * (-10)
    
    assert test_poly.derivative(0) == expected_dF_dq1
    assert test_poly.derivative(3) == expected_dF_dp1
    assert test_poly.derivative(4) == expected_dF_dp2
    
    # Check that other derivatives are zero
    assert test_poly.derivative(1) == P_zero  # dF/dq2
    assert test_poly.derivative(2) == P_zero  # dF/dq3
    assert test_poly.derivative(5) == P_zero  # dF/dp3

def test_poisson_bracket(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test Poisson bracket computation."""
    # Basic Poisson bracket tests
    assert P_q1.poisson(P_q1) == P_zero
    assert P_q1.poisson(P_p1) == P_one
    assert P_p1.poisson(P_q1) == -P_one
    assert P_q1.poisson(P_q2) == P_zero
    assert P_p1.poisson(P_p2) == P_zero
    
    # Define polynomials for testing
    # Build the polynomials step by step to avoid any potential issues
    q1_sq = P_q1 * P_q1
    q1_p1 = P_q1 * P_p1
    p2_sq = P_p2 * P_p2
    q2_sq = P_q2 * P_q2
    q1_q2_sq = P_q1 * q2_sq
    q1_q2_sq_p1 = q1_q2_sq * P_p1
    p1_p2 = P_p1 * P_p2
    q3_p3 = P_q3 * P_p3
    
    f = q1_sq * 2 + q1_p1 * 3 - p2_sq * 5
    g = q1_q2_sq_p1 + p1_p2 - q3_p3
    h = P_q1 * P_q2 * P_p1 + p1_p2 - q3_p3
    
    # Calculate Poisson brackets
    pb1 = f.poisson(g)
    pb2 = g.poisson(f)
    pb3 = f.poisson(g + h)
    pb4 = f.poisson(g*h)
    pb5 = g.poisson(h)
    pb6 = h.poisson(f)
    pb7 = f.poisson(pb5)
    pb8 = g.poisson(pb6)
    pb9 = h.poisson(pb1)
    
    # Test antisymmetry property
    assert pb1 == -pb2
    
    # Test linearity property
    assert pb3 == f.poisson(g) + f.poisson(h)
    
    # Test Leibniz rule
    assert pb4 == f.poisson(g) * h + g * f.poisson(h)
    
    # Test Jacobi identity (may need smaller polynomials for this)
    sum_jacobi = pb7 + pb8 + pb9
    assert sum_jacobi == P_zero

def test_get_homogeneous_part(P_complex, P_high_order):
    """Test extracting homogeneous parts of a polynomial."""
    # Get homogeneous part of degree 3 from complex polynomial
    # P_complex = 3*P_q1**2*P_p1 + 2*P_q2*P_p2 - 5*P_q3*P_p3**3
    degree_3_part = P_complex.get_homogeneous_part(3)
    expected = 3 * P_q1**2 * P_p1
    assert degree_3_part == expected
    
    # Get homogeneous part of degree 2
    degree_2_part = P_complex.get_homogeneous_part(2)
    expected = 2 * P_q2 * P_p2
    assert degree_2_part == expected
    
    # Get homogeneous part of degree 4
    degree_4_part = P_complex.get_homogeneous_part(4)
    expected = -5 * P_q3 * P_p3**3
    assert degree_4_part == expected
    
    # Test with high order polynomial
    # P_high_order = P_q1**7*P_p1 + P_q2**6*P_p2 + P_q3**5*P_p3 + P_q3**2*7
    degree_8_part = P_high_order.get_homogeneous_part(8)
    expected = P_q1**7 * P_p1
    assert degree_8_part == expected
    
    # Get homogeneous part of degree that doesn't exist
    degree_5_part = P_complex.get_homogeneous_part(5)
    assert degree_5_part == P_zero

def test_truncate(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test truncation of polynomials by degree."""
    # Create a polynomial with terms of different degrees
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3 ** 3
    
    # Truncate to degree 3
    f3 = f.truncate(3)
    expected3 = P_q1 * P_q2 * P_p1 + P_p1 * P_p2
    assert f3 == expected3
    
    # Truncate to degree 2
    f2 = f.truncate(2)
    expected2 = P_p1 * P_p2
    assert f2 == expected2
    
    # Truncate to degree 1
    f1 = f.truncate(1)
    assert f1 == P_zero
    
    # Truncate to degree 0
    f0 = (f + 3).truncate(0)
    assert f0 == 3 * P_one
    
    # Truncate high order polynomial
    high_order = P_q1**7*P_p1 + P_q2**6*P_p2 + P_q3**5*P_p3 + P_q3**2*7
    truncated = high_order.truncate(7)
    expected = P_q3**5*P_p3 + P_q3**2*7
    assert truncated == expected

def test_total_degree(P_zero, P_one, P_q1, P_p1, P_q2, P_q3, P_p2, P_p3):
    """Test total degree calculation."""
    f = P_q1 * P_q2 * P_p1 + P_p1 * P_p2 - P_q3 * P_p3 ** 3
    assert f.total_degree() == 4
    
    assert P_q1.total_degree() == 1
    assert P_zero.total_degree() == -1  # DictPolynomial returns -1 for zero polynomial
    assert P_one.total_degree() == 0
    
    # Test with higher degree polynomial
    high_order = P_q1**7*P_p1 + P_q2**6*P_p2 + P_q3**5*P_p3 + P_q3**2*7
    assert high_order.total_degree() == 8  # q1^7 * p1 has degree 8

def test_iteration(P_complex):
    """Test iteration over terms in a polynomial."""
    # P_complex = 3*P_q1**2*P_p1 + 2*P_q2*P_p2 - 5*P_q3*P_p3**3
    terms = list(P_complex.iter_terms())
    
    # Check that we have 3 terms
    assert len(terms) == 3
    
    # Check the coefficients and exponents
    exps_found = set()
    for exps, coeff in terms:
        if exps == (2, 0, 0, 1, 0, 0):  # q1^2*p1
            assert coeff == 3.0
        elif exps == (0, 1, 0, 0, 1, 0):  # q2*p2
            assert coeff == 2.0
        elif exps == (0, 0, 1, 0, 0, 3):  # q3*p3^3
            assert coeff == -5.0
        exps_found.add(exps)
    
    # Ensure all expected terms were found
    assert (2, 0, 0, 1, 0, 0) in exps_found
    assert (0, 1, 0, 0, 1, 0) in exps_found
    assert (0, 0, 1, 0, 0, 3) in exps_found

def test_conversion():
    """Test conversion between SymEngine and DictPolynomial."""
    x, y = se.symbols("x y")
    vars = [x, y]
    
    # Test simple expression
    expr = 3*x**2 + 2*y + 5*x*y
    poly = DictPolynomial.from_symengine(vars, expr)
    
    # Verify the coefficients
    assert poly.coeffs.get((2, 0), 0) == 3.0
    assert poly.coeffs.get((0, 1), 0) == 2.0
    assert poly.coeffs.get((1, 1), 0) == 5.0
    
    # Convert back to SymEngine and verify
    expr2 = poly.to_symengine()
    assert se.expand(expr) == se.expand(expr2)
    
    # Test complex expression
    expr = (x**2 + y**2) * (x + y) - x*y
    poly = DictPolynomial.from_symengine(vars, expr)
    expr2 = poly.to_symengine()
    assert se.expand(expr) == se.expand(expr2)
    
    # Test with complex coefficients
    expr = 3j*x**2 + (2+1j)*y
    poly = DictPolynomial.from_symengine(vars, expr)
    assert abs(poly.coeffs.get((2, 0), 0) - 3j) < 1e-10
    assert abs(poly.coeffs.get((0, 1), 0) - (2+1j)) < 1e-10
    
    # Convert back and verify
    expr2 = poly.to_symengine()
    assert se.expand(expr) == se.expand(expr2)

def test_performance():
    """Test performance of DictPolynomial operations."""
    # Define variables
    x, y, z, px, py, pz = se.symbols("x y z px py pz")
    vars = [x, y, z, px, py, pz]
    
    # Create complex polynomials
    expr1 = (x**2 + y**2 + z**2) * (px**2 + py**2 + pz**2) + x*py - y*px
    expr2 = x*px + y*py + z*pz + 0.5*(x**2 + y**2 + z**2)
    
    # Convert to DictPolynomial
    poly1 = DictPolynomial.from_symengine(vars, expr1)
    poly2 = DictPolynomial.from_symengine(vars, expr2)
    
    # Perform multiplication
    result = poly1 * poly2
    
    # Verify the result by converting back to SymEngine
    se_result = se.expand(expr1 * expr2)
    dict_se_result = result.to_symengine()
    assert se.expand(se_result) == se.expand(dict_se_result)
    
    # Check number of terms in result
    assert len(result.coeffs) > 0
    
    # Test Poisson bracket performance
    poisson_result = poly1.poisson(poly2)
    # Just check that we get a result (don't verify for performance test)
    assert isinstance(poisson_result, DictPolynomial)

if __name__ == "__main__":
    pytest.main() 