import pytest
import numpy as np
import symengine as se

from algorithms.center.polynomials import Polynomial

# --- Helper for comparing polynomials ---
def assert_poly_equal(p1, p2):
    """Asserts that two Polynomial objects are equal."""
    assert isinstance(p1, Polynomial)
    assert isinstance(p2, Polynomial)
    assert p1.n_vars == p2.n_vars
    # Use the equals method which compares expressions
    assert p1 == p2, f"Expressions differ: {p1.expr} vs {p2.expr}"

# --- Pytest Fixtures ---
@pytest.fixture
def n_vars():
    return 6 # Default for 3 DOF CR3BP

@pytest.fixture
def P_zero(n_vars):
    return Polynomial.zero(n_vars=n_vars)

@pytest.fixture
def P_one(n_vars):
    return Polynomial.one(n_vars=n_vars)

@pytest.fixture
def P_q0(n_vars): # q1 in the text
    return Polynomial('x0', n_vars=n_vars)

@pytest.fixture
def P_p0(n_vars): # p1 in the text
    return Polynomial('x1', n_vars=n_vars)

@pytest.fixture
def P_q1(n_vars): # q2 in the text
    return Polynomial('x2', n_vars=n_vars)

@pytest.fixture
def P_p1(n_vars): # p2 in the text
    return Polynomial('x3', n_vars=n_vars)

@pytest.fixture
def P_qp_sum(P_q0, P_p0): # q0 + p0
    return P_q0 + P_p0

@pytest.fixture
def P_H_simple(n_vars): # Simple Hamiltonian H = 0.5*p0^2 + q0^2
    return Polynomial('0.5*x1**2 + x0**2', n_vars=n_vars)

@pytest.fixture
def P_mixed(n_vars): # A more complex polynomial with mixed terms
    return Polynomial('2*x0**2 + 3*x1 + 4*x0*x2 - 5*x3**3 + 7', n_vars=n_vars)

# --- Test Functions ---

def test_initialization(n_vars, P_zero, P_one, P_q0):
    assert P_zero.n_vars == n_vars
    assert str(P_zero.expr) == '0'

    assert P_one.n_vars == n_vars
    assert str(P_one.expr) == '1'

    assert P_q0.n_vars == n_vars
    assert str(P_q0.expr) == 'x0'

    # Test initialization with symengine expressions
    x0 = se.Symbol('x0')
    p_with_expr = Polynomial(x0 + 2, n_vars=n_vars)
    # The string representation may have different ordering (2 + x0 vs x0 + 2)
    assert p_with_expr.equals(Polynomial('x0 + 2', n_vars=n_vars))
    
    # Test creation with numerical value
    p_with_num = Polynomial(42, n_vars=n_vars)
    assert str(p_with_num.expr) == '42'

    # Test wrong n_vars
    with pytest.raises(ValueError):
        Polynomial(1, n_vars=3) # n_vars must be even

def test_addition_subtraction(P_zero, P_one, P_q0, P_p0, P_qp_sum):
    # Identities
    assert_poly_equal(P_q0 + P_zero, P_q0)
    assert_poly_equal(P_q0 - P_q0, P_zero)
    assert_poly_equal(P_q0 + (-1*P_q0), P_zero)

    # Commutativity
    assert_poly_equal(P_q0 + P_p0, P_p0 + P_q0)

    # Known result
    assert_poly_equal(P_q0 + P_p0, P_qp_sum)
    assert_poly_equal(P_qp_sum - P_p0, P_q0)

    # Scalar add/sub
    P_q0_plus_5 = P_q0 + 5
    expected = Polynomial('x0 + 5', n_vars=P_q0.n_vars)
    assert_poly_equal(P_q0_plus_5, expected)
    assert_poly_equal(P_q0_plus_5 - 5, P_q0)

def test_multiplication(P_zero, P_one, P_q0, P_p0, P_qp_sum):
    # Identities
    assert_poly_equal(P_q0 * P_one, P_q0)
    assert_poly_equal(P_q0 * P_zero, P_zero)
    assert_poly_equal(P_zero * P_q0, P_zero)

    # Commutativity
    assert_poly_equal(P_q0 * P_p0, P_p0 * P_q0)

    # Scalar
    P_3q0 = 3 * P_q0
    # Using equals directly to handle formatting differences (3*x0 vs 3.0*x0)
    P_q0_times_3 = P_q0 * 3.0
    assert P_3q0.equals(P_q0_times_3), f"Expressions differ: {P_3q0.expr} vs {P_q0_times_3.expr}"

    # Known results
    P_q0p0 = P_q0 * P_p0
    expected_q0p0 = Polynomial('x0*x1', n_vars=P_q0.n_vars)
    assert_poly_equal(P_q0p0, expected_q0p0)

    # (q0+p0)^2 = q0^2 + p0^2 + 2*q0*p0
    P_sq = P_qp_sum * P_qp_sum
    expected_sq = Polynomial('x0**2 + x1**2 + 2*x0*x1', n_vars=P_q0.n_vars)
    assert_poly_equal(P_sq, expected_sq)

    # Distributivity P*(Q+R) = P*Q + P*R
    P = P_q0
    Q = P_p0
    R = P_one
    LHS = P * (Q + R)
    RHS = (P * Q) + (P * R)
    assert_poly_equal(LHS, RHS)

def test_differentiation(n_vars, P_zero, P_one, P_q0, P_p0, P_q1, P_H_simple):
    # Constants
    assert_poly_equal(P_one.differentiate(0), P_zero) # d(1)/dq0 = 0
    assert_poly_equal(P_one.differentiate(1), P_zero) # d(1)/dp0 = 0

    # Single variable
    assert_poly_equal(P_q0.differentiate(0), P_one) # dq0/dq0 = 1
    assert_poly_equal(P_q0.differentiate(1), P_zero) # dq0/dp0 = 0
    assert_poly_equal(P_q0.differentiate(2), P_zero) # dq0/dq1 = 0

    # Known derivative H = 0.5*p0^2 + q0^2
    dH_dq0 = P_H_simple.differentiate(0) # Should be 2*q0
    expected_dH_dq0 = Polynomial('2*x0', n_vars=n_vars)
    assert_poly_equal(dH_dq0, expected_dH_dq0)

    dH_dp0 = P_H_simple.differentiate(1) # Should be p0
    expected_dH_dp0 = Polynomial('x1', n_vars=n_vars)
    assert_poly_equal(dH_dp0, expected_dH_dp0)

    # Linearity d(P+Q)/dx = dP/dx + dQ/dx
    P = P_H_simple
    Q = P_q0
    var_idx = 0 # differentiate w.r.t q0
    LHS = (P + Q).differentiate(var_idx)
    RHS = P.differentiate(var_idx) + Q.differentiate(var_idx)
    assert_poly_equal(LHS, RHS)

    # Product Rule d(P*Q)/dx = P*dQ/dx + Q*dP/dx
    P = P_q0
    Q = P_p0
    var_idx = 0 # differentiate w.r.t q0
    LHS = (P * Q).differentiate(var_idx) # d(q0*p0)/dq0 = p0
    RHS = P * Q.differentiate(var_idx) + Q * P.differentiate(var_idx) # q0*0 + p0*1 = p0
    assert_poly_equal(LHS, P_p0)
    assert_poly_equal(RHS, P_p0)
    assert_poly_equal(LHS, RHS)

    # Mixed Partials d2P/dxdy = d2P/dydx
    P = P_q0 * P_p0 * P_q1 # q0 * p0 * q1
    dPdq0 = P.differentiate(0) # p0 * q1
    d2Pdq0dp0 = dPdq0.differentiate(1) # q1
    assert_poly_equal(d2Pdq0dp0, P_q1)

    dPdp0 = P.differentiate(1) # q0 * q1
    d2Pdp0dq0 = dPdp0.differentiate(0) # q1
    assert_poly_equal(d2Pdp0dq0, P_q1)

    assert_poly_equal(d2Pdq0dp0, d2Pdp0dq0)

def test_poisson_bracket(P_zero, P_one, P_q0, P_p0, P_q1, P_p1, P_H_simple):
    # Fundamental Brackets
    assert_poly_equal(P_q0.poisson_bracket(P_q1), P_zero) # {q0, q1} = 0
    assert_poly_equal(P_p0.poisson_bracket(P_p1), P_zero) # {p0, p1} = 0
    assert_poly_equal(P_q0.poisson_bracket(P_p0), P_one)  # {q0, p0} = 1
    assert_poly_equal(P_q0.poisson_bracket(P_p1), P_zero) # {q0, p1} = 0
    assert_poly_equal(P_q1.poisson_bracket(P_p0), P_zero) # {q1, p0} = 0
    assert_poly_equal(P_q1.poisson_bracket(P_p1), P_one)  # {q1, p1} = 1

    # Antisymmetry {P, Q} = -{Q, P}
    P = P_H_simple
    Q = P_q0
    PB1 = P.poisson_bracket(Q)
    PB2 = Q.poisson_bracket(P)
    assert_poly_equal(PB1, -1 * PB2)

    # Linearity {aP+bQ, R} = a{P,R} + b{Q,R}
    P = P_q0 * P_q0 # q0^2
    Q = P_p0 * P_p0 # p0^2
    R = P_q0 + P_p0 # q0 + p0
    a = 2.0
    b = 3.0
    LHS = (a*P + b*Q).poisson_bracket(R)
    RHS = a * P.poisson_bracket(R) + b * Q.poisson_bracket(R)
    assert_poly_equal(LHS, RHS)
    # Test second slot linearity similarly if needed

    # Jacobi {P,{Q,R}} + {Q,{R,P}} + {R,{P,Q}} = 0
    P = P_q0
    Q = P_p0
    R = P_q0 * P_p0 # q0*p0 from fixture test
    P_Q_R = P.poisson_bracket(Q.poisson_bracket(R)) # {q0, {p0, q0p0}} = {q0, q0} = 0
    Q_R_P = Q.poisson_bracket(R.poisson_bracket(P)) # {p0, {q0p0, q0}} = {p0, -p0} = 0
    R_P_Q = R.poisson_bracket(P.poisson_bracket(Q)) # {q0p0, {q0, p0}} = {q0p0, 1} = 0
    JacobiSum = P_Q_R + Q_R_P + R_P_Q
    assert_poly_equal(JacobiSum, P_zero)

    # Bracket with Hamiltonian H = 0.5*p0^2 + q0^2
    # {H, H} = 0
    assert_poly_equal(P_H_simple.poisson_bracket(P_H_simple), P_zero)

    # {H, q0} = -dH/dp0 = -p0  (note the negative sign in this implementation)
    PB_H_q0 = P_H_simple.poisson_bracket(P_q0)
    P_neg_p0 = -1 * P_p0
    assert_poly_equal(PB_H_q0, P_neg_p0)

    # {H, p0} = dH/dq0 = 2*q0  (note the positive sign in this implementation)
    PB_H_p0 = P_H_simple.poisson_bracket(P_p0)
    P_2q0 = 2.0 * P_q0
    assert_poly_equal(PB_H_p0, P_2q0)

def test_get_coefficient(n_vars, P_zero, P_one, P_q0, P_p0, P_mixed, P_H_simple):
    """Test the new get_coefficient method."""
    # Test constants
    assert P_zero.get_coefficient(tuple([0] * n_vars)) == 0.0
    assert P_one.get_coefficient(tuple([0] * n_vars)) == 1.0
    
    # Test simple monomials
    x0_exp = tuple([1] + [0] * (n_vars - 1))
    assert P_q0.get_coefficient(x0_exp) == 1.0
    assert P_q0.get_coefficient(tuple([0] * n_vars)) == 0.0  # Constant term is zero
    
    x1_exp = tuple([0, 1] + [0] * (n_vars - 2))
    assert P_p0.get_coefficient(x1_exp) == 1.0
    
    # Test terms in complex polynomial
    assert P_mixed.get_coefficient(tuple([0] * n_vars)) == 7.0  # Constant term
    assert P_mixed.get_coefficient(tuple([2] + [0] * (n_vars - 1))) == 2.0  # x0^2 term
    assert P_mixed.get_coefficient(tuple([0, 1] + [0] * (n_vars - 2))) == 3.0  # x1 term
    assert P_mixed.get_coefficient(tuple([1, 0, 1] + [0] * (n_vars - 3))) == 4.0  # x0*x2 term
    assert P_mixed.get_coefficient(tuple([0, 0, 0, 3] + [0] * (n_vars - 4))) == -5.0  # x3^3 term
    
    # Test non-existent terms
    assert P_mixed.get_coefficient(tuple([3] + [0] * (n_vars - 1))) == 0.0  # x0^3 term (not in polynomial)
    
    # Test Hamiltonian polynomial
    assert P_H_simple.get_coefficient(tuple([2] + [0] * (n_vars - 1))) == 1.0  # x0^2 term
    assert P_H_simple.get_coefficient(tuple([0, 2] + [0] * (n_vars - 2))) == 0.5  # x1^2 term

    # Test error for wrong exponent length
    with pytest.raises(ValueError):
        P_q0.get_coefficient(tuple([1]))  # Too short
        
    # Test error for negative exponents
    with pytest.raises(ValueError):
        P_q0.get_coefficient(tuple([-1] + [0] * (n_vars - 1)))  # Negative exponent

def test_get_terms(n_vars, P_zero, P_one, P_q0, P_mixed, P_H_simple):
    """Test the new get_terms method."""
    # Test constant polynomial
    terms = list(P_zero.get_terms())
    # A zero polynomial might return no terms or a zero coefficient term
    # depending on implementation, so we allow either
    if terms:
        assert len(terms) == 1
        assert terms[0][0] == tuple([0] * n_vars)
        assert abs(terms[0][1]) < 1e-10  # Coefficient should be zero or very close to it
    else:
        # If no terms, that's valid too for zero polynomial
        pass
    
    terms = list(P_one.get_terms())
    assert len(terms) == 1
    assert terms[0] == (tuple([0] * n_vars), 1.0)
    
    # Test simple monomial
    terms = list(P_q0.get_terms())
    assert len(terms) == 1
    assert terms[0] == (tuple([1] + [0] * (n_vars - 1)), 1.0)
    
    # Test Hamiltonian polynomial
    terms = list(P_H_simple.get_terms())
    # Convert to a dictionary for easier comparison (order may vary)
    terms_dict = dict(terms)
    
    # Hamiltonian polynomial may include zero constant term or not
    # Filter out terms with zero or very small coefficients
    filtered_terms = {k: v for k, v in terms_dict.items() if abs(v) > 1e-10}
    assert len(filtered_terms) == 2  # Two non-zero terms
    
    # Check the expected terms
    assert abs(filtered_terms[tuple([2] + [0] * (n_vars - 1))] - 1.0) < 1e-10  # x0^2 term
    assert abs(filtered_terms[tuple([0, 2] + [0] * (n_vars - 2))] - 0.5) < 1e-10  # 0.5*x1^2 term
    
    # Test complex polynomial
    terms = list(P_mixed.get_terms())
    terms_dict = dict(terms)
    
    # Filter out terms with zero coefficients (if any)
    filtered_terms = {k: v for k, v in terms_dict.items() if abs(v) > 1e-10}
    assert len(filtered_terms) == 5  # Five non-zero terms
    
    # Check the expected terms
    assert abs(filtered_terms[tuple([0] * n_vars)] - 7.0) < 1e-10  # Constant term
    assert abs(filtered_terms[tuple([2] + [0] * (n_vars - 1))] - 2.0) < 1e-10  # x0^2 term
    assert abs(filtered_terms[tuple([0, 1] + [0] * (n_vars - 2))] - 3.0) < 1e-10  # x1 term
    assert abs(filtered_terms[tuple([1, 0, 1] + [0] * (n_vars - 3))] - 4.0) < 1e-10  # x0*x2 term
    assert abs(filtered_terms[tuple([0, 0, 0, 3] + [0] * (n_vars - 4))] + 5.0) < 1e-10  # x3^3 term (-5.0)
    
    # Verify integrity of terms returned from get_terms
    # by reconstructing the polynomial
    reconstructed = Polynomial.zero(n_vars=n_vars)
    for exp, coeff in P_mixed.get_terms():
        # Skip terms with zero coefficients
        if abs(coeff) < 1e-10:
            continue
            
        term_str = str(coeff)
        for i, power in enumerate(exp):
            if power > 0:
                if power == 1:
                    term_str += f"*x{i}"
                else:
                    term_str += f"*x{i}**{power}"
        term_poly = Polynomial(term_str, n_vars=n_vars)
        reconstructed = reconstructed + term_poly
    
    # Check that the reconstructed polynomial equals the original
    assert_poly_equal(reconstructed, P_mixed)

def run_all_tests():
    """Run all test functions in this module."""
    n_vars = 6
    
    P_zero = Polynomial.zero(n_vars=n_vars)
    P_one = Polynomial.one(n_vars=n_vars)
    P_q0 = Polynomial('x0', n_vars=n_vars)
    P_p0 = Polynomial('x1', n_vars=n_vars)
    P_qp_sum = P_q0 + P_p0
    P_q1 = Polynomial('x2', n_vars=n_vars)
    P_p1 = Polynomial('x3', n_vars=n_vars)
    P_H_simple = Polynomial('0.5*x1**2 + x0**2', n_vars=n_vars)
    P_mixed = Polynomial('2*x0**2 + 3*x1 + 4*x0*x2 - 5*x3**3 + 7', n_vars=n_vars)
    
    test_initialization(n_vars, P_zero, P_one, P_q0)
    test_addition_subtraction(P_zero, P_one, P_q0, P_p0, P_qp_sum)
    test_multiplication(P_zero, P_one, P_q0, P_p0, P_qp_sum)
    test_differentiation(n_vars, P_zero, P_one, P_q0, P_p0, P_q1, P_H_simple)
    test_poisson_bracket(P_zero, P_one, P_q0, P_p0, P_q1, P_p1, P_H_simple)
    test_get_coefficient(n_vars, P_zero, P_one, P_q0, P_p0, P_mixed, P_H_simple)
    test_get_terms(n_vars, P_zero, P_one, P_q0, P_mixed, P_H_simple)
    
    print("All polynomial tests passed!")

