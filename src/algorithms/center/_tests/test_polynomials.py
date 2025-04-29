import pytest
import numpy as np
from collections import defaultdict

from algorithms.center.polynomials import Polynomial

# --- Helper for comparing polynomials ---
def assert_poly_equal(p1, p2):
    """Asserts that two Polynomial objects are equal."""
    assert isinstance(p1, Polynomial)
    assert isinstance(p2, Polynomial)
    assert p1.n_vars == p2.n_vars
    # Check keys and values using np.isclose for coefficients
    assert len(p1.coeffs) == len(p2.coeffs)
    for exp, coeff1 in p1.coeffs.items():
        assert exp in p2.coeffs, f"Exponent {exp} missing in second polynomial"
        coeff2 = p2.coeffs[exp]
        assert np.isclose(coeff1, coeff2), f"Coefficients for {exp} differ: {coeff1} vs {coeff2}"

# --- Pytest Fixtures ---
@pytest.fixture
def n_vars():
    return 6 # Default for 3 DOF CR3BP

@pytest.fixture
def P_zero(n_vars):
    return Polynomial(n_vars=n_vars)

@pytest.fixture
def P_one(n_vars):
    zero_exp = tuple([0] * n_vars)
    return Polynomial({zero_exp: 1.0}, n_vars=n_vars)

@pytest.fixture
def P_q0(n_vars): # q1 in the text
    exp = [0] * n_vars
    exp[0] = 1
    return Polynomial({tuple(exp): 1.0}, n_vars=n_vars)

@pytest.fixture
def P_p0(n_vars): # p1 in the text
    exp = [0] * n_vars
    exp[1] = 1
    return Polynomial({tuple(exp): 1.0}, n_vars=n_vars)

@pytest.fixture
def P_q1(n_vars): # q2 in the text
    exp = [0] * n_vars
    exp[2] = 1
    return Polynomial({tuple(exp): 1.0}, n_vars=n_vars)

@pytest.fixture
def P_p1(n_vars): # p2 in the text
    exp = [0] * n_vars
    exp[3] = 1
    return Polynomial({tuple(exp): 1.0}, n_vars=n_vars)

@pytest.fixture
def P_qp_sum(P_q0, P_p0): # q0 + p0
    return P_q0 + P_p0

@pytest.fixture
def P_H_simple(n_vars): # Simple Hamiltonian H = 0.5*p0^2 + q0^2
    exp_p0_sq = [0] * n_vars
    exp_p0_sq[1] = 2
    exp_q0_sq = [0] * n_vars
    exp_q0_sq[0] = 2
    return Polynomial({tuple(exp_p0_sq): 0.5, tuple(exp_q0_sq): 1.0}, n_vars=n_vars)

# --- Test Functions ---

def test_initialization(n_vars, P_zero, P_one, P_q0):
    assert P_zero.n_vars == n_vars
    assert len(P_zero) == 0
    assert P_zero.coeffs == {}

    assert P_one.n_vars == n_vars
    assert len(P_one) == 1
    assert np.isclose(P_one.coeffs[tuple([0]*n_vars)], 1.0)

    assert P_q0.n_vars == n_vars
    assert len(P_q0) == 1
    exp_q0 = tuple([1] + [0]*(n_vars-1))
    assert np.isclose(P_q0.coeffs[exp_q0], 1.0)

    # Test zero coefficient removal
    p_with_zero = Polynomial({(1,0,0,0,0,0): 1.0, (0,1,0,0,0,0): 0.0}, n_vars=n_vars)
    assert len(p_with_zero) == 1
    assert (0,1,0,0,0,0) not in p_with_zero.coeffs

    # Test wrong exponent length
    with pytest.raises(ValueError):
        Polynomial({(1, 0): 1.0}, n_vars=n_vars) # Only 2 exponents given

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
    exp_q0 = tuple([1] + [0]*(P_q0.n_vars-1))
    zero_exp = tuple([0]*P_q0.n_vars)
    expected = Polynomial({exp_q0: 1.0, zero_exp: 5.0}, n_vars=P_q0.n_vars)
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
    exp_q0 = tuple([1] + [0]*(P_q0.n_vars-1))
    expected_3q0 = Polynomial({exp_q0: 3.0}, n_vars=P_q0.n_vars)
    assert_poly_equal(P_3q0, expected_3q0)
    assert_poly_equal(P_q0 * 3.0, expected_3q0)

    # Known results
    P_q0p0 = P_q0 * P_p0
    exp_q0p0 = tuple([1, 1] + [0]*(P_q0.n_vars-2))
    expected_q0p0 = Polynomial({exp_q0p0: 1.0}, n_vars=P_q0.n_vars)
    assert_poly_equal(P_q0p0, expected_q0p0)

    # (q0+p0)^2 = q0^2 + p0^2 + 2*q0*p0
    P_sq = P_qp_sum * P_qp_sum
    exp_q0_sq = tuple([2] + [0]*(P_q0.n_vars-1))
    exp_p0_sq = tuple([0, 2] + [0]*(P_q0.n_vars-2))
    exp_q0p0_term = tuple([1, 1] + [0]*(P_q0.n_vars-2))
    expected_sq = Polynomial({
        exp_q0_sq: 1.0,
        exp_p0_sq: 1.0,
        exp_q0p0_term: 2.0
    }, n_vars=P_q0.n_vars)
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
    exp_q0 = tuple([1] + [0]*(n_vars-1))
    expected_dH_dq0 = Polynomial({exp_q0: 2.0}, n_vars=n_vars)
    assert_poly_equal(dH_dq0, expected_dH_dq0)

    dH_dp0 = P_H_simple.differentiate(1) # Should be p0
    exp_p0 = tuple([0, 1] + [0]*(n_vars-2))
    expected_dH_dp0 = Polynomial({exp_p0: 1.0}, n_vars=n_vars)
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

    # {H, q0} = dH/dp0 = p0
    PB_H_q0 = P_H_simple.poisson_bracket(P_q0)
    assert_poly_equal(PB_H_q0, P_p0)

    # {H, p0} = -dH/dq0 = -2*q0
    PB_H_p0 = P_H_simple.poisson_bracket(P_p0)
    P_neg2q0 = -2.0 * P_q0
    assert_poly_equal(PB_H_p0, P_neg2q0)

def run_all_tests():
    """Run all test functions in this module."""
    test_initialization(6, Polynomial(n_vars=6), Polynomial({(0,0,0,0,0,0): 1.0}, n_vars=6), 
                       Polynomial({(1,0,0,0,0,0): 1.0}, n_vars=6))
    
    P_zero = Polynomial(n_vars=6)
    P_one = Polynomial({(0,0,0,0,0,0): 1.0}, n_vars=6)
    P_q0 = Polynomial({(1,0,0,0,0,0): 1.0}, n_vars=6)
    P_p0 = Polynomial({(0,1,0,0,0,0): 1.0}, n_vars=6)
    P_qp_sum = P_q0 + P_p0
    
    test_addition_subtraction(P_zero, P_one, P_q0, P_p0, P_qp_sum)
    
    P_q1 = Polynomial({(0,0,1,0,0,0): 1.0}, n_vars=6)
    P_p1 = Polynomial({(0,0,0,1,0,0): 1.0}, n_vars=6)
    
    P_H_simple = Polynomial({(0,2,0,0,0,0): 0.5, (2,0,0,0,0,0): 1.0}, n_vars=6)
    
    test_multiplication(P_zero, P_one, P_q0, P_p0, P_qp_sum)
    test_differentiation(6, P_zero, P_one, P_q0, P_p0, P_q1, P_H_simple)
    test_poisson_bracket(P_zero, P_one, P_q0, P_p0, P_q1, P_p1, P_H_simple)
    
    print("All polynomial tests passed!")

