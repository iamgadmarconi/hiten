import pytest
import numpy as np
import symengine as se
import sympy as sp
from collections import defaultdict
import random


from algorithms.center.factory import _build_T_polynomials, hamiltonian, physical_to_real_normal, real_normal_to_complex_canonical
from algorithms.center.lie import _get_homogeneous_terms, _select_monomials_for_elimination, lie_transform, _solve_homological_equation
from algorithms.center.core import Polynomial

from system.libration import L1Point


TEST_MU_EARTH_MOON = 0.01215  # Earth-Moon system
TEST_MU_SUN_EARTH = 3.00348e-6  # Sun-Earth system
TEST_MU_SUN_JUPITER = 9.5387e-4  # Sun-Jupiter system
TEST_MU_UNSTABLE = 0.04  # Above Routh's critical value for triangular points
MAX_DEGREE = 6


x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')

q1, q2, q3 = se.symbols('q1 q2 q3')
p1, p2, p3 = se.symbols('p1 p2 p3')

x_rn, y_rn, z_rn = se.symbols('x_rn y_rn z_rn')
px_rn, py_rn, pz_rn = se.symbols('px_rn py_rn pz_rn')


@pytest.fixture()
def lp():
    """Collinear point with μ chosen once for all the tests."""
    mu = 0.0121505856          # Earth-Moon
    return L1Point(mu)

@pytest.fixture()
def H_phys(lp):
    """Full CR3BP Hamiltonian translated so that the point is at the origin."""
    return hamiltonian(lp, MAX_DEGREE)        # Polynomial wrapper

@pytest.fixture()
def H_rn(lp, H_phys):
    """Hamiltonian in real normal form variables."""
    return physical_to_real_normal(lp, H_phys)

@pytest.fixture()
def H_cc(lp, H_rn):
    """Hamiltonian in complex canonical form variables."""
    return real_normal_to_complex_canonical(lp, H_rn)

def test_select_monomials_for_elimination():
    expression = 2*q1*q2*p3 + 4*q1*p1*p2 + 3*q2**2*p2 + 5*q1**2*p1
    H_n = Polynomial([q1, q2, q3, p1, p2, p3], expression)
    H_n_homogeneous = _select_monomials_for_elimination(H_n)
    expression_to_eliminate = 2*q1*q2*p3 + 5*q1**2*p1
    H_n_to_eliminate = Polynomial([q1, q2, q3, p1, p2, p3], expression_to_eliminate)
    assert H_n_homogeneous.expression == H_n_to_eliminate.expression

def test_get_homogeneous_terms():
    expression = 2*q1**2 + 3*q1*q2 + 5*q1*q2*q3 + 7*p1*p2
    H_n = Polynomial([q1, q2, q3, p1, p2, p3], expression)
    H_n_homogeneous = _get_homogeneous_terms(H_n, 2)
    expected_expression = 2*q1**2 + 3*q1*q2 + 7*p1*p2
    H_n_expected = Polynomial([q1, q2, q3, p1, p2, p3], expected_expression)
    assert H_n_homogeneous.expression == H_n_expected.expression

    H_n_homogeneous = _get_homogeneous_terms(H_n, 3)
    expected_expression = 5*q1*q2*q3
    H_n_expected = Polynomial([q1, q2, q3, p1, p2, p3], expected_expression)
    assert H_n_homogeneous.expression == H_n_expected.expression

def test_solve_homological_equation():
    # Create a simple quadratic Hamiltonian (H_2)
    H_2_expression = 0.5*q1*p1 + 1.2*se.I*q2*p2 + 0.8*se.I*q3*p3
    H_2 = Polynomial([q1, q2, q3, p1, p2, p3], H_2_expression)
    
    # Term to eliminate
    term_to_eliminate = 2*q1*q2*p3
    H_n_to_eliminate = Polynomial([q1, q2, q3, p1, p2, p3], term_to_eliminate)
    
    # Set up eta vector (eigenvalues)
    eta_vector = [0.5, 1.2*se.I, 0.8*se.I]

    # Call the function
    G_n = _solve_homological_equation(H_n_to_eliminate, H_2, eta_vector, [q1, q2, q3, p1, p2, p3])
    
    # Verify the result
    # Manually calculate what the result should be
    # For term 2*q1*q2*p3:
    # kq = (1,1,0), kp = (0,0,1)
    # kp - kq = (-1,-1,1)
    # denominator = (-1)*0.5 + (-1)*1.2j + (1)*0.8j = -0.5 - 0.4j
    # coefficient = -2/(-0.5-0.4j) ≈ 2.94 - 2.35j
    
    # Get the coefficient of q1*q2*p3 in G_n
    coeff = None
    for c, monomial in G_n.iter_terms():
        if monomial.kq == (1, 1, 0) and monomial.kp == (0, 0, 1):
            coeff = c
            break
    
    assert coeff is not None, "Expected term not found in result"
    
    # Expected coefficient (calculated manually)
    expected_coeff = -2 / (-0.5 - 0.4j)
    
    # Check that the coefficient is close to expected
    assert abs(coeff - expected_coeff) < 1e-10, f"Expected {expected_coeff}, got {coeff}"
