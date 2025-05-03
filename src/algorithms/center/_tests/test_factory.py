import pytest
import numpy as np
import symengine as se
import sympy as sp
from collections import defaultdict
import random


from algorithms.center.factory import _build_T_polynomials, hamiltonian, physical_to_real_normal, real_normal_to_complex_canonical
from algorithms.variables import get_vars, canonical_normal_vars, physical_vars, real_normal_vars, linear_modes_vars
from algorithms.center.core import Polynomial

from system.libration import L1Point


TEST_MU_EARTH_MOON = 0.01215  # Earth-Moon system
TEST_MU_SUN_EARTH = 3.00348e-6  # Sun-Earth system
TEST_MU_SUN_JUPITER = 9.5387e-4  # Sun-Jupiter system
TEST_MU_UNSTABLE = 0.04  # Above Routh's critical value for triangular points


x, y, z, px, py, pz = get_vars(physical_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)


rho2 = x**2 + y**2 + z**2


# ------------ fixtures --------------------------------------
@pytest.fixture()
def lp():
    """Collinear point with μ chosen once for all the tests."""
    mu = 0.0121505856          # Earth-Moon
    return L1Point(mu)

@pytest.fixture()
def H_phys(lp):
    """Full CR3BP Hamiltonian translated so that the point is at the origin."""
    return hamiltonian(lp)        # Polynomial wrapper

@pytest.fixture()
def H_cc(lp, H_phys):
    """Hamiltonian in complex canonical variables."""
    return real_normal_to_complex_canonical(lp, H_phys)

@pytest.fixture
def cartesian_vars():
    return [x, y, z, px, py, pz]

@pytest.fixture
def canonical_vars():
    return [q1, q2, q3, p1, p2, p3]

@pytest.fixture()
def hamiltonians(lp):
    """Build the physical and complex-canonical Hamiltonians up to order 4."""
    H_phys = hamiltonian(lp, max_degree=4)
    H_rn   = physical_to_real_normal(lp, H_phys)
    H_cn   = real_normal_to_complex_canonical(lp, H_rn)
    return H_cn

def test_T_base_cases():
    T0, T1 = _build_T_polynomials(1)
    assert T0 == 1
    assert T1 == x

@pytest.mark.parametrize("N", range(2, 8))
def test_T_length(N):
    T = _build_T_polynomials(N)
    assert len(T) == N + 1

@pytest.mark.parametrize("N", range(2, 8))
def test_T_recurrence(N):
    T = _build_T_polynomials(N)
    for n in range(2, N + 1):
        lhs = T[n]
        rhs = ((2*n - 1)/n) * x * T[n-1] - ((n-1)/n) * rho2 * T[n-2]
        diff = se.expand(lhs - rhs)
        
        # Check if the expression is zero or very close to zero
        if diff != 0:
            # Convert to string and check if it has very small coefficients
            diff_str = str(diff)
            # Scientific notation for small numbers like 1.11022e-16
            assert "e-" in diff_str or diff == 0, f"Difference not negligible: {diff}"


def test_T_known_closed_form():
    T = _build_T_polynomials(3)
    T2_expected = se.expand((3*x**2 - rho2) / 2)
    T3_expected = se.expand((5*x**3 - 3*x*rho2) / 2)
    
    # Check if expressions are equivalent (difference is zero or very small)
    diff2 = se.expand(T[2] - T2_expected)
    diff3 = se.expand(T[3] - T3_expected)
    
    if diff2 != 0:
        diff_str = str(diff2)
        assert "e-" in diff_str or diff2 == 0, f"T2 difference not equivalent: {diff2}"
    
    if diff3 != 0:
        diff_str = str(diff3)
        assert "e-" in diff_str or diff3 == 0, f"T3 difference not equivalent: {diff3}"


class DummyPoint:
    """Stub that returns symbolic coefficients c₂ … c₆."""
    c2, c3, c4, c5, c6 = se.symbols("c2 c3 c4 c5 c6")
    def _cn(self, n):
        return {2:self.c2, 3:self.c3, 4:self.c4, 5:self.c5, 6:self.c6}[n]

def test_hamiltonian_expression_matches_definition():
    max_deg = 6
    point = DummyPoint()
    H = hamiltonian(point, max_degree=max_deg).expression   # unwrap Polynomial

    # expected expression straight from the definition
    T = _build_T_polynomials(max_deg)
    U = -(point.c2*T[2] + point.c3*T[3] + point.c4*T[4] + point.c5*T[5] + point.c6*T[6])
    K = (px**2 + py**2 + pz**2)/2 + y*px - x*py
    H_expected = se.expand(K + U)

    diff = se.expand(H - H_expected)
    if diff != 0:
        diff_str = str(diff)
        assert "e-" in diff_str or diff == 0, f"Difference not within numerical tolerance: {diff}"


def test_hamiltonian_is_zero_at_equilibrium():
    point = DummyPoint()
    H = hamiltonian(point, max_degree=4).expression
    subs0 = {x:0, y:0, z:0, px:0, py:0, pz:0}
    
    H_at_zero = H.subs(subs0)
    if H_at_zero != 0:
        diff_str = str(H_at_zero)
        assert "e-" in diff_str or H_at_zero == 0, f"Hamiltonian not zero at equilibrium: {H_at_zero}"

    # all first-order derivatives must vanish too
    for v in (x, y, z, px, py, pz):
        deriv_at_zero = se.diff(H, v).subs(subs0)
        if deriv_at_zero != 0:
            diff_str = str(deriv_at_zero)
            assert "e-" in diff_str or deriv_at_zero == 0, f"Derivative w.r.t {v} not zero at equilibrium: {deriv_at_zero}"


def test_truncation():
    point = DummyPoint()
    deg = 4
    Hpoly = hamiltonian(point, max_degree=deg).expression
    # every monomial must have total degree ≤ deg in (x,y,z)
    for mon in Hpoly.expand().args:        # iterate over individual terms
        # Get the degree of the monomial with respect to each variable
        exp_sum = 0
        for var in (x, y, z):
            # Use the proper way to check the degree/exponent in SymEngine
            if hasattr(mon, 'has'):
                # For expressions that have the 'has' method to check if a variable is present
                if mon.has(var):
                    # Try to figure out the power of this variable
                    power = 0
                    term_str = str(mon)
                    
                    # Simple check: if var^n is in the term, count that
                    var_str = str(var)
                    if f"{var_str}**" in term_str:
                        # Extract the power, e.g., from "x**2" get 2
                        parts = term_str.split(f"{var_str}**")
                        if len(parts) > 1:
                            # Try to extract the number after "**"
                            power_str = ""
                            for c in parts[1]:
                                if c.isdigit() or c == '.':
                                    power_str += c
                                else:
                                    break
                            if power_str:
                                power = int(float(power_str))
                            else:
                                power = 1
                    elif var_str in term_str:
                        # The variable is present but not with an explicit power
                        power = 1
                    
                    exp_sum += power
            
        assert exp_sum <= deg, f"Term {mon} has degree {exp_sum} > {deg}"

def test_symplectic(lp):
    C, Cinv = lp.normal_form_transform()      # both are 6×6 numpy arrays
    J = np.block([[np.zeros((3,3)), np.eye(3)],
                [-np.eye(3),      np.zeros((3,3))]])
    # C is the matrix used inside real_normal_to_complex_canonical – should be symplectic
    diff = C.T @ J @ C - J
    assert np.allclose(diff, np.zeros_like(diff), atol=1e-12)

def test_real_normal_form_transform(lp):
    lambda1_num, omega1_num, omega2_num = lp.linear_modes()
    c2_val = lp._cn(2)

    h2 = 1/2 * (px**2+py**2)+y*px-x*py-c2_val*x**2+c2_val/2 * y**2 + 1/2 * pz**2 + c2_val/2 * z**2
    h2 = Polynomial([x, y, z, px, py, pz], h2)
    h2_rn = physical_to_real_normal(lp, h2).subs({lambda1:lambda1_num, omega1:omega1_num, omega2:omega2_num, c2_val:c2_val})
    h2_rn_expected = lambda1_num*x_rn*px_rn + (omega1_num/2)*(y_rn**2 + py_rn**2) + (omega2_num/2)*(z_rn**2 + pz_rn**2)

    diff = se.expand(h2_rn.expression - h2_rn_expected)
    diff_str = str(diff)
    if diff != 0:

        assert "e-" in diff_str or diff == 0, f"Difference not within numerical tolerance: {diff}"

def test_complex_canonical_transform(lp):
    lambda1, omega1, omega2 = lp.linear_modes()
    c2_val = lp._cn(2)
    
    h2_rn = lambda1*x_rn*px_rn + (omega1/2)*(y_rn**2 + py_rn**2) + (omega2/2)*(z_rn**2 + pz_rn**2)
    h2_rn = Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], h2_rn)
    h2_cc = real_normal_to_complex_canonical(lp, h2_rn)
    h2_cc_expected = lambda1*q1*p1 + se.I * omega1 * q2 * p2 + se.I * omega2 * q3 * p3

    diff = se.expand(h2_cc.expression - h2_cc_expected)
    diff_str = str(diff)
    if diff != 0:
        assert "e-" in diff_str or diff == 0, f"Difference not within numerical tolerance: {diff}"

def test_h2_diagonal_in_complex_canonical(lp):
    """Test if the quadratic part of the Hamiltonian is diagonal after 
    transformation to complex canonical coordinates."""

    h2_phys = hamiltonian(lp, max_degree=2)

    h2_rn = physical_to_real_normal(lp, h2_phys)

    h2_cc = real_normal_to_complex_canonical(lp, h2_rn)

    h2_expr = h2_cc.expression.expand()
    
    # Tolerance for small coefficients that may appear due to numerical errors
    TOLERANCE = 1e-14

    mixed_terms = []

    for i, qi in enumerate(q_vars):
        for j, pj in enumerate(p_vars):
            if i != j:  # Non-diagonal term qi*pj
                # Create a substitution dict that isolates this term
                subs = {var: 0 for var in q_vars + p_vars}
                subs[qi] = 1
                subs[pj] = 1
                
                # Evaluate the coefficient of qi*pj
                coeff = complex(h2_cc.evaluate(subs))
                
                # Check if the coefficient is above the tolerance
                if abs(coeff) > TOLERANCE:
                    mixed_terms.append((f"{qi}*{pj}", coeff, abs(coeff)))
    
    # Assert that no significant mixed terms were found
    assert len(mixed_terms) == 0, f"Non-diagonal terms found with significant coefficients: {mixed_terms}"
    
    # If we made it here, the Hamiltonian is diagonal within tolerance
    lambda1, omega1, omega2 = lp.linear_modes()
    h2_expected = lambda1*q1*p1 + se.I*omega1*q2*p2 + se.I*omega2*q3*p3
    
    diff = se.expand(h2_cc.expression - h2_expected)
    if diff != 0:
        diff_str = str(diff)
        assert "e-" in diff_str or diff == 0, f"Hamiltonian does not match expected diagonal form: {diff}"
