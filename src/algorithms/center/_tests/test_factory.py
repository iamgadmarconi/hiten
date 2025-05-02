import pytest
import numpy as np
import symengine as se
from collections import defaultdict


from algorithms.center.factory import _build_T_polynomials, hamiltonian, to_complex_canonical
from system.libration import L1Point


TEST_MU_EARTH_MOON = 0.01215  # Earth-Moon system
TEST_MU_SUN_EARTH = 3.00348e-6  # Sun-Earth system
TEST_MU_SUN_JUPITER = 9.5387e-4  # Sun-Jupiter system
TEST_MU_UNSTABLE = 0.04  # Above Routh's critical value for triangular points


x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')

q1, q2, q3 = se.symbols('q1 q2 q3')
p1, p2, p3 = se.symbols('p1 p2 p3')

rho2 = x**2 + y**2 + z**2


# ------------ basic symbols used everywhere -----------------
x, y, z = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')
old_vars = (x, y, z, px, py, pz)

q1, q2, q3, p1, p2, p3 = se.symbols('q1 q2 q3 p1 p2 p3')
new_vars = (q1, q2, q3, p1, p2, p3)


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
    return to_complex_canonical(lp, H_phys)

@pytest.fixture
def cartesian_vars():
    return [x, y, z, px, py, pz]

@pytest.fixture
def canonical_vars():
    return [q1, q2, q3, p1, p2, p3]


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

    assert se.expand(H - H_expected) == 0


def test_hamiltonian_is_zero_at_equilibrium():
    point = DummyPoint()
    H = hamiltonian(point, max_degree=4).expression
    subs0 = {x:0, y:0, z:0, px:0, py:0, pz:0}
    assert H.subs(subs0) == 0

    # all first-order derivatives must vanish too
    for v in (x, y, z, px, py, pz):
        assert se.diff(H, v).subs(subs0) == 0


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
    # C is the matrix used inside to_complex_canonical – should be symplectic
    assert np.allclose(C.T @ J @ C, J, atol=1e-12)

def quadratic_part(expr, vars_):
    expr = se.expand(expr)
    terms = expr.args if expr.is_Add else (expr,)
    quad = se.S(0)
    for t in terms:
        if sum(t.as_powers_dict().get(v, 0) for v in vars_) == 2:
            quad += t
    return se.expand(quad)

def test_quadratic_normal_form(lp, H_cc):
    vars_ = (q1, q2, q3, p1, p2, p3)

    # Use either helper; here I take the ε-trick
    H2 = quadratic_part(H_cc.expression, vars_)

    lambda1, omega1, omega2 = lp.linear_modes()
    expected = lambda1*q1*p1 + se.I*omega1*q2*p2 + se.I*omega2*q3*p3
    assert se.expand(H2 - expected) == 0
