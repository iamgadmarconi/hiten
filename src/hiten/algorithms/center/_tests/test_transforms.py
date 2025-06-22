import numpy as np
import pytest
import sympy as sp

from hiten.algorithms.center.hamiltonian import build_physical_hamiltonian
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from hiten.algorithms.polynomial.conversion import poly2sympy, sympy2poly
from hiten.algorithms.polynomial.operations import (
    _linear_variable_polys, polynomial_add_inplace, polynomial_multiply,
    polynomial_poisson_bracket, polynomial_power, polynomial_variable,
    polynomial_zero_list)
from hiten.algorithms.center.transforms import (_local2realmodal, solve_complex,
                                          solve_real, substitute_complex,
                                          substitute_linear, substitute_real)
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.utils.constants import Constants

TEST_MAX_DEG = 6
TEST_L_POINT_IDX = 1


@pytest.fixture(scope="module")
def transforms_test_setup():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    distance = Constants.get_orbital_distance("earth", "moon")
    system = System(systemConfig(Earth, Moon, distance))
    libration_point = system.get_libration_point(TEST_L_POINT_IDX)

    psi, clmo = init_index_tables(TEST_MAX_DEG)
    encode_dict = _create_encode_dict_from_clmo(clmo)

    return psi, clmo, encode_dict, libration_point

def sympy_reference(point) -> sp.Expr:
    x, y, z, px, py, pz = sp.symbols("x y z px py pz")
    vars_tuple = (x, y, z, px, py, pz)

    rho2 = x**2 + y**2 + z**2
    rho = sp.sqrt(rho2)

    H = sp.Rational(1, 2) * (px**2 + py**2 + pz**2) + y * px - x * py

    for n in range(2, TEST_MAX_DEG + 1):
        cn = point._cn(n)
        Pn_expr = sp.legendre(n, x / rho)
        term_to_add = sp.simplify(cn * rho**n * Pn_expr)
        H -= term_to_add
    
    expanded_H = sp.simplify(H)

    try:
        poly_obj = sp.Poly(expanded_H, *vars_tuple)
        return poly_obj.as_expr()
    except sp.PolynomialError as e:
        raise ValueError from e


def test_linear_variable_polys(transforms_test_setup):

    psi, clmo, encode_dict, _ = transforms_test_setup

    C_base = np.array([
        [1., 2., 0., 0.5, 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [3., 0., 0., 0., 1., 0.2],
        [0., 0., 0., 0., 0., 1.],
    ], dtype=float)

    C = C_base.astype(np.complex128)

    L_actual = _linear_variable_polys(C, TEST_MAX_DEG, psi, clmo, encode_dict)

    new_basis_expected = [polynomial_variable(j, TEST_MAX_DEG, psi, clmo, encode_dict) for j in range(6)]
    
    L_expected = []
    for i in range(6):
        pol_expected_i = polynomial_zero_list(TEST_MAX_DEG, psi)
        for j in range(6):
            if C[i, j] != 0: 
                polynomial_add_inplace(pol_expected_i, new_basis_expected[j], C[i, j], TEST_MAX_DEG)
        L_expected.append(pol_expected_i)

    assert len(L_actual) == 6, f"L_actual should have 6 polynomials"
    assert len(L_expected) == 6, f"L_expected should have 6 polynomials"
    
    for i in range(6):
        poly_actual_i = L_actual[i]
        poly_expected_i = L_expected[i]

        assert len(poly_actual_i) == TEST_MAX_DEG + 1, f"Mismatch in num degree slices for L_actual[{i}]"
        assert len(poly_expected_i) == TEST_MAX_DEG + 1, f"Mismatch in num degree slices for L_expected[{i}]"

        for deg_idx in range(TEST_MAX_DEG + 1):
            coeffs_actual = poly_actual_i[deg_idx]
            coeffs_expected = poly_expected_i[deg_idx]
            
            assert np.allclose(
                coeffs_actual, coeffs_expected, atol=1e-15, rtol=1e-12
            ), (f"Mismatch for old_var {i}, degree {deg_idx}.\n"
                f"Actual: {coeffs_actual}\nExpected: {coeffs_expected}")


def test_substitute_linear(transforms_test_setup):
    """Test the substitute_linear function for correctness."""
    psi, clmo, encode_dict, _ = transforms_test_setup

    def create_const_poly(val, max_deg_local, psi_local):
        p = polynomial_zero_list(max_deg_local, psi_local)
        p[0][0] = val
        return p

    # Test Case 0: H_old is a constant
    H_old0 = polynomial_zero_list(TEST_MAX_DEG, psi)
    const_val = 5.0 - 2.0j
    H_old0[0][0] = const_val
    
    C0 = np.array([[2.0, 1.0, 0,0,0,0],
                     [0.5, 1.0, 0,0,0,0],
                     [0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.complex128)

    H_actual0 = substitute_linear(H_old0, C0, TEST_MAX_DEG, psi, clmo, encode_dict) # Pass local encode_dict
    # Expected is just H_old0 itself, as constants are unaffected by variable substitution.
    for d_idx in range(TEST_MAX_DEG + 1):
        assert np.allclose(H_actual0[d_idx], H_old0[d_idx], atol=1e-15, rtol=1e-12), \
            f"SubstLinear TC0 (const) failed: max_deg={TEST_MAX_DEG}, d_idx={d_idx}"

    if TEST_MAX_DEG >= 1:
        H_old1 = polynomial_zero_list(TEST_MAX_DEG, psi)
        c0_val = 2.0 + 1.0j
        c1_val = 3.0 - 0.5j

        k_x0 = tuple([1 if i == 0 else 0 for i in range(6)])
        idx_x0 = encode_multiindex(np.array(k_x0, dtype=np.int64), 1, encode_dict)
        H_old1[1][idx_x0] = c0_val

        k_x1 = tuple([1 if i == 1 else 0 for i in range(6)])
        idx_x1 = encode_multiindex(np.array(k_x1, dtype=np.int64), 1, encode_dict)
        H_old1[1][idx_x1] = c1_val
        
        C1 = np.identity(6, dtype=np.complex128)
        C1[0,1] = 0.5 + 0.2j # x_old_0 = 1*x_new_0 + (0.5+0.2j)*x_new_1
        C1[1,0] = 0.3 - 0.1j # x_old_1 = (0.3-0.1j)*x_new_0 + 1*x_new_1

        H_actual1 = substitute_linear(H_old1, C1, TEST_MAX_DEG, psi, clmo, encode_dict)
        
        L1 = _linear_variable_polys(C1, TEST_MAX_DEG, psi, clmo, encode_dict)
        
        const_poly_c0 = create_const_poly(c0_val, TEST_MAX_DEG, psi)
        const_poly_c1 = create_const_poly(c1_val, TEST_MAX_DEG, psi)

        term_for_c0_x_old_0 = polynomial_multiply(const_poly_c0, L1[0], TEST_MAX_DEG, psi, clmo, encode_dict)
        term_for_c1_x_old_1 = polynomial_multiply(const_poly_c1, L1[1], TEST_MAX_DEG, psi, clmo, encode_dict)
        
        H_expected1 = polynomial_zero_list(TEST_MAX_DEG, psi)
        polynomial_add_inplace(H_expected1, term_for_c0_x_old_0, 1.0, TEST_MAX_DEG)
        polynomial_add_inplace(H_expected1, term_for_c1_x_old_1, 1.0, TEST_MAX_DEG)

        for d_idx in range(TEST_MAX_DEG + 1):
            assert np.allclose(H_actual1[d_idx], H_expected1[d_idx], atol=1e-15, rtol=1e-12), \
                f"SubstLinear TC1 (linear) failed: max_deg={TEST_MAX_DEG}, d_idx={d_idx}"

    # Test Case 2: H_old = c_sq * (x_old_0)^2 (only if max_deg_test >= 2)
    if TEST_MAX_DEG >= 2:
        H_old2 = polynomial_zero_list(TEST_MAX_DEG, psi)
        c_sq_val = 1.5 + 0.5j
        
        k_x0sq = tuple([2 if i == 0 else 0 for i in range(6)])
        idx_x0sq = encode_multiindex(np.array(k_x0sq, dtype=np.int64), 2, encode_dict)
        H_old2[2][idx_x0sq] = c_sq_val

        C2 = np.identity(6, dtype=np.complex128)
        C2[0,0] = 1.2 - 0.3j 
        C2[0,1] = 0.7 + 0.4j # x_old_0 = C2[0,0]*x_new_0 + C2[0,1]*x_new_1
        
        H_actual2 = substitute_linear(H_old2, C2, TEST_MAX_DEG, psi, clmo, encode_dict)
        
        L2 = _linear_variable_polys(C2, TEST_MAX_DEG, psi, clmo, encode_dict)
        const_poly_c_sq = create_const_poly(c_sq_val, TEST_MAX_DEG, psi)
        
        powered_L0 = polynomial_power(L2[0], 2, TEST_MAX_DEG, psi, clmo, encode_dict)
        H_expected2 = polynomial_multiply(const_poly_c_sq, powered_L0, TEST_MAX_DEG, psi, clmo, encode_dict)

        for d_idx in range(TEST_MAX_DEG + 1):
            assert np.allclose(H_actual2[d_idx], H_expected2[d_idx], atol=1e-14, rtol=1e-11), \
                f"SubstLinear TC2 (quad) failed: max_deg={TEST_MAX_DEG}, d_idx={d_idx}"


def test_identity(transforms_test_setup):
    psi, clmo, encode_dict, _ = transforms_test_setup
    I = np.eye(6)
    _sympy_vars = sp.symbols("x y z px py pz")

    # random polynomial with integer coefficients in [‑3, 3]
    rng = np.random.default_rng(0)
    coeffs = rng.integers(-3, 4, size=20)  # 20 random terms

    expr = 0
    for c in coeffs:
        exps = rng.integers(0, 3, size=6)
        if sum(exps) > TEST_MAX_DEG:
            continue
        mon = 1
        for v, k in zip(_sympy_vars, exps):
            mon *= v**int(k)
        expr += int(c) * mon

    P = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict)

    while len(P) < TEST_MAX_DEG + 1:
        P.append(polynomial_zero_list(len(P), psi)[0])

    P_sub = substitute_linear(P, I, TEST_MAX_DEG, psi, clmo, encode_dict)

    assert all(
        np.allclose(a, b, atol=1e-14, rtol=1e-12) for a, b in zip(P, P_sub)
    ), "Identity substitution should return an identical polynomial (within numerical precision)."


def test_permutation(transforms_test_setup):
    psi, clmo, encode_dict, _ = transforms_test_setup

    Pmat = np.eye(6)
    Pmat[[0, 1]] = Pmat[[1, 0]]
    
    x, y, z, px, py, pz = sp.symbols("x y z px py pz")
    _sympy_vars = (x, y, z, px, py, pz)

    expr = x**2 + 2*y*px + 3*z**2

    P_old = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict)
    
    while len(P_old) < TEST_MAX_DEG + 1:
        P_old.append(polynomial_zero_list(len(P_old), psi)[0])
        
    P_new = substitute_linear(P_old, Pmat, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    expected_expr = y**2 + 2*x*px + 3*z**2
    expr_test = poly2sympy(P_new, _sympy_vars, psi, clmo)

    diff = sp.expand(expected_expr - expr_test)
    
    assert diff == 0, f"Mismatch for permutation test with max_deg={TEST_MAX_DEG}. Difference: {diff}"

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_matrix(seed, transforms_test_setup):
    psi, clmo, encode_dict, _ = transforms_test_setup

    x, y, z, px, py, pz = sp.symbols("x y z px py pz")
    _sympy_vars = (x, y, z, px, py, pz)

    rng = np.random.default_rng(seed)

    C = rng.integers(-2, 3, size=(6, 6))
    while np.linalg.matrix_rank(C) < 6:
        C = rng.integers(-2, 3, size=(6, 6))
    
    C = C.T

    coeffs = rng.integers(-5, 6, size=15)
    expr = 0
    for c in coeffs:
        exps = rng.integers(0, 3, size=6)
        if sum(exps) > TEST_MAX_DEG:
            continue
        mon = 1
        for v, k in zip(_sympy_vars, exps):
            mon *= v**int(k)
        expr += int(c) * mon

    P_old = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict)
    
    while len(P_old) < TEST_MAX_DEG + 1:
        P_old.append(polynomial_zero_list(len(P_old), psi)[0])
        
    P_new = substitute_linear(P_old, C, TEST_MAX_DEG, psi, clmo, encode_dict)

    x_old = np.array(_sympy_vars[:6])
    subs_dict = {x_old[i]: sum(int(C[i, j]) * x_old[j] for j in range(6)) for i in range(6)}
    expr_truth = expr.xreplace(subs_dict)
    expr_test = poly2sympy(P_new, _sympy_vars, psi, clmo)

    assert sp.expand(expr_truth - expr_test) == 0, f"Mismatch for seed {seed} and degree {TEST_MAX_DEG}"


def test_symplectic(transforms_test_setup):

    _, _, _, libration_point = transforms_test_setup

    C, _ = libration_point.normal_form_transform()
    J = np.block([[np.zeros((3, 3)),  np.eye(3)],
                  [-np.eye(3),        np.zeros((3, 3))]])
    assert np.allclose(C.T @ J @ C, J, atol=1e-13)


def test_real_normal_form(transforms_test_setup):
    psi, clmo, _, libration_point = transforms_test_setup

    H_phys = build_physical_hamiltonian(libration_point, TEST_MAX_DEG)
    H_rn   = _local2realmodal(libration_point, H_phys, TEST_MAX_DEG, psi, clmo)

    x, y, z, px, py, pz = sp.symbols('x y z px py pz')
    expr = poly2sympy(H_rn, (x, y, z, px, py, pz), psi, clmo)

    # pull out degree-2 terms
    poly = sp.Poly(expr, x, y, z, px, py, pz)
    quad_terms = {m: c for m, c in poly.terms() if sum(m) == 2}
    
    # Filter out terms with negligible coefficients (numerical noise)
    significant_quad_terms = {m: c for m, c in quad_terms.items() if abs(float(c)) > 1e-12}

    allowed = {(1, 0, 0, 1, 0, 0),   # x * px
               (0, 2, 0, 0, 0, 0),   # y**2
               (0, 0, 0, 0, 2, 0),   # py**2
               (0, 0, 2, 0, 0, 0),   # z**2
               (0, 0, 0, 0, 0, 2)}   # pz**2

    assert set(significant_quad_terms).issubset(allowed), (
        "Unexpected quadratic monomials after phys→rn transformation")

    lambda1, omega1, omega2 = libration_point.linear_modes
    
    coeff_xpx = float(significant_quad_terms[(1,0,0,1,0,0)])
    coeff_y2  = float(significant_quad_terms[(0,2,0,0,0,0)])
    coeff_py2 = float(significant_quad_terms[(0,0,0,0,2,0)])
    coeff_z2  = float(significant_quad_terms[(0,0,2,0,0,0)])
    coeff_pz2 = float(significant_quad_terms[(0,0,0,0,0,2)])

    assert np.isclose(coeff_xpx, lambda1, rtol=1e-12)
    assert np.isclose(coeff_y2,  0.5*omega1, rtol=1e-12)
    assert np.isclose(coeff_py2, 0.5*omega1, rtol=1e-12)
    assert np.isclose(coeff_z2,  0.5*omega2, rtol=1e-12)
    assert np.isclose(coeff_pz2, 0.5*omega2, rtol=1e-12)


def test_complex_normal_form(transforms_test_setup):
    psi, clmo, encode_dict, libration_point = transforms_test_setup

    H_phys = build_physical_hamiltonian(libration_point, TEST_MAX_DEG)
    H_rn   = _local2realmodal(libration_point, H_phys, TEST_MAX_DEG, psi, clmo)
    H_cn   = substitute_complex(H_rn, TEST_MAX_DEG, psi, clmo)

    q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
    expr = poly2sympy(H_cn, (q1, q2, q3, p1, p2, p3), psi, clmo)

    quad_terms = {
        m: c for m, c in sp.Poly(expr, q1, q2, q3, p1, p2, p3).terms() if sum(m) == 2
    }

    quad_terms = {
        m: complex(c.evalf()) for m, c in quad_terms.items() if abs(complex(c)) > 1e-12
    }

    allowed = {
        (1, 0, 0, 1, 0, 0): "q1p1",  # q1 * p1  ->  lambda_1      (real)
        (0, 1, 0, 0, 1, 0): "q2p2",  # q2 * p2  ->  i * omega_1    (imag)
        (0, 0, 1, 0, 0, 1): "q3p3",  # q3 * p3  ->  i * omega_2    (imag)
    }

    assert set(quad_terms).issubset(allowed), "Unexpected quadratic monomials after rn→cn"

    lambda1, omega1, omega2 = libration_point.linear_modes

    coeff_q1p1 = quad_terms[(1, 0, 0, 1, 0, 0)]
    coeff_q2p2 = quad_terms[(0, 1, 0, 0, 1, 0)]
    coeff_q3p3 = quad_terms[(0, 0, 1, 0, 0, 1)]

    # real hyperbolic coefficient (should be lambda_1)
    assert np.isclose(coeff_q1p1.real, lambda1, rtol=1e-12)
    assert abs(coeff_q1p1.imag) < 1e-12

    # imaginary elliptic coefficients (should be  i * omega_1 and i * omega_2)
    assert np.isclose(coeff_q2p2 / 1j, omega1, rtol=1e-12)
    assert np.isclose(coeff_q3p3 / 1j, omega2, rtol=1e-12)

    H2 = polynomial_zero_list(TEST_MAX_DEG, psi)
    for d in range(len(H_cn)):
        if d == 2:  # Only copy degree 2 terms
            H2[d] = H_cn[d].copy()
    
    # Create |q2|^2 = q2 * p2 polynomial
    q2_var = polynomial_variable(1, TEST_MAX_DEG, psi, clmo, encode_dict)
    p2_var = polynomial_variable(4, TEST_MAX_DEG, psi, clmo, encode_dict)
    q2p2_poly = polynomial_multiply(q2_var, p2_var, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    # Create |q3|^2 = q3 * p3 polynomial
    q3_var = polynomial_variable(2, TEST_MAX_DEG, psi, clmo, encode_dict)
    p3_var = polynomial_variable(5, TEST_MAX_DEG, psi, clmo, encode_dict)
    q3p3_poly = polynomial_multiply(q3_var, p3_var, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    # Compute the Poisson brackets
    pb_H2_q2p2 = polynomial_poisson_bracket(H2, q2p2_poly, TEST_MAX_DEG, psi, clmo, encode_dict)
    pb_H2_q3p3 = polynomial_poisson_bracket(H2, q3p3_poly, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    # Check that the Poisson brackets are zero (within numerical tolerance)
    for d in range(TEST_MAX_DEG + 1):
        if pb_H2_q2p2[d].size > 0:
            assert np.allclose(pb_H2_q2p2[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q2|^2}} should be zero, but degree {d} terms are not"
        if pb_H2_q3p3[d].size > 0:
            assert np.allclose(pb_H2_q3p3[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q3|^2}} should be zero, but degree {d} terms are not"
    
    # Also test the bracket with hyperbolic action I1 = q1 * p1
    q1_var = polynomial_variable(0, TEST_MAX_DEG, psi, clmo, encode_dict)
    p1_var = polynomial_variable(3, TEST_MAX_DEG, psi, clmo, encode_dict)
    q1p1_poly = polynomial_multiply(q1_var, p1_var, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    pb_H2_q1p1 = polynomial_poisson_bracket(H2, q1p1_poly, TEST_MAX_DEG, psi, clmo, encode_dict)
    
    for d in range(TEST_MAX_DEG + 1):
        if pb_H2_q1p1[d].size > 0:
            assert np.allclose(pb_H2_q1p1[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q1|^2}} should be zero, but degree {d} terms are not"


def test_poly_realification_complexification(transforms_test_setup):
    psi, clmo, _, libration_point = transforms_test_setup

    H_phys = build_physical_hamiltonian(libration_point, TEST_MAX_DEG)
    H_rn   = _local2realmodal(libration_point, H_phys, TEST_MAX_DEG, psi, clmo)
    H_cn   = substitute_complex(H_rn, TEST_MAX_DEG, psi, clmo)
    H_back = substitute_real(H_cn, TEST_MAX_DEG, psi, clmo)

    for d in range(TEST_MAX_DEG+1):
        assert np.allclose(H_back[d], H_rn[d], atol=1e-14, rtol=1e-14), f"degree {d} mismatch"

        x,y,z,px,py,pz = sp.symbols('x y z px py pz')
        expr = poly2sympy(H_back, (x,y,z,px,py,pz), psi, clmo)
        all_terms = {m:c for m,c in sp.Poly(expr, x,y,z,px,py,pz).terms()}
        
        quad = {m:c for m,c in all_terms.items() if sum(m)==2 and abs(complex(c)) > 1e-12}

        lambda1, omega1, omega2 = libration_point.linear_modes
        expected = {
            (1,0,0,1,0,0):  lambda1,
            (0,2,0,0,0,0):  0.5*omega1,  (0,0,0,0,2,0): 0.5*omega1,
            (0,0,2,0,0,0):  0.5*omega2,  (0,0,0,0,0,2): 0.5*omega2,
        }
        
        assert set(quad.keys()) == set(expected.keys()), "Quadratic terms have different monomials"
        for k in expected:
            assert np.isclose(abs(complex(quad[k])), abs(complex(expected[k])), atol=1e-12, rtol=1e-12), f"Value mismatch for term {k}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_coordinate_realification_complexification(seed):

    np.random.seed(seed)

    test_real_coords = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),  # x-direction
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),  # y-direction
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),  # z-direction
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64),  # px-direction
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),  # py-direction
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # pz-direction
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64),  # mixed small
        np.random.uniform(-1.0, 1.0, 6).astype(np.float64),          # random small
        np.random.uniform(-5.0, 5.0, 6).astype(np.float64),          # random larger
    ]
    
    for i, real_coords in enumerate(test_real_coords):
        complex_coords = solve_real(real_coords)
        recovered_real_coords = solve_complex(complex_coords)
        
        np.testing.assert_allclose(
            recovered_real_coords, real_coords, 
            rtol=1e-14, atol=1e-14,
            err_msg=f"Real->Complex->Real round trip failed for test case {i}: "
                   f"input={real_coords}, recovered={recovered_real_coords}"
        )