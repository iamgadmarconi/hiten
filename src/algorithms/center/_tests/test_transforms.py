import numpy as np
import pytest
import sympy as sp
from numba import njit, types  # Added njit
from numba.typed import Dict, List  # Added Dict

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.polynomial.base import (ENCODE_DICT_GLOBAL,
                                               _create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.conversion import poly2sympy, sympy2poly
from algorithms.center.polynomial.operations import (
    polynomial_add_inplace, polynomial_multiply, polynomial_poisson_bracket,
    polynomial_power, polynomial_variable, polynomial_zero_list)
from algorithms.center.transforms import (_linear_variable_polys, realify,
                                          local2realmodal, complexify, realmodal2local,
                                          substitute_linear)
from system.libration import L1Point

_sympy_vars = sp.symbols("x y z px py pz")


@pytest.fixture(scope="module")
def point() -> L1Point:  # noqa: D401 – pytest fixture
    """Return an Earth-Moon L1 point (mu value taken from JPL DE-430)."""
    mu_earth_moon = 0.012150585609624
    return L1Point(mu=mu_earth_moon)

@pytest.fixture(scope="module", params=[4, 6])
def max_deg(request):
    return request.param

@pytest.fixture(scope="function")
def psi_clmo(max_deg):
    psi, clmo = init_index_tables(max_deg)
    encode_dict = _create_encode_dict_from_clmo(clmo) # Create encode_dict
    return psi, clmo, encode_dict # Return encode_dict

def sympy_reference(point: L1Point, max_deg: int) -> sp.Expr:
    """Exact Hamiltonian expanded with SymPy up to *max_deg* total degree."""
    x, y, z, px, py, pz = _sympy_vars
    vars_tuple = (x, y, z, px, py, pz)

    rho2 = x**2 + y**2 + z**2
    rho = sp.sqrt(rho2)

    H = sp.Rational(1, 2) * (px**2 + py**2 + pz**2) + y * px - x * py

    for n in range(2, max_deg + 1):
        cn = point._cn(n)
        Pn_expr = sp.legendre(n, x / rho)
        # Each term cn * rho**n * Pn_expr should be a polynomial.
        # Expanding each term individually might help SymPy simplify it correctly.
        term_to_add = sp.simplify(cn * rho**n * Pn_expr)
        H -= term_to_add
    
    # Expand the full Hamiltonian expression
    expanded_H = sp.simplify(H)

    # Convert to Poly object and back to expression to ensure canonical polynomial form
    # and to catch errors if expanded_H is not a polynomial in vars_tuple.
    try:
        poly_obj = sp.Poly(expanded_H, *vars_tuple)
        return poly_obj.as_expr()
    except sp.PolynomialError as e:
        # This means expanded_H is not recognized by SymPy as a polynomial in vars_tuple.
        # This would be unexpected if the underlying mathematical formulation is correct
        # and implies an issue with how SymPy is simplifying the expression.
        error_msg = (
            f"Failed to convert SymPy expression to polynomial form in sympy_reference.\n"
            f"Expression: {expanded_H}\n"
            f"Error: {e}"
        )
        raise type(e)(error_msg) from e






@pytest.mark.parametrize("max_deg", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("complex_dtype_bool", [False, True])
def test_linear_variable_polys(max_deg, complex_dtype_bool, psi_clmo):
    """Test the _linear_variable_polys function for correctness."""
    psi, clmo, encode_dict = psi_clmo # Unpack encode_dict

    # Define a base C matrix (float)
    C_base = np.array([
        [1., 2., 0., 0.5, 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [3., 0., 0., 0., 1., 0.2],
        [0., 0., 0., 0., 0., 1.],
    ], dtype=float)

    if complex_dtype_bool:
        C = C_base.astype(np.complex128)
        # Introduce some complex values for testing
        C[0, 1] += 1.5j
        C[4, 5] -= 0.75j
        C[1, 0] = 0.0 + 0.1j # Test a purely imaginary small number
    else:
        C = C_base.astype(float)

    # Call the function to be tested
    # _linear_variable_polys from transforms.py now expects encode_dict_list
    L_actual = _linear_variable_polys(C, max_deg, psi, clmo, encode_dict) # Pass encode_dict

    # Construct the expected result
    # polynomial_variable also needs encode_dict
    new_basis_expected = [polynomial_variable(j, max_deg, psi, clmo, encode_dict) for j in range(6)] # Pass encode_dict
    
    L_expected = []
    for i in range(6):
        pol_expected_i = polynomial_zero_list(max_deg, psi)
        for j in range(6):
            if C[i, j] != 0: 
                polynomial_add_inplace(pol_expected_i, new_basis_expected[j], C[i, j], max_deg)
        L_expected.append(pol_expected_i)

    # Assertions
    assert len(L_actual) == 6, f"Test({max_deg=}, {complex_dtype_bool=}): L_actual should have 6 polynomials"
    assert len(L_expected) == 6, f"Test({max_deg=}, {complex_dtype_bool=}): L_expected should have 6 polynomials"

    expected_np_dtype = np.complex128 if complex_dtype_bool else np.float64

    for i in range(6):
        poly_actual_i = L_actual[i]
        poly_expected_i = L_expected[i]

        assert len(poly_actual_i) == max_deg + 1, f"Test({max_deg=}, {complex_dtype_bool=}): Mismatch in num degree slices for L_actual[{i}]"
        assert len(poly_expected_i) == max_deg + 1, f"Test({max_deg=}, {complex_dtype_bool=}): Mismatch in num degree slices for L_expected[{i}]"

        for deg_idx in range(max_deg + 1):
            coeffs_actual = poly_actual_i[deg_idx]
            coeffs_expected = poly_expected_i[deg_idx]
            
            assert np.allclose(
                coeffs_actual, coeffs_expected, atol=1e-15, rtol=1e-12
            ), (f"Test({max_deg=}, {complex_dtype_bool=}): Mismatch for old_var {i}, degree {deg_idx}.\n"
                f"Actual: {coeffs_actual}\nExpected: {coeffs_expected}")

@pytest.mark.parametrize("max_deg_test", [0, 1, 2, 3, 5]) 
@pytest.mark.parametrize("complex_dtype_bool", [False, True])
def test_substitute_linear(max_deg_test, complex_dtype_bool, psi_clmo):
    """Test the substitute_linear function for correctness."""
    # psi_clmo fixture for this test should provide the encode_dict for this max_deg_test
    # However, substitute_linear takes max_deg, psi, clmo, encode_dict as direct args.
    # We need to generate them for max_deg_test here.
    psi_local, clmo_local = init_index_tables(max_deg_test)
    encode_dict_local = _create_encode_dict_from_clmo(clmo_local)
    dtype = np.complex128 if complex_dtype_bool else np.float64

    # Common helper to create a polynomial for a constant coeff
    def create_const_poly(val, max_deg_local, psi_local):
        p = polynomial_zero_list(max_deg_local, psi_local)
        p[0][0] = val
        return p

    # Test Case 0: H_old is a constant
    H_old0 = polynomial_zero_list(max_deg_test, psi_local)
    const_val = dtype(5.0 - (2.0j if complex_dtype_bool else 0.0))
    H_old0[0][0] = const_val
    
    C0 = np.array([[2.0, 1.0, 0,0,0,0],
                     [0.5, 1.0, 0,0,0,0],
                     [0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=dtype)

    H_actual0 = substitute_linear(H_old0, C0, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
    # Expected is just H_old0 itself, as constants are unaffected by variable substitution.
    for d_idx in range(max_deg_test + 1):
        assert np.allclose(H_actual0[d_idx], H_old0[d_idx], atol=1e-15, rtol=1e-12), \
            f"SubstLinear TC0 (const) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

    # Test Case 1: H_old = c0 * x_old_0 + c1 * x_old_1 (only if max_deg_test >= 1)
    if max_deg_test >= 1:
        H_old1 = polynomial_zero_list(max_deg_test, psi_local)
        c0_val = dtype(2.0 + (1.0j if complex_dtype_bool else 0.0))
        c1_val = dtype(3.0 - (0.5j if complex_dtype_bool else 0.0))

        k_x0 = tuple([1 if i == 0 else 0 for i in range(6)])
        # encode_multiindex here is from base, ensure it uses the local encode_dict_local
        idx_x0 = encode_multiindex(np.array(k_x0, dtype=np.int64), 1, encode_dict_local)
        H_old1[1][idx_x0] = c0_val

        k_x1 = tuple([1 if i == 1 else 0 for i in range(6)])
        idx_x1 = encode_multiindex(np.array(k_x1, dtype=np.int64), 1, encode_dict_local)
        H_old1[1][idx_x1] = c1_val
        
        C1 = np.identity(6, dtype=dtype)
        C1[0,1] = dtype(0.5 + (0.2j if complex_dtype_bool else 0.0)) # x_old_0 = 1*x_new_0 + (0.5+0.2j)*x_new_1
        C1[1,0] = dtype(0.3 - (0.1j if complex_dtype_bool else 0.0)) # x_old_1 = (0.3-0.1j)*x_new_0 + 1*x_new_1

        H_actual1 = substitute_linear(H_old1, C1, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        
        L1 = _linear_variable_polys(C1, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        
        const_poly_c0 = create_const_poly(c0_val, max_deg_test, psi_local)
        const_poly_c1 = create_const_poly(c1_val, max_deg_test, psi_local)

        term_for_c0_x_old_0 = polynomial_multiply(const_poly_c0, L1[0], max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        term_for_c1_x_old_1 = polynomial_multiply(const_poly_c1, L1[1], max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        
        H_expected1 = polynomial_zero_list(max_deg_test, psi_local)
        polynomial_add_inplace(H_expected1, term_for_c0_x_old_0, 1.0, max_deg_test)
        polynomial_add_inplace(H_expected1, term_for_c1_x_old_1, 1.0, max_deg_test)

        for d_idx in range(max_deg_test + 1):
            assert np.allclose(H_actual1[d_idx], H_expected1[d_idx], atol=1e-15, rtol=1e-12), \
                f"SubstLinear TC1 (linear) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

    # Test Case 2: H_old = c_sq * (x_old_0)^2 (only if max_deg_test >= 2)
    if max_deg_test >= 2:
        H_old2 = polynomial_zero_list(max_deg_test, psi_local)
        c_sq_val = dtype(1.5 + (0.5j if complex_dtype_bool else 0.0))
        
        k_x0sq = tuple([2 if i == 0 else 0 for i in range(6)])
        idx_x0sq = encode_multiindex(np.array(k_x0sq, dtype=np.int64), 2, encode_dict_local) # Use local encode_dict
        H_old2[2][idx_x0sq] = c_sq_val

        C2 = np.identity(6, dtype=dtype)
        C2[0,0] = dtype(1.2 - (0.3j if complex_dtype_bool else 0.0)) 
        C2[0,1] = dtype(0.7 + (0.4j if complex_dtype_bool else 0.0)) # x_old_0 = C2[0,0]*x_new_0 + C2[0,1]*x_new_1
        
        H_actual2 = substitute_linear(H_old2, C2, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        
        L2 = _linear_variable_polys(C2, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        const_poly_c_sq = create_const_poly(c_sq_val, max_deg_test, psi_local)
        
        powered_L0 = polynomial_power(L2[0], 2, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict
        H_expected2 = polynomial_multiply(const_poly_c_sq, powered_L0, max_deg_test, psi_local, clmo_local, encode_dict_local) # Pass local encode_dict

        for d_idx in range(max_deg_test + 1):
            assert np.allclose(H_actual2[d_idx], H_expected2[d_idx], atol=1e-14, rtol=1e-11), \
                f"SubstLinear TC2 (quad) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

def test_identity(max_deg, psi_clmo):
    psi, clmo, encode_dict = psi_clmo # Unpack encode_dict
    I = np.eye(6)

    # random polynomial with integer coefficients in [‑3, 3]
    rng = np.random.default_rng(0)
    coeffs = rng.integers(-3, 4, size=20)  # 20 random terms

    expr = 0
    for c in coeffs:
        exps = rng.integers(0, 3, size=6)
        if sum(exps) > max_deg:
            continue
        mon = 1
        for v, k in zip(_sympy_vars, exps):
            mon *= v**int(k)
        expr += int(c) * mon

    P = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict) # Pass _sympy_vars instead of max_deg

    # Ensure P has enough degree components (max_deg + 1 elements)
    while len(P) < max_deg + 1:
        P.append(polynomial_zero_list(len(P), psi)[0])  # Append zero coefficients for missing degrees

    P_sub = substitute_linear(P, I, max_deg, psi, clmo, encode_dict) # Pass encode_dict

    # Use allclose instead of array_equal to account for numerical precision
    assert all(
        np.allclose(a, b, atol=1e-14, rtol=1e-12) for a, b in zip(P, P_sub)
    ), "Identity substitution should return an identical polynomial (within numerical precision)."

def test_permutation(max_deg, psi_clmo):
    psi, clmo, encode_dict = psi_clmo # Unpack encode_dict
    # permutation matrix that swaps x and y (indices 0,1)
    Pmat = np.eye(6)
    Pmat[[0, 1]] = Pmat[[1, 0]]
    
    # The permutation matrix is symmetric (Pmat = Pmat.T), so transpose has no effect
    # Let's debug to see what's happening
    x, y, z, px, py, pz = _sympy_vars
    expr = x**2 + 2*y*px + 3*z**2
    
    # In the permutation matrix, we have:
    # - Pmat[0,1] = 1, meaning old_x = new_y
    # - Pmat[1,0] = 1, meaning old_y = new_x
    # So this matches the SymPy substitution [(x, y), (y, x)]
    
    P_old = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict) # Pass _sympy_vars instead of max_deg
    
    # Ensure P_old has enough degree components (max_deg + 1 elements)
    while len(P_old) < max_deg + 1:
        P_old.append(polynomial_zero_list(len(P_old), psi)[0])  # Append zero coefficients for missing degrees
        
    P_new = substitute_linear(P_old, Pmat, max_deg, psi, clmo, encode_dict) # Pass encode_dict
    
    # Expected result after substitution: x→y, y→x
    # x^2 → y^2, 2*y*px → 2*x*px, 3*z^2 → 3*z^2
    expected_expr = y**2 + 2*x*px + 3*z**2
    expr_test = poly2sympy(P_new, _sympy_vars, psi, clmo)

    diff = sp.expand(expected_expr - expr_test)
    
    assert diff == 0, f"Mismatch for permutation test with max_deg={max_deg}. Difference: {diff}"

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_matrix(seed, max_deg, psi_clmo):
    psi, clmo, encode_dict = psi_clmo # Unpack encode_dict
    rng = np.random.default_rng(seed)

    # Generate a random 6×6 integer matrix with entries in {‑2, ‑1, 0, 1, 2}
    C = rng.integers(-2, 3, size=(6, 6))
    while np.linalg.matrix_rank(C) < 6:
        # ensure it is invertible so the symbolic substitution is well‑defined
        C = rng.integers(-2, 3, size=(6, 6))
    
    # Transpose the matrix to match sympy's substitution direction
    # This is necessary because _linear_variable_polys uses C as old_var = C * new_var
    # while SymPy substitution uses the opposite direction
    C = C.T

    # random polynomial with ≤ max_deg and ≤ 15 terms
    coeffs = rng.integers(-5, 6, size=15)
    expr = 0
    for c in coeffs:
        exps = rng.integers(0, 3, size=6)
        if sum(exps) > max_deg:
            continue
        mon = 1
        for v, k in zip(_sympy_vars, exps):
            mon *= v**int(k)
        expr += int(c) * mon

    P_old = sympy2poly(expr, _sympy_vars, psi, clmo, encode_dict) # Pass _sympy_vars instead of max_deg
    
    # Ensure P_old has enough degree components (max_deg + 1 elements)
    while len(P_old) < max_deg + 1:
        P_old.append(polynomial_zero_list(len(P_old), psi)[0])  # Append zero coefficients for missing degrees
        
    P_new = substitute_linear(P_old, C, max_deg, psi, clmo, encode_dict) # Pass encode_dict

    # SymPy ground‑truth substitution
    x_old = np.array(_sympy_vars[:6])  # same symbols
    subs_dict = {x_old[i]: sum(int(C[i, j]) * x_old[j] for j in range(6)) for i in range(6)}
    expr_truth = expr.xreplace(subs_dict)
    expr_test = poly2sympy(P_new, _sympy_vars, psi, clmo)

    assert sp.expand(expr_truth - expr_test) == 0, f"Mismatch for seed {seed} and degree {max_deg}"

def test_symplectic(point):
    C, _ = point.normal_form_transform()
    J = np.block([[np.zeros((3, 3)),  np.eye(3)],
                  [-np.eye(3),        np.zeros((3, 3))]])
    assert np.allclose(C.T @ J @ C, J, atol=1e-13)

@pytest.mark.parametrize("max_deg", [2, 3, 4, 6])
def test_real_normal_form(point, max_deg):
    # Create fresh psi, clmo for each test instead of using the fixture
    psi, clmo = init_index_tables(max_deg)
    # Create encode_dict from clmo
    encode_dict = _create_encode_dict_from_clmo(clmo)
    
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn   = local2realmodal(point, H_phys, max_deg, psi, clmo)

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

    # (1) no cross-terms remain
    assert set(significant_quad_terms).issubset(allowed), (
        "Unexpected quadratic monomials after phys→rn transformation")

    # (2) coefficients equal the eigenvalues of the physical H2
    #     ( λ1 for x·px  , ω1**2/2  for y**2 & py**2 , etc. )
    # Get eigenvalues from the point's linear_modes method instead of calculating them
    lambda1, omega1, omega2 = point.linear_modes()
    
    # read coefficients back from the quadratic dict
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

@pytest.mark.parametrize("max_deg", [2, 3, 4, 6])
def test_complex_normal_form(point, max_deg):
    # Create fresh psi, clmo for each test instead of using the fixture
    psi, clmo = init_index_tables(max_deg)
    # Create encode_dict from clmo
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # 1) build physical Hamiltonian, go to real normal form, then complex
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn   = local2realmodal(point, H_phys, max_deg, psi, clmo)
    H_cn   = complexify(       H_rn,   max_deg, psi, clmo)

    # 2) symbolic expression of degree‑2 part
    q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
    expr = poly2sympy(H_cn, (q1, q2, q3, p1, p2, p3), psi, clmo)

    quad_terms = {
        m: c for m, c in sp.Poly(expr, q1, q2, q3, p1, p2, p3).terms() if sum(m) == 2
    }

    # discard numerical noise
    quad_terms = {
        m: complex(c.evalf()) for m, c in quad_terms.items() if abs(complex(c)) > 1e-12
    }

    # ---- allowed monomials & their expected coefficients ------------------
    allowed = {
        (1, 0, 0, 1, 0, 0): "q1p1",  # q1 * p1  ->  λ1      (real)
        (0, 1, 0, 0, 1, 0): "q2p2",  # q2 * p2  ->  i ω1    (imag)
        (0, 0, 1, 0, 0, 1): "q3p3",  # q3 * p3  ->  i ω2    (imag)
    }

    assert set(quad_terms).issubset(allowed), "Unexpected quadratic monomials after rn→cn"

    # ---- numerical values --------------------------------------------------
    lambda1, omega1, omega2 = point.linear_modes()

    coeff_q1p1 = quad_terms[(1, 0, 0, 1, 0, 0)]
    coeff_q2p2 = quad_terms[(0, 1, 0, 0, 1, 0)]
    coeff_q3p3 = quad_terms[(0, 0, 1, 0, 0, 1)]

    # real hyperbolic coefficient
    assert np.isclose(coeff_q1p1.real, lambda1, rtol=1e-12)
    assert abs(coeff_q1p1.imag) < 1e-12

    # imaginary elliptic coefficients (should be  i ω )
    assert np.isclose(coeff_q2p2 / 1j, omega1, rtol=1e-12)
    assert np.isclose(coeff_q3p3 / 1j, omega2, rtol=1e-12)
    # ---------------- Poisson‑bracket sanity tests -------------------------
    # Extract the degree-2 part of the Hamiltonian
    H2 = polynomial_zero_list(max_deg, psi)
    for d in range(len(H_cn)):
        if d == 2:  # Only copy degree 2 terms
            H2[d] = H_cn[d].copy()
    
    # Create |q2|² = q2 * p2 polynomial
    q2_var = polynomial_variable(1, max_deg, psi, clmo, encode_dict)
    p2_var = polynomial_variable(4, max_deg, psi, clmo, encode_dict)
    q2p2_poly = polynomial_multiply(q2_var, p2_var, max_deg, psi, clmo, encode_dict)
    
    # Create |q3|² = q3 * p3 polynomial
    q3_var = polynomial_variable(2, max_deg, psi, clmo, encode_dict)
    p3_var = polynomial_variable(5, max_deg, psi, clmo, encode_dict)
    q3p3_poly = polynomial_multiply(q3_var, p3_var, max_deg, psi, clmo, encode_dict)
    
    # Compute the Poisson brackets
    pb_H2_q2p2 = polynomial_poisson_bracket(H2, q2p2_poly, max_deg, psi, clmo, encode_dict)
    pb_H2_q3p3 = polynomial_poisson_bracket(H2, q3p3_poly, max_deg, psi, clmo, encode_dict)
    
    # Check that the Poisson brackets are zero (within numerical tolerance)
    for d in range(max_deg + 1):
        if pb_H2_q2p2[d].size > 0:
            assert np.allclose(pb_H2_q2p2[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q2|²}} should be zero, but degree {d} terms are not"
        if pb_H2_q3p3[d].size > 0:
            assert np.allclose(pb_H2_q3p3[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q3|²}} should be zero, but degree {d} terms are not"
    
    # Also test the bracket with hyperbolic action I1 = q1 * p1
    q1_var = polynomial_variable(0, max_deg, psi, clmo, encode_dict)
    p1_var = polynomial_variable(3, max_deg, psi, clmo, encode_dict)
    q1p1_poly = polynomial_multiply(q1_var, p1_var, max_deg, psi, clmo, encode_dict)
    
    pb_H2_q1p1 = polynomial_poisson_bracket(H2, q1p1_poly, max_deg, psi, clmo, encode_dict)
    
    for d in range(max_deg + 1):
        if pb_H2_q1p1[d].size > 0:
            assert np.allclose(pb_H2_q1p1[d], 0, atol=1e-12), \
                f"Poisson bracket {{{H2}, |q1|²}} should be zero, but degree {d} terms are not"

@pytest.mark.parametrize("max_deg", [2, 3, 4, 6])
def test_cn2rn_inverse(point, max_deg):
    # Create fresh psi, clmo for each test instead of using the fixture
    psi, clmo = init_index_tables(max_deg)
    # Create encode_dict from clmo
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # pipeline ---------------------------------------------------------------
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn   = local2realmodal(point, H_phys, max_deg, psi, clmo)
    H_cn   = complexify(       H_rn,   max_deg, psi, clmo)
    H_back = realify(       H_cn,   max_deg, psi, clmo)

    # (1) coefficient-by-coefficient equality with appropriate tolerance
    for d in range(max_deg+1):
        assert np.allclose(H_back[d], H_rn[d], atol=1e-14, rtol=1e-14), f"degree {d} mismatch"

        # (2) quadratic-block sanity (reuse earlier helper)
        x,y,z,px,py,pz = sp.symbols('x y z px py pz')
        expr = poly2sympy(H_back, (x,y,z,px,py,pz), psi, clmo)
        all_terms = {m:c for m,c in sp.Poly(expr, x,y,z,px,py,pz).terms()}
        
        # Filter out terms with very small coefficients (numerical artifacts)
        quad = {m:c for m,c in all_terms.items() if sum(m)==2 and abs(complex(c)) > 1e-12}

        lambda1, omega1, omega2 = point.linear_modes()
        expected = {
            (1,0,0,1,0,0):  lambda1,
            (0,2,0,0,0,0):  0.5*omega1,  (0,0,0,0,2,0): 0.5*omega1,
            (0,0,2,0,0,0):  0.5*omega2,  (0,0,0,0,0,2): 0.5*omega2,
        }
        
        # Compare with approximate equality
        assert set(quad.keys()) == set(expected.keys()), "Quadratic terms have different monomials"
        for k in expected:
            assert np.isclose(abs(complex(quad[k])), abs(complex(expected[k])), atol=1e-12, rtol=1e-12), f"Value mismatch for term {k}"

@pytest.mark.parametrize("max_deg", [2, 3, 4, 6])
def test_phys2rn_rn2phys_roundtrip(point, max_deg):
    """Test that realmodal2local is the inverse of local2realmodal: realmodal2local(local2realmodal(H_phys)) ≈ H_phys."""
    # Create fresh psi, clmo for each test instead of using the fixture
    psi, clmo = init_index_tables(max_deg)
    # Create encode_dict from clmo
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # Build the physical Hamiltonian
    H_phys_original = build_physical_hamiltonian(point, max_deg)
    
    # Forward transformation: physical → real normal form
    H_rn = local2realmodal(point, H_phys_original, max_deg, psi, clmo)
    
    # Backward transformation: real normal form → physical
    H_phys_roundtrip = realmodal2local(point, H_rn, max_deg, psi, clmo)

    # Verify roundtrip: coefficient-by-coefficient equality with appropriate tolerance
    for d in range(max_deg + 1):
        assert np.allclose(
            H_phys_roundtrip[d], H_phys_original[d], 
            atol=1e-13, rtol=1e-13
        ), f"Roundtrip failed for degree {d} terms: max relative error = {np.max(np.abs((H_phys_roundtrip[d] - H_phys_original[d]) / (H_phys_original[d] + 1e-16)))}"

    # Additional verification: check that the symbolic expressions match
    x, y, z, px, py, pz = sp.symbols('x y z px py pz')
    vars_tuple = (x, y, z, px, py, pz)
    
    expr_original = poly2sympy(H_phys_original, vars_tuple, psi, clmo)
    expr_roundtrip = poly2sympy(H_phys_roundtrip, vars_tuple, psi, clmo)
    
    # Compute the difference and verify it's essentially zero
    diff = sp.expand(expr_original - expr_roundtrip)
    
    # For symbolic verification, we'll check that all coefficients are below tolerance
    if isinstance(diff, (int, float, complex)):
        # If diff is a scalar, check it directly
        assert abs(complex(diff)) < 1e-12, f"Symbolic difference is not zero: {diff}"
    else:
        # If diff is a polynomial expression, check all coefficients
        try:
            poly_diff = sp.Poly(diff, *vars_tuple)
            max_coeff = max(abs(complex(c)) for c in poly_diff.all_coeffs()) if poly_diff.all_coeffs() else 0
            assert max_coeff < 1e-12, f"Maximum coefficient in symbolic difference: {max_coeff}"
        except sp.PolynomialError:
            # If we can't convert to polynomial, evaluate at several points
            test_points = [
                {x: 0.1, y: 0.2, z: 0.1, px: 0.15, py: 0.1, pz: 0.05},
                {x: -0.1, y: 0.1, z: -0.05, px: -0.1, py: 0.2, pz: -0.1},
                {x: 0.05, y: -0.15, z: 0.2, px: 0.1, py: -0.05, pz: 0.15}
            ]
            for point_vals in test_points:
                diff_val = complex(diff.subs(point_vals))
                assert abs(diff_val) < 1e-12, f"Symbolic difference at {point_vals}: {diff_val}"

@pytest.mark.parametrize("max_deg", [2, 3, 4, 6])
def test_full_roundtrip(point, max_deg):
    """Test the complete transformation pipeline: Phys → RN → CN → RN → Phys."""
    # Create fresh psi, clmo for each test instead of using the fixture
    psi, clmo = init_index_tables(max_deg)
    # Create encode_dict from clmo
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # Build the original physical Hamiltonian
    H_phys_original = build_physical_hamiltonian(point, max_deg)
    
    # Forward pipeline: Phys → RN → CN
    H_rn_forward = local2realmodal(point, H_phys_original, max_deg, psi, clmo)
    H_cn = complexify(H_rn_forward, max_deg, psi, clmo)
    
    # Backward pipeline: CN → RN → Phys
    H_rn_backward = realify(H_cn, max_deg, psi, clmo)
    H_phys_final = realmodal2local(point, H_rn_backward, max_deg, psi, clmo)

    # Verify full roundtrip: coefficient-by-coefficient equality
    for d in range(max_deg + 1):
        assert np.allclose(
            H_phys_final[d], H_phys_original[d], 
            atol=1e-12, rtol=1e-12
        ), f"Full roundtrip failed for degree {d} terms: max absolute error = {np.max(np.abs(H_phys_final[d] - H_phys_original[d]))}"

    # Verify intermediate RN forms match (RN forward vs RN backward)
    for d in range(max_deg + 1):
        assert np.allclose(
            H_rn_backward[d], H_rn_forward[d], 
            atol=1e-12, rtol=1e-12
        ), f"RN roundtrip (via CN) failed for degree {d} terms: max absolute error = {np.max(np.abs(H_rn_backward[d] - H_rn_forward[d]))}"

    # Additional symbolic verification for low-degree terms
    x, y, z, px, py, pz = sp.symbols('x y z px py pz')
    vars_tuple = (x, y, z, px, py, pz)
    
    expr_original = poly2sympy(H_phys_original, vars_tuple, psi, clmo)
    expr_final = poly2sympy(H_phys_final, vars_tuple, psi, clmo)
    
    # Compute the difference and verify it's essentially zero
    diff = sp.expand(expr_original - expr_final)
    
    # Check symbolic difference
    if isinstance(diff, (int, float, complex)):
        assert abs(complex(diff)) < 1e-11, f"Symbolic difference is not zero: {diff}"
    else:
        try:
            poly_diff = sp.Poly(diff, *vars_tuple)
            if poly_diff.all_coeffs():
                max_coeff = max(abs(complex(c)) for c in poly_diff.all_coeffs())
                assert max_coeff < 1e-11, f"Maximum coefficient in symbolic difference: {max_coeff}"
        except sp.PolynomialError:
            # If we can't convert to polynomial, evaluate at test points
            test_points = [
                {x: 0.1, y: 0.2, z: 0.1, px: 0.15, py: 0.1, pz: 0.05},
                {x: -0.1, y: 0.1, z: -0.05, px: -0.1, py: 0.2, pz: -0.1},
                {x: 0.05, y: -0.15, z: 0.2, px: 0.1, py: -0.05, pz: 0.15}
            ]
            for point_vals in test_points:
                diff_val = complex(diff.subs(point_vals))
                assert abs(diff_val) < 1e-11, f"Full roundtrip symbolic difference at {point_vals}: {diff_val}"

    # Test that the quadratic structure is preserved through the full pipeline
    # Extract quadratic terms from both original and final Hamiltonians
    poly_original = sp.Poly(expr_original, *vars_tuple)
    poly_final = sp.Poly(expr_final, *vars_tuple)
    
    quad_original = {m: c for m, c in poly_original.terms() if sum(m) == 2}
    quad_final = {m: c for m, c in poly_final.terms() if sum(m) == 2}
    
    # Filter out numerical noise
    quad_original = {m: c for m, c in quad_original.items() if abs(complex(c)) > 1e-12}
    quad_final = {m: c for m, c in quad_final.items() if abs(complex(c)) > 1e-12}
    
    # Verify that the quadratic terms match
    assert set(quad_original.keys()) == set(quad_final.keys()), \
        f"Quadratic term structure changed: original has {set(quad_original.keys())}, final has {set(quad_final.keys())}"
    
    for monomial in quad_original:
        coeff_diff = abs(complex(quad_original[monomial]) - complex(quad_final[monomial]))
        assert coeff_diff < 1e-11, \
            f"Quadratic coefficient mismatch for {monomial}: |{quad_original[monomial]} - {quad_final[monomial]}| = {coeff_diff}"
