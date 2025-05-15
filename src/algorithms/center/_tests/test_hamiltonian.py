from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from algorithms.center.polynomial import base, operations as op, algebra
from algorithms.center.hamiltonian import (
    build_physical_hamiltonian, 
    _build_T_polynomials, 
    _linear_variable_polys,
    substitute_linear,
    phys2rn, rn2cn, cn2rn
)
from system.libration import L1Point  # noqa: F401 – used in fixture

_sympy_vars = sp.symbols("x y z px py pz")

@pytest.fixture(scope="module")
def point() -> L1Point:  # noqa: D401 – pytest fixture
    """Return an Earth-Moon L1 point (mu value taken from JPL DE-430)."""
    mu_earth_moon = 0.012150585609624
    return L1Point(mu=mu_earth_moon)

@pytest.fixture(scope="module", params=[4, 6])
def max_deg(request):
    return request.param

@pytest.fixture(scope="module")
def psi_clmo(max_deg):
    psi, clmo = base.init_index_tables(max_deg)
    return psi, clmo

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

def sympy_to_poly(sym_expr: sp.Expr, max_deg: int, psi, clmo):
    """Convert a SymPy polynomial into the coefficient-list format used by the codebase."""

    poly = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)

    for term in sym_expr.as_ordered_terms():
        coeff, monom = term.as_coeff_Mul()
        if monom == 1:
            exps = (0, 0, 0, 0, 0, 0)
        else:
            exps = sp.Poly(monom, *_sympy_vars).monoms()[0]

        deg = sum(exps)
        if deg > max_deg:
            continue  # truncated away in our polynomial representation

        idx = algebra.encode_multiindex(exps, deg, psi, clmo) # Pass full psi and clmo tables
        poly[deg][idx] = float(coeff)

    return poly

def poly_list_to_sympy(P, vars, psi, clmo):
    """
    Convert coefficient-list *P* to a SymPy expression.

    Parameters
    ----------
    P : List[np.ndarray]
        one coefficient vector per total degree
    vars : Sequence[sympy.Symbol]
        the symbol that corresponds to each of the six packed exponents
    psi, clmo : lookup tables
    """
    expr = 0
    for deg, coeff_vec in enumerate(P):
        for pos, coeff in enumerate(coeff_vec):
            if coeff == 0:
                continue
            exps = algebra.decode_multiindex(pos, deg, clmo)  # 6-tuple
            mon  = sp.prod(v**k for v, k in zip(vars, exps) if k)
            expr += coeff * mon
    return sp.expand(expr)


def evaluate_poly(poly, values, psi, clmo):
    """Brute-force evaluation of a coefficient-list polynomial at *values* (ℝ⁶)."""

    total = 0.0
    for deg, coeffs in enumerate(poly):
        if coeffs.size == 0:
            continue
        for pos, c in enumerate(coeffs):
            if c == 0.0:
                continue
            exps = algebra.decode_multiindex(pos, deg, clmo) # Pass full clmo table
            total += c * np.prod(values**np.asarray(exps))
    return total


@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_symbolic_identity(point, max_deg):
    """Coefficient arrays must match a SymPy ground-truth for small degrees."""

    psi, clmo = base.init_index_tables(max_deg)
    bph_psi_config = (None, False)
    bph_clmo_deg = max_deg
    H_build = build_physical_hamiltonian(point, max_deg, bph_psi_config, bph_clmo_deg)

    H_sympy = sympy_reference(point, max_deg)
    H_ref = sympy_to_poly(H_sympy, max_deg, psi, clmo)

    for d in range(max_deg + 1):
        assert np.allclose(
            H_build[d], H_ref[d], atol=1e-12, rtol=1e-9
        ), f"Mismatch found in degree slice {d}.\nBuild: {H_build[d]}\nRef:   {H_ref[d]}"

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_legendre_recursion(point, max_deg):
    """Internal `T[n]` sequence must satisfy Legendre three-term recursion."""

    psi, clmo = base.init_index_tables(max_deg)
    x_poly, y_poly, z_poly, *_ = [
        op.polynomial_variable(i, max_deg, psi, complex_dtype=False) for i in range(6)
    ]

    # Call the refactored helper function directly
    T = _build_T_polynomials(x_poly, y_poly, z_poly, max_deg, psi, clmo, complex_dt=False)

    # The rest of the test logic for verification remains largely the same,
    # but it uses the T computed above.
    # We still need sum_sq for the RHS of the recursion check.
    sum_sq = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
    for var in (x_poly, y_poly, z_poly): # Use polynomial variables
        op.polynomial_add_inplace(sum_sq, op.polynomial_multiply(var, var, max_deg, psi, clmo), 1.0)
    
    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        # LHS is T[n] from the function call
        lhs = T[n]

        # RHS is constructed using the recursion formula
        # (a * x * T[n-1] - b * sum_sq * T[n-2])
        
        term1_mult = op.polynomial_multiply(x_poly, T[n - 1], max_deg, psi, clmo)
        term1 = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(term1, term1_mult, a)
        
        term2_mult = op.polynomial_multiply(sum_sq, T[n - 2], max_deg, psi, clmo)
        term2 = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(term2, term2_mult, -b) # -b factor

        rhs = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(rhs, term1, 1.0)
        op.polynomial_add_inplace(rhs, term2, 1.0)

        for d in range(max_deg + 1):
            assert np.array_equal(lhs[d], rhs[d]), f"Legendre recursion failed at n={n}, degree slice d={d}"

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_numerical_evaluation(point, max_deg):
    """Evaluate both Hamiltonians at random points and compare numerically."""

    psi, clmo = base.init_index_tables(max_deg)
    H_poly = build_physical_hamiltonian(point, max_deg, (psi, np.float64), clmo) # Pass psi as first element of tuple for psi_config
    H_sym = sympy_reference(point, max_deg)

    rng = np.random.default_rng(42)
    vars_syms = sp.symbols("x y z px py pz")

    for _ in range(50):
        vals = rng.uniform(-0.1, 0.1, 6)
        H_num_poly = evaluate_poly(H_poly, vals, psi, clmo)
        H_num_sym = float(H_sym.subs(dict(zip(vars_syms, vals))))
        assert np.isclose(
            H_num_poly, H_num_sym, atol=1e-12
        ), "Numerical mismatch between polynomial and SymPy Hamiltonians"

@pytest.mark.parametrize("max_deg", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("complex_dtype_bool", [False, True])
def test_linear_variable_polys(max_deg, complex_dtype_bool):
    """Test the _linear_variable_polys function for correctness."""
    psi, clmo = base.init_index_tables(max_deg)

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
    L_actual = _linear_variable_polys(C, max_deg, psi, clmo, complex_dtype_bool)

    # Construct the expected result
    new_basis_expected = [op.polynomial_variable(j, max_deg, psi, complex_dtype_bool) for j in range(6)]
    
    L_expected = []
    for i in range(6):
        pol_expected_i = op.polynomial_zero_list(max_deg, psi, complex_dtype_bool)
        for j in range(6):
            if C[i, j] != 0: 
                op.polynomial_add_inplace(pol_expected_i, new_basis_expected[j], C[i, j], max_deg)
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
            
            assert coeffs_actual.dtype == expected_np_dtype, \
                f"Test({max_deg=}, {complex_dtype_bool=}): L_actual[{i}][{deg_idx}] dtype mismatch. Expected {expected_np_dtype}, got {coeffs_actual.dtype}"
            assert coeffs_expected.dtype == expected_np_dtype, \
                f"Test({max_deg=}, {complex_dtype_bool=}): L_expected[{i}][{deg_idx}] dtype mismatch. Expected {expected_np_dtype}, got {coeffs_expected.dtype}"

            assert np.allclose(
                coeffs_actual, coeffs_expected, atol=1e-15, rtol=1e-12
            ), (f"Test({max_deg=}, {complex_dtype_bool=}): Mismatch for old_var {i}, degree {deg_idx}.\n"
                f"Actual: {coeffs_actual}\nExpected: {coeffs_expected}")

@pytest.mark.parametrize("max_deg_test", [0, 1, 2, 3, 5]) 
@pytest.mark.parametrize("complex_dtype_bool", [False, True])
def test_substitute_linear(max_deg_test, complex_dtype_bool):
    """Test the substitute_linear function for correctness."""
    psi, clmo = base.init_index_tables(max_deg_test)
    dtype = np.complex128 if complex_dtype_bool else np.float64

    # Common helper to create a polynomial for a constant coeff
    def create_const_poly(val, max_deg_local, psi_local, complex_local):
        p = op.polynomial_zero_list(max_deg_local, psi_local, complex_local)
        p[0][0] = val
        return p

    # Test Case 0: H_old is a constant
    H_old0 = op.polynomial_zero_list(max_deg_test, psi, complex_dtype_bool)
    const_val = dtype(5.0 - (2.0j if complex_dtype_bool else 0.0))
    H_old0[0][0] = const_val
    
    C0 = np.array([[2.0, 1.0, 0,0,0,0],
                     [0.5, 1.0, 0,0,0,0],
                     [0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=dtype)

    H_actual0 = substitute_linear(H_old0, C0, max_deg_test, psi, clmo, complex_dtype_bool)
    # Expected is just H_old0 itself, as constants are unaffected by variable substitution.
    for d_idx in range(max_deg_test + 1):
        assert np.allclose(H_actual0[d_idx], H_old0[d_idx], atol=1e-15, rtol=1e-12), \
            f"SubstLinear TC0 (const) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

    # Test Case 1: H_old = c0 * x_old_0 + c1 * x_old_1 (only if max_deg_test >= 1)
    if max_deg_test >= 1:
        H_old1 = op.polynomial_zero_list(max_deg_test, psi, complex_dtype_bool)
        c0_val = dtype(2.0 + (1.0j if complex_dtype_bool else 0.0))
        c1_val = dtype(3.0 - (0.5j if complex_dtype_bool else 0.0))

        k_x0 = tuple([1 if i == 0 else 0 for i in range(6)])
        idx_x0 = algebra.encode_multiindex(k_x0, 1, psi, clmo)
        H_old1[1][idx_x0] = c0_val

        k_x1 = tuple([1 if i == 1 else 0 for i in range(6)])
        idx_x1 = algebra.encode_multiindex(k_x1, 1, psi, clmo)
        H_old1[1][idx_x1] = c1_val
        
        C1 = np.identity(6, dtype=dtype)
        C1[0,1] = dtype(0.5 + (0.2j if complex_dtype_bool else 0.0)) # x_old_0 = 1*x_new_0 + (0.5+0.2j)*x_new_1
        C1[1,0] = dtype(0.3 - (0.1j if complex_dtype_bool else 0.0)) # x_old_1 = (0.3-0.1j)*x_new_0 + 1*x_new_1

        H_actual1 = substitute_linear(H_old1, C1, max_deg_test, psi, clmo, complex_dtype_bool)
        
        L1 = _linear_variable_polys(C1, max_deg_test, psi, clmo, complex_dtype_bool)
        
        const_poly_c0 = create_const_poly(c0_val, max_deg_test, psi, complex_dtype_bool)
        const_poly_c1 = create_const_poly(c1_val, max_deg_test, psi, complex_dtype_bool)

        term_for_c0_x_old_0 = op.polynomial_multiply(const_poly_c0, L1[0], max_deg_test, psi, clmo)
        term_for_c1_x_old_1 = op.polynomial_multiply(const_poly_c1, L1[1], max_deg_test, psi, clmo)
        
        H_expected1 = op.polynomial_zero_list(max_deg_test, psi, complex_dtype_bool)
        op.polynomial_add_inplace(H_expected1, term_for_c0_x_old_0, 1.0, max_deg_test)
        op.polynomial_add_inplace(H_expected1, term_for_c1_x_old_1, 1.0, max_deg_test)

        for d_idx in range(max_deg_test + 1):
            assert np.allclose(H_actual1[d_idx], H_expected1[d_idx], atol=1e-15, rtol=1e-12), \
                f"SubstLinear TC1 (linear) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

    # Test Case 2: H_old = c_sq * (x_old_0)^2 (only if max_deg_test >= 2)
    if max_deg_test >= 2:
        H_old2 = op.polynomial_zero_list(max_deg_test, psi, complex_dtype_bool)
        c_sq_val = dtype(1.5 + (0.5j if complex_dtype_bool else 0.0))
        
        k_x0sq = tuple([2 if i == 0 else 0 for i in range(6)])
        idx_x0sq = algebra.encode_multiindex(k_x0sq, 2, psi, clmo)
        H_old2[2][idx_x0sq] = c_sq_val

        C2 = np.identity(6, dtype=dtype)
        C2[0,0] = dtype(1.2 - (0.3j if complex_dtype_bool else 0.0)) 
        C2[0,1] = dtype(0.7 + (0.4j if complex_dtype_bool else 0.0)) # x_old_0 = C2[0,0]*x_new_0 + C2[0,1]*x_new_1
        
        H_actual2 = substitute_linear(H_old2, C2, max_deg_test, psi, clmo, complex_dtype_bool)
        
        L2 = _linear_variable_polys(C2, max_deg_test, psi, clmo, complex_dtype_bool)
        const_poly_c_sq = create_const_poly(c_sq_val, max_deg_test, psi, complex_dtype_bool)
        
        powered_L0 = op.polynomial_power(L2[0], 2, max_deg_test, psi, clmo)
        H_expected2 = op.polynomial_multiply(const_poly_c_sq, powered_L0, max_deg_test, psi, clmo)

        for d_idx in range(max_deg_test + 1):
            assert np.allclose(H_actual2[d_idx], H_expected2[d_idx], atol=1e-14, rtol=1e-11), \
                f"SubstLinear TC2 (quad) failed: max_deg={max_deg_test}, complex={complex_dtype_bool}, d_idx={d_idx}"

def test_identity(max_deg, psi_clmo):
    psi, clmo = psi_clmo
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

    P = sympy_to_poly(expr, max_deg, psi, clmo)
    P_sub = substitute_linear(P, I, max_deg, psi, clmo)

    assert all(
        np.array_equal(a, b) for a, b in zip(P, P_sub)
    ), "Identity substitution should return an identical polynomial."

def test_permutation(max_deg, psi_clmo):
    psi, clmo = psi_clmo
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
    
    P_old = sympy_to_poly(expr, max_deg, psi, clmo)
    P_new = substitute_linear(P_old, Pmat, max_deg, psi, clmo)
    
    # Expected result after substitution: x→y, y→x
    # x^2 → y^2, 2*y*px → 2*x*px, 3*z^2 → 3*z^2
    expected_expr = y**2 + 2*x*px + 3*z**2
    expr_test = poly_list_to_sympy(P_new, _sympy_vars, psi, clmo)

    diff = sp.expand(expected_expr - expr_test)
    
    assert diff == 0, f"Mismatch for permutation test with max_deg={max_deg}. Difference: {diff}"

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_matrix(seed, max_deg, psi_clmo):
    psi, clmo = psi_clmo
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

    P_old = sympy_to_poly(expr, max_deg, psi, clmo)
    P_new = substitute_linear(P_old, C, max_deg, psi, clmo)

    # SymPy ground‑truth substitution
    x_old = np.array(_sympy_vars[:6])  # same symbols
    subs_dict = {x_old[i]: sum(int(C[i, j]) * x_old[j] for j in range(6)) for i in range(6)}
    expr_truth = expr.xreplace(subs_dict)
    expr_test = poly_list_to_sympy(P_new, _sympy_vars, psi, clmo)

    assert sp.expand(expr_truth - expr_test) == 0, f"Mismatch for seed {seed} and degree {max_deg}"

def test_symplectic(point):
    C, _ = point.normal_form_transform()
    J = np.block([[np.zeros((3, 3)),  np.eye(3)],
                  [-np.eye(3),        np.zeros((3, 3))]])
    assert np.allclose(C.T @ J @ C, J, atol=1e-13)

def test_real_normal_form(point, max_deg, psi_clmo):
    psi, clmo = psi_clmo
    H_phys = build_physical_hamiltonian(point, max_deg, (psi, np.float64), clmo)
    H_rn   = phys2rn(point, H_phys, max_deg, psi, clmo)

    x, y, z, px, py, pz = sp.symbols('x y z px py pz')
    expr = poly_list_to_sympy(H_rn, (x, y, z, px, py, pz), psi, clmo)
    
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

def test_complex_normal_form(point, max_deg, psi_clmo):
    psi, clmo = psi_clmo

    # 1) build physical Hamiltonian, go to real normal form, then complex
    H_phys = build_physical_hamiltonian(point, max_deg, (psi, np.float64), clmo)
    H_rn   = phys2rn(point, H_phys, max_deg, psi, clmo)
    H_cn   = rn2cn(       H_rn,   max_deg, psi, clmo)

    # 2) symbolic expression of degree‑2 part
    q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
    expr = poly_list_to_sympy(H_cn, (q1, q2, q3, p1, p2, p3), psi, clmo)

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
    # H2 = op.polynomial_zero_list(max_deg, psi, complex_dtype=True)
    # for d in range(len(H_cn)):
    #     if d == 2:  # Only copy degree 2 terms
    #         H2[d] = H_cn[d].copy()
    
    # # Create |q2|² = q2 * p2 polynomial
    # q2_var = op.polynomial_variable(1, max_deg, psi, complex_dtype=True)
    # p2_var = op.polynomial_variable(4, max_deg, psi, complex_dtype=True)
    # q2p2_poly = op.polynomial_multiply(q2_var, p2_var, max_deg, psi, clmo)
    
    # # Create |q3|² = q3 * p3 polynomial
    # q3_var = op.polynomial_variable(2, max_deg, psi, complex_dtype=True)
    # p3_var = op.polynomial_variable(5, max_deg, psi, complex_dtype=True)
    # q3p3_poly = op.polynomial_multiply(q3_var, p3_var, max_deg, psi, clmo)
    
    # # Compute the Poisson brackets
    # pb_H2_q2p2 = op.polynomial_poisson_bracket(H2, q2p2_poly, max_deg, psi, clmo)
    # pb_H2_q3p3 = op.polynomial_poisson_bracket(H2, q3p3_poly, max_deg, psi, clmo)
    
    # # Check that the Poisson brackets are zero (within numerical tolerance)
    # for d in range(max_deg + 1):
    #     if pb_H2_q2p2[d].size > 0:
    #         assert np.allclose(pb_H2_q2p2[d], 0, atol=1e-12), \
    #             f"Poisson bracket {{{H2}, |q2|²}} should be zero, but degree {d} terms are not"
    #     if pb_H2_q3p3[d].size > 0:
    #         assert np.allclose(pb_H2_q3p3[d], 0, atol=1e-12), \
    #             f"Poisson bracket {{{H2}, |q3|²}} should be zero, but degree {d} terms are not"
    
    # # Also test the bracket with hyperbolic action I1 = q1 * p1
    # q1_var = op.polynomial_variable(0, max_deg, psi, complex_dtype=True)
    # p1_var = op.polynomial_variable(3, max_deg, psi, complex_dtype=True)
    # q1p1_poly = op.polynomial_multiply(q1_var, p1_var, max_deg, psi, clmo)
    
    # pb_H2_q1p1 = op.polynomial_poisson_bracket(H2, q1p1_poly, max_deg, psi, clmo)
    
    # for d in range(max_deg + 1):
    #     if pb_H2_q1p1[d].size > 0:
    #         assert np.allclose(pb_H2_q1p1[d], 0, atol=1e-12), \
    #             f"Poisson bracket {{{H2}, |q1|²}} should be zero, but degree {d} terms are not"

