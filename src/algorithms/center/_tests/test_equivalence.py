import pytest
import numpy as np
import symengine as se
import sympy as sp

from system.libration import L1Point
from algorithms.variables import (
    get_vars, physical_vars, real_normal_vars, canonical_normal_vars, 
    linear_modes_vars, scale_factors_vars
)

# Old implementation imports
from algorithms.center._deprecated.dep_hamiltonian import (
    _build_T_polynomials as dep_build_T_polynomials,
    hamiltonian as dep_hamiltonian,
    physical_to_real_normal as dep_physical_to_real_normal,
    real_normal_to_complex_canonical as dep_real_normal_to_complex_canonical,
    complex_canonical_to_real_normal as dep_complex_canonical_to_real_normal
)
from algorithms.center._deprecated.dep_core import Polynomial, _clean_numerical_artifacts

# New implementation imports
from algorithms.center.hamiltonian import (
    _build_T_polynomials as new_build_T_polynomials,
    hamiltonian_arrays,
    physical_to_real_normal_arrays,
    real_normal_to_complex_arrays,
    complex_to_real_arrays,
    initialise_tables
)
from algorithms.center.polynomial.conversions import poly2symengine

# Get symbols for variables
x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)
s1, s2 = get_vars(scale_factors_vars)


@pytest.fixture()
def lp():
    """Create a libration point for testing."""
    mu = 0.0121505856  # Earth-Moon
    return L1Point(mu)


@pytest.fixture()
def max_degree():
    """Maximum polynomial degree for tests."""
    return 4


@pytest.fixture()
def psi_clmo(max_degree):
    """Initialize index tables for the new implementation."""
    return initialise_tables(max_degree)


def coefficients_match(sym_expr, array_expr, variables, rtol=1e-10, atol=1e-12):
    """
    Compare symbolic expression with array-based expression for equality.
    
    Parameters:
    sym_expr: SymEngine expression from deprecated implementation
    array_expr: SymEngine expression converted from array implementation
    variables: List of variables to compare against
    rtol, atol: Relative and absolute tolerance for floating point comparison
    
    Returns:
    bool: True if the expressions are equivalent within tolerance
    """
    # Convert both to SymPy for more robust comparison
    sym_sp = sp.sympify(str(sym_expr))
    array_sp = sp.sympify(str(array_expr))
    
    # Convert SymEngine variables to SymPy symbols
    sp_vars = [sp.Symbol(str(v)) for v in variables]
    
    # Check equality using SymPy's simplify
    diff = sp.expand(sym_sp - array_sp)
    
    # If the difference is exactly zero, they match perfectly
    if diff == 0:
        return True
        
    # For approximate equality, check individual coefficients
    for term in diff.as_ordered_terms():
        coeff, _ = term.as_coeff_add(*sp_vars)
        if not np.isclose(float(abs(coeff)), 0, rtol=rtol, atol=atol):
            print(f"Mismatch in term: {term}, coefficient magnitude: {float(abs(coeff))}")
            return False
            
    return True


def test_T_polynomials_equivalence(max_degree):
    """Test that both T polynomial implementations produce equivalent results."""
    # Generate polynomials using both implementations
    dep_T = dep_build_T_polynomials(max_degree)
    new_T = new_build_T_polynomials(max_degree)
    
    # Compare each polynomial term by term
    for n in range(max_degree + 1):
        dep_expr = dep_T[n]
        new_expr = new_T[n]
        
        # Clean up any numerical artifacts in old implementation
        dep_expr_sp = sp.sympify(dep_expr)
        dep_expr_clean = _clean_numerical_artifacts(dep_expr_sp)
        
        # Convert new expression to SymPy for comparison
        new_expr_sp = sp.sympify(new_expr)
        
        # Compare expressions mathematically using sympy's simplify
        diff = sp.simplify(dep_expr_clean - new_expr_sp)
        assert diff == 0, f"T_{n} polynomials don't match: {dep_expr_clean} vs {new_expr_sp}"


def test_hamiltonian_equivalence(lp, max_degree, psi_clmo):
    """Test that both Hamiltonian implementations produce equivalent results."""
    psi, clmo = psi_clmo
    
    # Generate Hamiltonians using both implementations
    H_dep = dep_hamiltonian(lp, max_degree)
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    
    # Convert array implementation to symbolic for comparison
    H_new_sym = poly2symengine(H_arrays, [x, y, z, px, py, pz], psi, clmo)
    
    # Convert both to SymPy and clean up numerical artifacts
    H_dep_sp = sp.sympify(H_dep.expression)
    H_dep_clean = _clean_numerical_artifacts(H_dep_sp)
    H_new_sp = sp.sympify(H_new_sym)
    H_new_clean = _clean_numerical_artifacts(H_new_sp)
    
    # Variables to compare against
    phys_vars = [x, y, z, px, py, pz]
    
    # Use direct simplify for comparison (more robust)
    diff = sp.expand(H_dep_clean - H_new_clean)
    
    # Check that difference is approximately zero
    assert coefficients_match(H_dep_clean, H_new_clean, phys_vars), \
        f"Hamiltonians don't match: {diff}"


def test_physical_to_real_normal_equivalence(lp, max_degree, psi_clmo):
    """Test that both physical to real normal transformations produce equivalent results."""
    psi, clmo = psi_clmo
    
    # Generate Hamiltonians
    H_dep = dep_hamiltonian(lp, max_degree)
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    
    # Transform to real normal form using both implementations
    H_rn_dep = dep_physical_to_real_normal(lp, H_dep, symbolic=False, max_degree=max_degree)
    H_rn_arrays = physical_to_real_normal_arrays(lp, H_arrays, max_degree, psi, clmo)
    
    # Convert array implementation to symbolic for comparison
    H_rn_new_sym = poly2symengine(H_rn_arrays, [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], psi, clmo)
    
    # Clean up numerical artifacts
    H_rn_dep_sp = sp.sympify(H_rn_dep.expression)
    H_rn_dep_clean = _clean_numerical_artifacts(H_rn_dep_sp)
    H_rn_new_sp = sp.sympify(H_rn_new_sym)
    H_rn_new_clean = _clean_numerical_artifacts(H_rn_new_sp)
    
    # Variables to compare against
    rn_vars = [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn]
    
    # Print the expressions for debugging
    print(f"Deprecated real normal form: {H_rn_dep_clean}")
    print(f"New real normal form: {H_rn_new_clean}")
    
    # Check that coefficients match within tolerance
    assert coefficients_match(H_rn_dep_clean, H_rn_new_clean, rn_vars, rtol=1e-9, atol=1e-10), \
        "Real normal forms don't match"


def test_real_normal_to_complex_equivalence(lp, max_degree, psi_clmo):
    """Test that both real normal to complex canonical transformations produce equivalent results."""
    psi, clmo = psi_clmo
    
    # Generate Hamiltonians and transform to real normal
    H_dep = dep_hamiltonian(lp, max_degree)
    H_rn_dep = dep_physical_to_real_normal(lp, H_dep, symbolic=False, max_degree=max_degree)
    
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    H_rn_arrays = physical_to_real_normal_arrays(lp, H_arrays, max_degree, psi, clmo)
    
    # Transform to complex canonical
    H_cc_dep = dep_real_normal_to_complex_canonical(lp, H_rn_dep, symbolic=False, max_degree=max_degree)
    H_cc_arrays = real_normal_to_complex_arrays(lp, H_rn_arrays, max_degree, psi, clmo)
    
    # Convert array implementation to symbolic
    H_cc_new_sym = poly2symengine(H_cc_arrays, [q1, q2, q3, p1, p2, p3], psi, clmo)
    
    # Clean up numerical artifacts
    H_cc_dep_sp = sp.sympify(H_cc_dep.expression)
    H_cc_dep_clean = _clean_numerical_artifacts(H_cc_dep_sp)
    H_cc_new_sp = sp.sympify(H_cc_new_sym)
    H_cc_new_clean = _clean_numerical_artifacts(H_cc_new_sp)
    
    # Variables to compare against
    cc_vars = [q1, q2, q3, p1, p2, p3]
    
    # Print the expressions for debugging
    print(f"Deprecated complex canonical form: {H_cc_dep_clean}")
    print(f"New complex canonical form: {H_cc_new_clean}")
    
    # Check that coefficients match within tolerance
    assert coefficients_match(H_cc_dep_clean, H_cc_new_clean, cc_vars, rtol=1e-8, atol=1e-9), \
        "Complex canonical forms don't match"


def test_complex_to_real_normal_equivalence(lp, max_degree, psi_clmo):
    """Test that both complex canonical to real normal transformations produce equivalent results."""
    psi, clmo = psi_clmo
    
    # Generate Hamiltonians and transform through the chain
    H_dep = dep_hamiltonian(lp, max_degree)
    H_rn_dep = dep_physical_to_real_normal(lp, H_dep, symbolic=False, max_degree=max_degree)
    H_cc_dep = dep_real_normal_to_complex_canonical(lp, H_rn_dep, symbolic=False, max_degree=max_degree)
    
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    H_rn_arrays = physical_to_real_normal_arrays(lp, H_arrays, max_degree, psi, clmo)
    H_cc_arrays = real_normal_to_complex_arrays(lp, H_rn_arrays, max_degree, psi, clmo)
    
    # Transform back to real normal
    H_rn2_dep = dep_complex_canonical_to_real_normal(lp, H_cc_dep, symbolic=False, max_degree=max_degree)
    H_rn2_arrays = complex_to_real_arrays(lp, H_cc_arrays, max_degree, psi, clmo)
    
    # Convert array implementation to symbolic
    H_rn2_new_sym = poly2symengine(H_rn2_arrays, [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], psi, clmo)
    
    # Clean up numerical artifacts
    H_rn2_dep_sp = sp.sympify(H_rn2_dep.expression)
    H_rn2_dep_clean = _clean_numerical_artifacts(H_rn2_dep_sp)
    H_rn2_new_sp = sp.sympify(H_rn2_new_sym)
    H_rn2_new_clean = _clean_numerical_artifacts(H_rn2_new_sp)
    
    # Variables to compare against
    rn_vars = [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn]
    
    # Check that coefficients match within tolerance
    assert coefficients_match(H_rn2_dep_clean, H_rn2_new_clean, rn_vars, rtol=1e-8, atol=1e-9), \
        "Real normal forms after round-trip don't match"


def test_round_trip_transformations(lp, max_degree, psi_clmo):
    """Test that both implementations maintain consistency in round-trip transformations."""
    psi, clmo = psi_clmo
    
    # Generate Hamiltonians
    H_dep = dep_hamiltonian(lp, max_degree)
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    
    # Round-trip: physical -> real normal -> complex canonical -> real normal -> physical
    # Deprecated implementation
    H_rn_dep = dep_physical_to_real_normal(lp, H_dep, symbolic=False, max_degree=max_degree)
    H_cc_dep = dep_real_normal_to_complex_canonical(lp, H_rn_dep, symbolic=False, max_degree=max_degree)
    H_rn2_dep = dep_complex_canonical_to_real_normal(lp, H_cc_dep, symbolic=False, max_degree=max_degree)
    
    # New implementation
    H_rn_arrays = physical_to_real_normal_arrays(lp, H_arrays, max_degree, psi, clmo)
    H_cc_arrays = real_normal_to_complex_arrays(lp, H_rn_arrays, max_degree, psi, clmo)
    H_rn2_arrays = complex_to_real_arrays(lp, H_cc_arrays, max_degree, psi, clmo)
    
    # Convert to symbolic for comparison
    H_rn2_new_sym = poly2symengine(H_rn2_arrays, [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], psi, clmo)
    
    # Clean up numerical artifacts
    H_rn2_dep_sp = sp.sympify(H_rn2_dep.expression)
    H_rn2_dep_clean = _clean_numerical_artifacts(H_rn2_dep_sp)
    H_rn2_new_sp = sp.sympify(H_rn2_new_sym)
    H_rn2_new_clean = _clean_numerical_artifacts(H_rn2_new_sp)
    
    # Variables to compare against
    rn_vars = [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn]
    
    # Check that coefficients match within tolerance
    assert coefficients_match(H_rn2_dep_clean, H_rn2_new_clean, rn_vars, rtol=1e-8, atol=1e-9), \
        "Round-trip transformations don't match"


def test_linear_modes_preservation(lp, max_degree, psi_clmo):
    """Test that linear modes are preserved correctly in both implementations."""
    psi, clmo = psi_clmo
    
    # Get linear modes
    lambda1_val, omega1_val, omega2_val = lp.linear_modes()
    
    # Generate Hamiltonians and transform to real normal
    H_dep = dep_hamiltonian(lp, max_degree)
    H_rn_dep = dep_physical_to_real_normal(lp, H_dep, symbolic=False, max_degree=max_degree)
    
    H_arrays = hamiltonian_arrays(lp, max_degree, psi, clmo)
    H_rn_arrays = physical_to_real_normal_arrays(lp, H_arrays, max_degree, psi, clmo)
    
    # Convert array implementation to symbolic
    H_rn_new_sym = poly2symengine(H_rn_arrays, [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], psi, clmo)
    
    # Clean up numerical artifacts
    H_rn_dep_sp = sp.sympify(H_rn_dep.expression)
    H_rn_new_sp = sp.sympify(H_rn_new_sym)
    
    # Extract the quadratic part (degree 2)
    H2_dep = sp.Poly(H_rn_dep_sp, [str(v) for v in [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn]])
    H2_new = sp.Poly(H_rn_new_sp, [str(v) for v in [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn]])
    
    # Get coefficients of the expected quadratic terms
    # λ₁·x_rn·px_rn term
    coef_dep_lambda = float(H2_dep.coeff_monomial(f"{x_rn}*{px_rn}"))
    coef_new_lambda = float(H2_new.coeff_monomial(f"{x_rn}*{px_rn}"))
    assert np.isclose(coef_dep_lambda, lambda1_val, rtol=1e-6)
    assert np.isclose(coef_new_lambda, lambda1_val, rtol=1e-6)
    
    # ω₁/2·(y_rn²+py_rn²) terms
    coef_dep_omega1_y = float(H2_dep.coeff_monomial(f"{y_rn}**2"))
    coef_new_omega1_y = float(H2_new.coeff_monomial(f"{y_rn}**2"))
    coef_dep_omega1_py = float(H2_dep.coeff_monomial(f"{py_rn}**2"))
    coef_new_omega1_py = float(H2_new.coeff_monomial(f"{py_rn}**2"))
    
    assert np.isclose(coef_dep_omega1_y, omega1_val/2, rtol=1e-6)
    assert np.isclose(coef_new_omega1_y, omega1_val/2, rtol=1e-6)
    assert np.isclose(coef_dep_omega1_py, omega1_val/2, rtol=1e-6)
    assert np.isclose(coef_new_omega1_py, omega1_val/2, rtol=1e-6)
    
    # ω₂/2·(z_rn²+pz_rn²) terms
    coef_dep_omega2_z = float(H2_dep.coeff_monomial(f"{z_rn}**2"))
    coef_new_omega2_z = float(H2_new.coeff_monomial(f"{z_rn}**2"))
    coef_dep_omega2_pz = float(H2_dep.coeff_monomial(f"{pz_rn}**2"))
    coef_new_omega2_pz = float(H2_new.coeff_monomial(f"{pz_rn}**2"))
    
    assert np.isclose(coef_dep_omega2_z, omega2_val/2, rtol=1e-6)
    assert np.isclose(coef_new_omega2_z, omega2_val/2, rtol=1e-6)
    assert np.isclose(coef_dep_omega2_pz, omega2_val/2, rtol=1e-6)
    assert np.isclose(coef_new_omega2_pz, omega2_val/2, rtol=1e-6)
