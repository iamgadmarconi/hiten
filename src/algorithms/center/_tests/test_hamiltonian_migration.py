import pytest
import symengine as se
import numpy as np

# Imports from the new module
from algorithms.center.hamiltonian import (
    hamiltonian_arrays as new_hamiltonian_arrays,
    physical_to_real_normal_arrays as new_physical_to_real_normal_arrays,
    real_normal_to_complex_arrays as new_real_normal_to_complex_arrays,
    complex_to_real_arrays as new_complex_to_real_arrays,
    initialise_tables
)
from algorithms.variables import (
    physical_vars, real_normal_vars, canonical_normal_vars,
    get_vars, create_symbolic_cn,
    linear_modes_vars,
    scale_factors_vars
)

# Imports from the deprecated module
from algorithms.center._deprecated.dep_hamiltonian import (
    hamiltonian as dep_hamiltonian,
    physical_to_real_normal as dep_physical_to_real_normal,
    real_normal_to_complex_canonical as dep_real_normal_to_complex_canonical,
    complex_canonical_to_real_normal as dep_complex_canonical_to_real_normal
)
from algorithms.center._deprecated.dep_core import Polynomial as DepPolynomial

# Import real LibrationPoint
from system.libration import L1Point

# Import the conversion function from its new location
from algorithms.center.polynomial.conversions import poly2symengine

# Define symbolic variables (consistent with both modules)
x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)

# Symbolic parameters for substitutions (matching those in libration.py)
omega1_sym, omega2_sym, lambda1_sym, c2_sym = get_vars(linear_modes_vars)
s1_sym, s2_sym = get_vars(scale_factors_vars)

# Mock LibrationPoint class for initial Hamiltonian structure test
class MockLibrationPoint:
    def __init__(self, mu=0.01):
        self.mu = mu

    def _cn(self, n: int) -> se.Basic:
        # Consistently return create_symbolic_cn for the purely symbolic test
        return create_symbolic_cn(n)

    def linear_modes(self):
        # These are not strictly used by dep_hamiltonian if _cn provides symbols
        # and new_hamiltonian_arrays doesn't use point.linear_modes directly
        return se.Symbol("lambda1_mock"), se.Symbol("omega1_mock"), se.Symbol("omega2_mock")

    def _symbolic_normal_form_transform(self):
        # For transformation tests, a real point is used.
        # This mock is mainly for the first test (hamiltonian_equivalence)
        # which doesn't involve this method directly for comparison.
        # If it were used, Identity makes sense for a simplified mock.
        C = se.eye(6)
        C_inv = se.eye(6)
        return C, C_inv

    def _scale_factor(self, lambda1_val, omega1_val, omega2_val):
         return se.Symbol("s1_mock"), se.Symbol("s2_mock")

@pytest.fixture
def mock_point_symbolic_cn():
    """Mock point that provides purely symbolic c_n coefficients."""
    return MockLibrationPoint()

@pytest.fixture
def real_l1_point():
    """Real L1 point for Earth-Moon system."""
    return L1Point(mu=0.0121505896090214) # mu_EM, a common value

@pytest.fixture
def common_params():
    max_degree = 4 # A reasonable degree for testing
    psi, clmo = initialise_tables(max_degree)
    return max_degree, psi, clmo

def compare_expressions(expr1: se.Basic, expr2: se.Basic, message: str, tol=1e-12):
    # Expand both expressions before differencing
    expanded_expr1 = se.expand(expr1)
    expanded_expr2 = se.expand(expr2)
    diff = se.expand(expanded_expr1 - expanded_expr2)
    
    # If diff is a number, check if it's close to zero
    if isinstance(diff, (se.Number, se.Integer, se.Rational, float, int)):
        assert abs(float(diff)) < tol, f"{message}\nExpr1: {expanded_expr1}\nExpr2: {expanded_expr2}\nDiff: {diff}"
    else:
        # If diff is still symbolic, it must be exactly zero
        assert diff == 0, f"{message}\nExpr1: {expanded_expr1}\nExpr2: {expanded_expr2}\nDiff: {diff}"

# --- Test Cases ---

def test_hamiltonian_equivalence(mock_point_symbolic_cn, common_params):
    """Tests equivalence of the initial Hamiltonian construction using symbolic c_n."""
    max_degree, psi, clmo = common_params
    point = mock_point_symbolic_cn

    # Deprecated version uses point._cn() which returns symbolic cn_k
    dep_H_phys_poly = dep_hamiltonian(point, max_degree)
    dep_H_phys_expr = dep_H_phys_poly.expansion.expression

    # New version internally uses create_symbolic_cn(k)
    new_H_phys_arrays = new_hamiltonian_arrays(point, max_degree, psi, clmo)
    new_H_phys_expr = poly2symengine(new_H_phys_arrays,
                                             get_vars(physical_vars),
                                             psi, clmo)

    compare_expressions(dep_H_phys_expr, new_H_phys_expr,
                        "Hamiltonian in physical coordinates (symbolic c_n) not equivalent.")

def test_physical_to_real_normal_equivalence(real_l1_point, common_params):
    max_degree, psi, clmo = common_params
    point = real_l1_point

    # --- Deprecated Path ---
    # dep_hamiltonian with real_point will use numerical _cn(k) from L1Point
    dep_H_phys_poly = dep_hamiltonian(point, max_degree)
    # symbolic=False ensures _generate_subs_dict is called, subbing numerical params
    dep_H_rn_poly = dep_physical_to_real_normal(point, dep_H_phys_poly, symbolic=False, max_degree=max_degree)
    dep_H_rn_expr = dep_H_rn_poly.expansion.expression

    # --- New Path ---
    # new_hamiltonian_arrays uses create_symbolic_cn(k) internally
    new_H_phys_arrays_sym_cn = new_hamiltonian_arrays(point, max_degree, psi, clmo)
    # new_physical_to_real_normal_arrays uses point._symbolic_normal_form_transform() which has lambda1_sym etc.
    new_H_rn_arrays = new_physical_to_real_normal_arrays(point, new_H_phys_arrays_sym_cn,
                                                          max_degree, psi, clmo)
    # This expression will have symbolic cn_k and symbolic lambda1_sym, etc.
    new_H_rn_expr_fully_symbolic = poly2symengine(new_H_rn_arrays,
                                                          get_vars(real_normal_vars),
                                                          psi, clmo)
    # Substitute numerical values for lambda1_sym, omega1_sym, c2_sym, s1_sym, s2_sym
    new_H_rn_expr_num_modes = point.substitute_parameters(new_H_rn_expr_fully_symbolic)
    
    # Substitute numerical values for remaining cn_k (k>2)
    cn_subs = {create_symbolic_cn(n_val): point._cn(n_val) for n_val in range(3, max_degree + 1)}
    new_H_rn_expr = new_H_rn_expr_num_modes.subs(cn_subs)

    compare_expressions(dep_H_rn_expr, se.expand(new_H_rn_expr), # expand new_H_rn_expr after subs
                        "Physical to Real Normal transformation not equivalent.")

def test_real_normal_to_complex_equivalence(real_l1_point, common_params):
    max_degree, psi, clmo = common_params
    point = real_l1_point

    # --- Generate initial H_rn using both paths, ensuring they are numerically substituted ---
    # Deprecated H_phys -> H_rn (numerically substituted)
    dep_H_phys_poly_for_rn = dep_hamiltonian(point, max_degree)
    dep_H_rn_poly = dep_physical_to_real_normal(point, dep_H_phys_poly_for_rn, symbolic=False, max_degree=max_degree)

    # New H_phys_arrays (symbolic cn) -> H_rn_arrays -> H_rn_expr (numerically substituted)
    new_H_phys_arrays_sym_cn_for_rn = new_hamiltonian_arrays(point, max_degree, psi, clmo)
    new_H_rn_arrays_for_cn = new_physical_to_real_normal_arrays(point, new_H_phys_arrays_sym_cn_for_rn,
                                                                max_degree, psi, clmo)
    new_H_rn_expr_sym_params = poly2symengine(new_H_rn_arrays_for_cn,
                                                      get_vars(real_normal_vars), psi, clmo)
    new_H_rn_expr_num_modes = point.substitute_parameters(new_H_rn_expr_sym_params)
    cn_subs_rn = {create_symbolic_cn(n_val): point._cn(n_val) for n_val in range(3, max_degree + 1)}
    # This is the H_rn Polynomial equivalent for the new path, numerically substituted.
    # For the deprecated function, it expects a Polynomial object as input.
    # We need its symbolic expression to feed into the new function if we were to mix them,
    # but here we are making parallel tracks.
    # The deprecated function `dep_real_normal_to_complex_canonical` will operate on `dep_H_rn_poly`.
    # The new function `new_real_normal_to_complex_arrays` will operate on `new_H_rn_arrays_for_cn`.

    # --- Deprecated transformation ---
    dep_H_cn_poly = dep_real_normal_to_complex_canonical(point, dep_H_rn_poly, symbolic=False, max_degree=max_degree)
    dep_H_cn_expr = dep_H_cn_poly.expansion.expression

    # --- New transformation ---
    new_H_cn_arrays = new_real_normal_to_complex_arrays(point, new_H_rn_arrays_for_cn, # From new H_rn
                                                         max_degree, psi, clmo)
    new_H_cn_expr_fully_symbolic = poly2symengine(new_H_cn_arrays,
                                                          get_vars(canonical_normal_vars),
                                                          psi, clmo, complex_out=True)
    new_H_cn_expr_num_modes = point.substitute_parameters(new_H_cn_expr_fully_symbolic)
    cn_subs_cn = {create_symbolic_cn(n_val): point._cn(n_val) for n_val in range(3, max_degree + 1)}
    new_H_cn_expr = new_H_cn_expr_num_modes.subs(cn_subs_cn)
    
    compare_expressions(dep_H_cn_expr, se.expand(new_H_cn_expr),
                        "Real Normal to Complex Canonical transformation not equivalent.")


def test_complex_to_real_equivalence(real_l1_point, common_params):
    max_degree, psi, clmo = common_params
    point = real_l1_point

    # --- Generate initial H_cn using both paths, numerically substituted ---
    # Deprecated H_phys -> H_rn -> H_cn (numerically substituted)
    dep_H_phys_poly_for_cn = dep_hamiltonian(point, max_degree)
    dep_H_rn_poly_for_cn = dep_physical_to_real_normal(point, dep_H_phys_poly_for_cn, symbolic=False, max_degree=max_degree)
    dep_H_cn_poly = dep_real_normal_to_complex_canonical(point, dep_H_rn_poly_for_cn, symbolic=False, max_degree=max_degree)

    # New H_phys (sym_cn) -> H_rn_arrays -> H_cn_arrays
    new_H_phys_arrays_sym_cn_for_cn = new_hamiltonian_arrays(point, max_degree, psi, clmo)
    new_H_rn_arrays_sym_cn_for_cn = new_physical_to_real_normal_arrays(point, new_H_phys_arrays_sym_cn_for_cn,
                                                                        max_degree, psi, clmo)
    new_H_cn_arrays_for_cnr = new_real_normal_to_complex_arrays(point, new_H_rn_arrays_sym_cn_for_cn,
                                                                 max_degree, psi, clmo)
    # new_H_cn_arrays_for_cnr is the input for the new complex_to_real_arrays transformation.

    # --- Deprecated transformation ---
    dep_H_cnr_poly = dep_complex_canonical_to_real_normal(point, dep_H_cn_poly, symbolic=False, max_degree=max_degree)
    dep_H_cnr_expr = dep_H_cnr_poly.expansion.expression

    # --- New transformation ---
    new_H_cnr_arrays = new_complex_to_real_arrays(point, new_H_cn_arrays_for_cnr, # From new H_cn
                                                  max_degree, psi, clmo)
    new_H_cnr_expr_fully_symbolic = poly2symengine(new_H_cnr_arrays,
                                                           get_vars(real_normal_vars), # Target variables
                                                           psi, clmo, complex_out=True)
    new_H_cnr_expr_num_modes = point.substitute_parameters(new_H_cnr_expr_fully_symbolic)
    cn_subs_cnr = {create_symbolic_cn(n_val): point._cn(n_val) for n_val in range(3, max_degree + 1)}
    new_H_cnr_expr = new_H_cnr_expr_num_modes.subs(cn_subs_cnr)

    compare_expressions(dep_H_cnr_expr, se.expand(new_H_cnr_expr),
                        "Complex Canonical to Real Normal transformation not equivalent.") 