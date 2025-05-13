import pytest
import symengine as se
import numpy as np

# Imports from the new module
from algorithms.center.hamiltonian import (
    hamiltonian_arrays as new_hamiltonian_arrays,
    physical_to_real_normal_arrays as new_physical_to_real_normal_arrays,
    real_normal_to_complex_arrays as new_real_normal_to_complex_arrays,
    complex_to_real_arrays as new_complex_to_real_arrays,
    poly_list_to_symengine,
    initialise_tables
)
from algorithms.variables import (
    physical_vars, real_normal_vars, canonical_normal_vars,
    get_vars
)

# Imports from the deprecated module
from algorithms.center._deprecated.dep_hamiltonian import (
    hamiltonian as dep_hamiltonian,
    physical_to_real_normal as dep_physical_to_real_normal,
    real_normal_to_complex_canonical as dep_real_normal_to_complex_canonical,
    complex_canonical_to_real_normal as dep_complex_canonical_to_real_normal
)
from algorithms.center._deprecated.dep_core import Polynomial as DepPolynomial
from algorithms.variables import create_symbolic_cn # Used by deprecated module indirectly

# Define symbolic variables (consistent with both modules)
x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)

# Mock LibrationPoint class
class MockLibrationPoint:
    def __init__(self, mu=0.01): # Example mu value
        self.mu = mu
        # Precompute some values that might be used by _cn or linear_modes
        # These are placeholders and might need adjustment based on actual usage in _cn
        self._gamma = (self.mu * (1 - self.mu))** (1/3) # Simplified placeholder

    def _cn(self, n: int) -> se.Basic:
        # Return a unique symbolic constant for each n for testing purposes
        # This matches the behavior of the new hamiltonian module which uses create_symbolic_cn
        # The deprecated one also uses create_symbolic_cn now, so this is consistent.
        if n == 2:
            return se.Symbol(f"c{n}") # old module uses c2 directly for n=2
        return create_symbolic_cn(n)

    def linear_modes(self):
        # Placeholder values for lambda1, omega1, omega2
        # These should be symbolic or carefully chosen numbers if they affect comparisons
        return se.Symbol("lambda1_val"), se.Symbol("omega1_val"), se.Symbol("omega2_val")

    def _symbolic_normal_form_transform(self):
        # Return a 6x6 identity matrix for simplicity in testing transformations.
        # This means physical vars = real_normal_vars in the mock scenario.
        # For more rigorous testing, a more complex matrix could be used.
        C = se.eye(6)
        C_inv = se.eye(6) # For identity, C_inv is C
        return C, C_inv

    def _scale_factor(self, lambda1_val, omega1_val, omega2_val):
        # Placeholder values for scale factors s1, s2
        return se.Symbol("s1_val"), se.Symbol("s2_val")


@pytest.fixture
def mock_point():
    return MockLibrationPoint()

@pytest.fixture
def common_params():
    max_degree = 4 # A reasonable degree for testing
    psi, clmo = initialise_tables(max_degree)
    return max_degree, psi, clmo

def compare_expressions(expr1: se.Basic, expr2: se.Basic, message: str):
    diff = se.expand(expr1 - expr2)
    assert diff == 0, f"{message}\nExpr1: {expr1}\nExpr2: {expr2}\nDiff: {diff}"

# --- Test Cases ---

def test_hamiltonian_equivalence(mock_point, common_params):
    max_degree, psi, clmo = common_params

    # Deprecated version
    dep_H_phys_poly = dep_hamiltonian(mock_point, max_degree)
    dep_H_phys_expr = dep_H_phys_poly.expansion.expression

    # New version
    new_H_phys_arrays = new_hamiltonian_arrays(mock_point, max_degree, psi, clmo)
    new_H_phys_expr = poly_list_to_symengine(new_H_phys_arrays,
                                             [x, y, z, px, py, pz],
                                             psi, clmo)

    compare_expressions(dep_H_phys_expr, new_H_phys_expr,
                        "Hamiltonian in physical coordinates not equivalent.")

def test_physical_to_real_normal_equivalence(mock_point, common_params):
    max_degree, psi, clmo = common_params

    # --- Generate initial H_phys using both methods ---
    # Deprecated H_phys
    dep_H_phys_poly = dep_hamiltonian(mock_point, max_degree)

    # New H_phys (arrays and symbolic for next step)
    new_H_phys_arrays = new_hamiltonian_arrays(mock_point, max_degree, psi, clmo)
    # For the new physical_to_real_normal_arrays, it expects H_phys_arrays as input.
    # The deprecated physical_to_real_normal expects a Polynomial object.

    # --- Deprecated transformation ---
    # The dep_physical_to_real_normal function in dep_hamiltonian.py
    # uses symbolic=True by default if max_degree is not passed, or if symbolic flag is set.
    # For equivalence, we want to compare symbolic forms.
    # It also has a path where it substitutes numerical values if symbolic=False
    # and max_degree is provided. To ensure we are comparing the core symbolic transformation,
    # we'll call it in a way that keeps things symbolic.
    # The mock point's _symbolic_normal_form_transform returns identity,
    # so x_rn should effectively be x, etc.
    dep_H_rn_poly = dep_physical_to_real_normal(mock_point, dep_H_phys_poly, symbolic=True)
    dep_H_rn_expr = dep_H_rn_poly.expansion.expression

    # --- New transformation ---
    new_H_rn_arrays = new_physical_to_real_normal_arrays(mock_point, new_H_phys_arrays,
                                                          max_degree, psi, clmo)
    new_H_rn_expr = poly_list_to_symengine(new_H_rn_arrays,
                                           [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn],
                                           psi, clmo)

    # Since the mock _symbolic_normal_form_transform is identity,
    # H_phys and H_rn should be the same expression but with different variables.
    # To compare them directly, we'd need to substitute vars_phys with vars_rn in dep_H_phys_expr.
    # However, we are comparing the *outputs* of the transformation functions.
    # dep_H_rn_expr is already in terms of real_normal_vars due to the transformation.
    # new_H_rn_expr is also in terms of real_normal_vars.
    compare_expressions(dep_H_rn_expr, new_H_rn_expr,
                        "Physical to Real Normal transformation not equivalent.")

def test_real_normal_to_complex_equivalence(mock_point, common_params):
    max_degree, psi, clmo = common_params

    # --- Generate initial H_rn using both methods' outputs ---
    # Deprecated H_phys -> H_rn
    dep_H_phys_poly_for_rn = dep_hamiltonian(mock_point, max_degree)
    dep_H_rn_poly = dep_physical_to_real_normal(mock_point, dep_H_phys_poly_for_rn, symbolic=True)

    # New H_phys_arrays -> H_rn_arrays
    new_H_phys_arrays_for_rn = new_hamiltonian_arrays(mock_point, max_degree, psi, clmo)
    new_H_rn_arrays = new_physical_to_real_normal_arrays(mock_point, new_H_phys_arrays_for_rn,
                                                          max_degree, psi, clmo)
    # For the new real_normal_to_complex_arrays, it expects H_rn_arrays.
    # The deprecated real_normal_to_complex_canonical expects a Polynomial H_rn.

    # --- Deprecated transformation ---
    dep_H_cn_poly = dep_real_normal_to_complex_canonical(mock_point, dep_H_rn_poly, symbolic=True)
    dep_H_cn_expr = dep_H_cn_poly.expansion.expression

    # --- New transformation ---
    new_H_cn_arrays = new_real_normal_to_complex_arrays(mock_point, new_H_rn_arrays,
                                                         max_degree, psi, clmo)
    new_H_cn_expr = poly_list_to_symengine(new_H_cn_arrays,
                                           [q1, q2, q3, p1, p2, p3],
                                           psi, clmo, complex_out=True) # Match complex output

    compare_expressions(dep_H_cn_expr, new_H_cn_expr,
                        "Real Normal to Complex Canonical transformation not equivalent.")


def test_complex_to_real_equivalence(mock_point, common_params):
    max_degree, psi, clmo = common_params

    # --- Generate initial H_cn using both methods' outputs ---
    # Deprecated H_phys -> H_rn -> H_cn
    dep_H_phys_poly_for_cn = dep_hamiltonian(mock_point, max_degree)
    dep_H_rn_poly_for_cn = dep_physical_to_real_normal(mock_point, dep_H_phys_poly_for_cn, symbolic=True)
    dep_H_cn_poly = dep_real_normal_to_complex_canonical(mock_point, dep_H_rn_poly_for_cn, symbolic=True)

    # New H_phys_arrays -> H_rn_arrays -> H_cn_arrays
    new_H_phys_arrays_for_cn = new_hamiltonian_arrays(mock_point, max_degree, psi, clmo)
    new_H_rn_arrays_for_cn = new_physical_to_real_normal_arrays(mock_point, new_H_phys_arrays_for_cn,
                                                               max_degree, psi, clmo)
    new_H_cn_arrays = new_real_normal_to_complex_arrays(mock_point, new_H_rn_arrays_for_cn,
                                                         max_degree, psi, clmo)
    # For the new complex_to_real_arrays, it expects H_cn_arrays.
    # The deprecated complex_canonical_to_real_normal expects a Polynomial H_cn.

    # --- Deprecated transformation ---
    # Note: the deprecated function also has `symbolic=True` and `max_degree`
    # It appears the `_generate_subs_dict` call within it might try to sub numerical values
    # if symbolic=False. We want symbolic comparison.
    dep_H_cnr_poly = dep_complex_canonical_to_real_normal(mock_point, dep_H_cn_poly, symbolic=True)
    dep_H_cnr_expr = dep_H_cnr_poly.expansion.expression

    # --- New transformation ---
    # The new function `complex_to_real_arrays` also takes a `point` argument, similar to others.
    new_H_cnr_arrays = new_complex_to_real_arrays(mock_point, new_H_cn_arrays,
                                                  max_degree, psi, clmo)
    # The output of complex_to_real_arrays should be in real_normal_vars.
    # The poly_list_to_symengine should use these vars.
    # The deprecated one also returns in real_normal_vars.
    # The complex_out should be True if the array contains complex numbers, but the resulting expression
    # after this specific transformation should be real, assuming inputs are set up for that.
    # Let's check the function signature and usage in hamiltonian.py:
    # `complex_to_real_arrays` calls `_linear_substitution` with `complex_out=True`.
    # This implies the intermediate array representation *might* be complex, but the symbolic
    # expression we reconstruct for comparison should be handled correctly by `poly_list_to_symengine`.
    # The variables for poly_list_to_symengine should be the target variables of the transformation.
    new_H_cnr_expr = poly_list_to_symengine(new_H_cnr_arrays,
                                            [x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn],
                                            psi, clmo, complex_out=True) # Array can be complex, vars are real

    compare_expressions(dep_H_cnr_expr, new_H_cnr_expr,
                        "Complex Canonical to Real Normal transformation not equivalent.") 