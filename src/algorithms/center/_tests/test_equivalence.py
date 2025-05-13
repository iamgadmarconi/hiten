import random
import numpy as np
import symengine as se
import pytest


from algorithms.center.polynomial.base import (
    N_VARS, init_index_tables, encode_multiindex, decode_multiindex,
    symengine_to_custom_poly
)


@pytest.mark.parametrize("complex_flag", [False, True])
def test_symengine_conversion_identity(complex_flag):
    psi, clmo = init_index_tables(4)
    # build a toy expression with real and optional complex coeffs
    expr = 3*x**2*p1 - 5*y*p2 + 7
    if complex_flag:
        expr += (2+1j)*q2**2
    arrays = symengine_to_custom_poly(expr, [x, y, z, px, py, pz], 4, psi, clmo, complex_dtype=complex_flag)
    expr_back = new_ham.poly_list_to_symengine(arrays, [x, y, z, px, py, pz], psi, clmo)
    assert se.expand(expr - expr_back) == 0

# ----------------------------------------------------------------------------
#  3. Old vs new centre‑manifold up to degree 4 ------------------------------
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("max_deg", [3, 4])
def test_center_manifold_equivalence(max_deg):
    pt = DummyPoint()

    # new implementation
    new_H_cm, _ = new_ham.compute_center_manifold_arrays(pt, max_deg)

    # old implementation ➜ Polynomial ➜ arrays
    old_poly, _ = old_ham.compute_center_manifold(pt, symbolic=False, max_degree=max_deg)
    psi, clmo = init_index_tables(max_deg)
    old_arrays = symengine_to_custom_poly(old_poly.expansion.expression,
                                          [q1, q2, q3, p1, p2, p3],
                                          max_deg, psi, clmo,
                                          complex_dtype=True)

    # degree‑by‑degree comparison (tight tolerance)
    for d in range(max_deg+1):
        assert np.allclose(new_H_cm[d], old_arrays[d], atol=1e-10, rtol=1e-10)
