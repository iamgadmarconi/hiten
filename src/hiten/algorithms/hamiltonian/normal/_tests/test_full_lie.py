import math

import numpy as np
import pytest
import sympy as sp

from hiten.algorithms.hamiltonian.lie import _apply_poly_transform
from hiten.algorithms.hamiltonian.normal._lie import (
    _lie_transform, _select_nonresonant_terms)
from hiten.algorithms.polynomial.base import (_CLMO_GLOBAL, _PSI_GLOBAL,
                                              _create_encode_dict_from_clmo,
                                              _decode_multiindex,
                                              _encode_multiindex, _make_poly)
from hiten.algorithms.polynomial.conversion import sympy2poly
from hiten.algorithms.polynomial.operations import _polynomial_zero_list
from hiten.system.base import System
from hiten.system.body import Body
from hiten.system.center import CenterManifold
from hiten.utils.constants import Constants

TEST_L_POINT_IDX = 1
TEST_MAX_DEG = 6  # Maximum polynomial degree used throughout the tests


@pytest.fixture(scope="module")
def libration_point():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    distance = Constants.get_orbital_distance("earth", "moon")
    system = System(Earth, Moon, distance)
    libration_point = system.get_libration_point(TEST_L_POINT_IDX)
    return libration_point


@pytest.fixture(scope="module")
def center_manifold(libration_point):
    """Return a CenterManifold instance pre-computed up to TEST_MAX_DEG."""
    cm = CenterManifold(libration_point, TEST_MAX_DEG)
    cm.compute()  # Triggers all intermediate computations and caches results
    return cm


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n", [3, 4, 6])
def test_select_nonresonant_terms(seed, n, libration_point):
    psi, clmo = _PSI_GLOBAL, _CLMO_GLOBAL

    lam, w1, w2 = libration_point.linear_modes
    omega = np.array([lam, -lam, 1j * w1, -1j * w1, 1j * w2, -1j * w2], dtype=np.complex128)

    rng = np.random.default_rng(seed)
    size = psi[6, n]
    Hn_orig = (rng.uniform(-1, 1, size) + 1j * rng.uniform(-1, 1, size)).astype(np.complex128)
    Hn_snapshot = Hn_orig.copy()

    p_elim = _select_nonresonant_terms(Hn_orig, n, omega, clmo)

    # Shape & dtype checks
    assert isinstance(p_elim, np.ndarray)
    assert p_elim.shape == Hn_orig.shape
    assert p_elim.dtype == Hn_orig.dtype

    # Ensure input untouched
    assert np.array_equal(Hn_orig, Hn_snapshot)

    # Coefficient-level verification
    resonance_tol = 1e-12
    for pos in range(size):
        k = _decode_multiindex(pos, n, clmo)
        res_val = (
            (k[0] - k[3]) * omega[0]
            + (k[1] - k[4]) * omega[2]
            + (k[2] - k[5]) * omega[4]
        )
        if abs(res_val) < resonance_tol:
            assert p_elim[pos] == 0j, (
                f"Resonant term at pos {pos} (k={k}) should be zeroed.")
        else:
            assert p_elim[pos] == Hn_orig[pos], (
                f"Non-resonant term at pos {pos} incorrectly modified.")


test_params = [
    pytest.param(
        "base_degG3_Nmax4_realH",
        3,
        (2, 0, 0, 0, 1, 0),
        0.7,
        1.3,
        4,
        id="Base_degG3_Nmax4_realH",
    ),
    pytest.param(
        "high_degG5_Nmax6_realH",
        5,
        (4, 0, 0, 0, 1, 0),
        0.7,
        1.3,
        6,
        id="High_degG5_Nmax6_realH",
    ),
    pytest.param(
        "Nmax6_degG4_realH",
        4,
        (3, 0, 0, 0, 1, 0),
        0.7,
        1.3,
        6,
        id="Nmax6_degG4_realH_Term2_deg6",
    ),
    pytest.param(
        "complexH_degG3_Nmax4",
        3,
        (2, 0, 0, 0, 1, 0),
        0.7,
        1.3 + 0.5j,
        4,
        id="ComplexH_degG3_Nmax4",
    ),
    pytest.param(
        "degG2_Nmax4_realH",
        2,
        (1, 0, 0, 0, 1, 0),
        0.7,
        1.3,
        4,
        id="Low_degG2_Nmax4_realH_K_is_1",
    ),
]


@pytest.mark.parametrize(
    "test_name, G_deg_actual, G_exps, G_coeff_val, H_coeff_val, N_max_test",
    test_params,
)
def test_apply_poly_transform_full(
    test_name,
    G_deg_actual,
    G_exps,
    G_coeff_val,
    H_coeff_val,
    N_max_test,
    libration_point,
):
    psi, clmo = _PSI_GLOBAL, _CLMO_GLOBAL
    encode_dict = _create_encode_dict_from_clmo(clmo)

    # Build simple quadratic Hamiltonian term
    H_deg_actual = 2
    H_exps_tuple = (0, 1, 0, 0, 1, 0)
    idx_H = _encode_multiindex(H_exps_tuple, H_deg_actual, encode_dict)

    H_coeffs_list = _polynomial_zero_list(N_max_test, psi)
    if H_deg_actual <= N_max_test:
        H_coeffs_list[H_deg_actual][idx_H] = H_coeff_val

    # Build generating function at degree G_deg_actual
    G_n_array = _make_poly(G_deg_actual, psi)
    idx_G = _encode_multiindex(G_exps, G_deg_actual, encode_dict)
    G_n_array[idx_G] = G_coeff_val

    # Apply transform (implementation under test)
    H_transformed = _apply_poly_transform(
        H_coeffs_list,
        G_n_array,
        G_deg_actual,
        N_max_test,
        psi,
        clmo,
        encode_dict,
        tol=1e-15,
    )

    # Reference via SymPy
    q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
    coords = (q1, q2, q3, p1, p2, p3)

    Hsym = H_coeff_val * coords[1] * coords[4]  # q2 p2
    Gsym = G_coeff_val
    for i, exp_val in enumerate(G_exps):
        if exp_val:
            Gsym *= coords[i] ** exp_val

    def sympy_poisson(f, g):
        q_vars, p_vars = coords[:3], coords[3:]
        expr = 0
        for i_var in range(3):
            expr += sp.diff(f, q_vars[i_var]) * sp.diff(g, p_vars[i_var])
            expr -= sp.diff(f, p_vars[i_var]) * sp.diff(g, q_vars[i_var])
        return sp.expand(expr)

    K_series = max(1, G_deg_actual - 1)
    current_term = Hsym
    Hsym_ref = Hsym
    for k_val in range(1, K_series + 1):
        current_term = sympy_poisson(current_term, Gsym)
        Hsym_ref += current_term / math.factorial(k_val)

    Href_poly = sympy2poly(Hsym_ref, list(coords), psi, clmo, encode_dict)

    # Compare degree by degree
    assert len(H_transformed) == N_max_test + 1
    for d in range(N_max_test + 1):
        coeff_impl = H_transformed[d]
        coeff_ref = Href_poly[d] if d < len(Href_poly) else np.zeros_like(coeff_impl)

        if coeff_impl.ndim == 0:
            coeff_impl = coeff_impl.reshape(1)
        if coeff_ref.ndim == 0:
            coeff_ref = coeff_ref.reshape(1)

        assert np.allclose(coeff_impl, coeff_ref, atol=1e-14, rtol=1e-14), (
            f"{test_name}: mismatch at degree {d}.")


def test_lie_transform_removes_nonresonant_terms(center_manifold):
    """Verify that the full Lie transform eliminates all non-resonant terms
    of a *physical* Hamiltonian obtained from the CenterManifold pipeline.
    """

    # Use the same psi/clmo tables as produced for the CenterManifold instance
    psi, clmo = center_manifold._psi, center_manifold._clmo

    # Retrieve the 6-dimensional complex modal Hamiltonian (before any Lie NF)
    poly_init = center_manifold._get_complex_modal_form()

    # Extract frequency vector ω from the underlying libration point
    lp = center_manifold.point
    lam, w1, w2 = lp.linear_modes
    omega = np.array([lam, -lam, 1j * w1, -1j * w1, 1j * w2, -1j * w2], dtype=np.complex128)

    # Apply the *full* Lie normalisation
    poly_trans, _, _ = _lie_transform(lp, poly_init, psi, clmo, TEST_MAX_DEG)

    # Check that all non-resonant monomials have been removed up to the truncation order
    tol = 1e-14
    for n in range(3, TEST_MAX_DEG + 1):
        bad = _select_nonresonant_terms(poly_trans[n], n, omega, clmo)
        max_bad = np.max(np.abs(bad)) if bad.size else 0.0
        assert max_bad < tol, (
            f"Non-resonant terms remain at degree {n}: {max_bad:.2e} > {tol:.1e}")

    # The quadratic part must remain unchanged by the Lie transformation
    assert np.allclose(poly_trans[2], poly_init[2], atol=0, rtol=0)

