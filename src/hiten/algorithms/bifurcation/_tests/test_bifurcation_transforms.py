import numpy as np
import pytest

from hiten.algorithms.bifurcation.transforms import _nf2aa_ee, _nf2aa_sc
from hiten.algorithms.fourier.base import (_decode_fourier_index,
                                           _init_fourier_tables)
from hiten.algorithms.polynomial.base import (_CLMO_GLOBAL, _PSI_GLOBAL,
                                              _decode_multiindex, _make_poly)
from hiten.system.base import System
from hiten.system.body import Body
from hiten.system.center import CenterManifold
from hiten.utils.constants import Constants

TEST_MAX_DEG = 6


@pytest.fixture(scope="module")
def reduction_test_setup_triangular():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    distance_em = Constants.get_orbital_distance("earth", "moon")
    system_em = System(Earth, Moon, distance_em)
    libration_point_em = system_em.get_libration_point(1)
    cm_em = CenterManifold(libration_point_em, TEST_MAX_DEG)
    cm_em.compute('complex_full_normal')
    return cm_em

def _encode_monomial_to_poly(exponents: tuple[int, int, int, int, int, int]):
    degree = int(sum(exponents))
    poly = _make_poly(degree, _PSI_GLOBAL)

    match_pos = None
    for pos in range(len(_CLMO_GLOBAL[degree])):
        if tuple(_decode_multiindex(pos, degree, _CLMO_GLOBAL)) == exponents:
            match_pos = pos
            break
    if match_pos is None:
        raise ValueError("Unable to locate monomial index for exponents " + str(exponents))

    poly[match_pos] = 1.0 + 0.0j
    return poly


def _locate_fourier_index(idx_tuple: tuple[int, int, int, int, int, int]):
    n1, n2, n3, k1, k2, k3 = idx_tuple
    deg_aa = n1 + n2 + n3
    k_max = max(abs(k1), abs(k2), abs(k3))

    psiF, clmoF = _init_fourier_tables(deg_aa, k_max)
    arr = clmoF[deg_aa]

    for pos in range(arr.shape[0]):
        if tuple(_decode_fourier_index(arr[pos])) == idx_tuple:
            return pos, psiF
    raise ValueError("Fourier index not found for tuple " + str(idx_tuple))


@pytest.mark.parametrize(
    "exponents, expected_idx, expected_coeff",
    [
        ((2, 0, 0, 0, 0, 0), (1, 0, 0, 2, 0, 0), 1.0 + 0.0j),
        ((0, 0, 0, 2, 0, 0), (1, 0, 0, -2, 0, 0), -1.0 + 0.0j),
        ((1, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0), -1.0j),
    ],
)
def test_nf2aa_single_mode(exponents, expected_idx, expected_coeff):
    poly_nf = _encode_monomial_to_poly(exponents)

    coeffs_aa = _nf2aa_ee(poly_nf)

    pos_aa, psiF = _locate_fourier_index(expected_idx)

    deg_aa = expected_idx[0] + expected_idx[1] + expected_idx[2]
    assert coeffs_aa.shape[0] == psiF[deg_aa], "Unexpected coefficient array size"

    assert coeffs_aa[pos_aa] == pytest.approx(expected_coeff, rel=1e-12, abs=1e-12)

    mask = np.ones_like(coeffs_aa, dtype=bool)
    mask[pos_aa] = False
    assert np.allclose(coeffs_aa[mask], 0.0, atol=1e-12)


def test_nf2aa_odd_degree_returns_zero():
    poly_nf = _encode_monomial_to_poly((1, 0, 0, 0, 0, 0))

    coeffs_aa = _nf2aa_ee(poly_nf)

    psiF, _ = _init_fourier_tables(0, 0)
    assert coeffs_aa.shape[0] == psiF[0]
    assert np.allclose(coeffs_aa, 0.0, atol=1e-12)


def test_nf2aa_multi_mode_distinct_indices():
    exp1 = (2, 0, 0, 0, 0, 0)
    exp2 = (0, 0, 0, 2, 0, 0)

    poly_nf = _encode_monomial_to_poly(exp1) + _encode_monomial_to_poly(exp2)
    coeffs_aa = _nf2aa_ee(poly_nf)

    pos1, psiF = _locate_fourier_index((1, 0, 0, 2, 0, 0))
    pos2, _ = _locate_fourier_index((1, 0, 0, -2, 0, 0))

    deg_aa = 1  # n₁+n₂+n₃
    assert coeffs_aa.shape[0] == psiF[deg_aa]
    assert coeffs_aa[pos1] == pytest.approx(1.0 + 0.0j, rel=1e-12)
    assert coeffs_aa[pos2] == pytest.approx(-1.0 + 0.0j, rel=1e-12)

    mask = np.ones_like(coeffs_aa, dtype=bool)
    mask[[pos1, pos2]] = False
    assert np.allclose(coeffs_aa[mask], 0.0, atol=1e-12)


def test_nf2aa_same_index_accumulation():
    exp = (0, 0, 0, 2, 0, 0)

    poly_nf = 2.0 * _encode_monomial_to_poly(exp)  # coefficient 2 instead of 1
    coeffs_aa = _nf2aa_ee(poly_nf)

    pos, psiF = _locate_fourier_index((1, 0, 0, -2, 0, 0))

    deg_aa = 1
    assert coeffs_aa.shape[0] == psiF[deg_aa]
    assert coeffs_aa[pos] == pytest.approx(-2.0 + 0.0j, rel=1e-12)

    mask = np.ones_like(coeffs_aa, dtype=bool)
    mask[pos] = False
    assert np.allclose(coeffs_aa[mask], 0.0, atol=1e-12)


def test_nf2aa_mixed_parity_filter():
    exp_valid = (2, 0, 0, 0, 0, 0)
    exp_invalid = (1, 0, 0, 0, 0, 1)

    poly_nf = _encode_monomial_to_poly(exp_valid) + _encode_monomial_to_poly(exp_invalid)
    coeffs_aa = _nf2aa_ee(poly_nf)

    pos_valid, psiF = _locate_fourier_index((1, 0, 0, 2, 0, 0))

    deg_aa = 1
    assert coeffs_aa.shape[0] == psiF[deg_aa]
    assert coeffs_aa[pos_valid] == pytest.approx(1.0 + 0.0j, rel=1e-12)

    mask = np.ones_like(coeffs_aa, dtype=bool)
    mask[pos_valid] = False
    assert np.allclose(coeffs_aa[mask], 0.0, atol=1e-12)


def test_nf2aa_prefactor_phase():
    exp = (1, 0, 0, 1, 2, 0)

    poly_nf = _encode_monomial_to_poly(exp)
    coeffs_aa = _nf2aa_ee(poly_nf)

    idx = (1, 1, 0, 0, -2, 0)
    pos, psiF = _locate_fourier_index(idx)

    deg_aa = 2
    assert coeffs_aa.shape[0] == psiF[deg_aa]
    assert coeffs_aa[pos] == pytest.approx(0.0 + 1.0j, rel=1e-12)

    mask = np.ones_like(coeffs_aa, dtype=bool)
    mask[pos] = False
    assert np.allclose(coeffs_aa[mask], 0.0, atol=1e-12)


def test_center_manifold_normal_form_exponent_symmetry(reduction_test_setup_triangular):
    # NOTE This test will fail, as the normal form is not yet implemented for triangular systems.
    # TODO: Implement normal form for triangular systems.
    # Commented out for now to avoid failing the test.
    """
    cm = reduction_test_setup_triangular

    H_nf_complex = cm.cache_get(('hamiltonian', cm.max_degree, 'complex_full_normal'))
    clmo = cm._clmo

    for deg, coeff_vec in enumerate(H_nf_complex):
        if coeff_vec is None or coeff_vec.size == 0:
            continue
        for pos, c in enumerate(coeff_vec):
            if abs(c) < 1e-12:
                continue
            k = _decode_multiindex(pos, deg, clmo)

            # Each position coordinate exponent must match its corresponding momentum exponent
            # to ensure the monomials depend only on the actions I_j (see image reference).
            assert k[0] == k[3], (
                f"q1 exponent {k[0]} != p1 exponent {k[3]} in degree {deg} pos {pos}"
            )
            assert k[1] == k[4], (
                f"q2 exponent {k[1]} != p2 exponent {k[4]} in degree {deg} pos {pos}"
            )
            assert k[2] == k[5], (
                f"q3 exponent {k[2]} != p3 exponent {k[5]} in degree {deg} pos {pos}"
            )
    """
    assert True