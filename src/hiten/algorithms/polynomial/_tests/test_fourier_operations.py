import numpy as np
import pytest

from hiten.algorithms.polynomial.fourier import (_create_encode_dict_fourier,
                                                 _encode_fourier_index,
                                                 _fpoly_add,
                                                 _fpoly_diff_action,
                                                 _fpoly_diff_angle, _fpoly_mul,
                                                 _fpoly_poisson, _fpoly_scale,
                                                 _init_fourier_tables,
                                                 _make_fourier_poly)

MAX_DEG = 3
K_MAX = 2
PSIF, CLMOF = _init_fourier_tables(MAX_DEG, K_MAX)
ENCODEF = _create_encode_dict_fourier(CLMOF)


def _assert_array_close(a, b, msg=""):
    assert a.shape == b.shape, msg + " shape mismatch"
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12), msg + f"\n{a}\n!=\n{b}"


def _make_monomial(deg, idx_tuple, coeff=1.0):
    """Return a homogeneous Fourier block with a single coefficient set."""
    arr = _make_fourier_poly(deg, PSIF)
    pos = _encode_fourier_index(idx_tuple, deg, ENCODEF)
    assert pos != -1, "Failed to encode index"
    arr[pos] = coeff
    return arr


def _zero_like(deg):
    """Convenience helper: return a zero Fourier poly of given degree."""
    return _make_fourier_poly(deg, PSIF)


def test_fpoly_add_scale():
    p = _make_monomial(0, (0, 0, 0, 0, 0, 0), 2.0)
    q = _make_monomial(0, (0, 0, 0, 0, 0, 0), 3.0)
    out = np.zeros_like(p)

    _fpoly_add(p, q, out)
    expected = _make_monomial(0, (0, 0, 0, 0, 0, 0), 5.0)
    _assert_array_close(out, expected, "addition failed")

    _fpoly_scale(out, -2.0, out)
    expected_scaled = _make_monomial(0, (0, 0, 0, 0, 0, 0), -10.0)
    _assert_array_close(out, expected_scaled, "scaling failed")


def test_fpoly_mul_simple():
    # f = I1   (n1=1)
    f = _make_monomial(1, (1, 0, 0, 0, 0, 0), 1.5)
    # g = I1
    g = _make_monomial(1, (1, 0, 0, 0, 0, 0), 2.0)

    prod = _fpoly_mul(f, 1, g, 1, PSIF, CLMOF, ENCODEF)  # degree 2
    expected = _make_monomial(2, (2, 0, 0, 0, 0, 0), 3.0)
    _assert_array_close(prod, expected, "multiplication failed")


def test_fpoly_diff_action():
    # h = I2^2  (n2=2)  degree 2
    h = _make_monomial(2, (0, 2, 0, 0, 0, 0), 4.0)

    dh_dI2 = _fpoly_diff_action(h, 2, 1, PSIF, CLMOF, ENCODEF)  # action_idx=1 (I2)
    expected = _make_monomial(1, (0, 1, 0, 0, 0, 0), 8.0)
    _assert_array_close(dh_dI2, expected, "action derivative failed")


def test_fpoly_diff_angle():
    # s = exp(i theta3)
    s = _make_monomial(0, (0, 0, 0, 0, 0, 1), 1.0)
    ds_dtheta3 = _fpoly_diff_angle(s, 0, 2, CLMOF)  # angle_idx=2 (theta3)
    expected = _make_monomial(0, (0, 0, 0, 0, 0, 1), 1j * 1.0)
    _assert_array_close(ds_dtheta3, expected, "angle derivative failed")


def test_fpoly_poisson_canonical():
    # f = exp(i theta1)
    f = _make_monomial(0, (0, 0, 0, 1, 0, 0), 1.0)  # k1=1
    # g = I1
    g = _make_monomial(1, (1, 0, 0, 0, 0, 0), 1.0)

    bracket = _fpoly_poisson(g, 1, f, 0, PSIF, CLMOF, ENCODEF)
    expected = _make_monomial(0, (0, 0, 0, 1, 0, 0), -1j * 1.0)
    _assert_array_close(bracket, expected, "{I1, e^{iθ1}} bracket failed")


def test_fpoly_poisson_antisymmetry():
    """Verify antisymmetry: {F, G} = -{G, F}."""

    # F = I1  (degree-1)
    F = _make_monomial(1, (1, 0, 0, 0, 0, 0), 2.0)
    # G = exp(i θ1) (degree-0)
    G = _make_monomial(0, (0, 0, 0, 1, 0, 0), 1.0)

    FG = _fpoly_poisson(F, 1, G, 0, PSIF, CLMOF, ENCODEF)
    GF = _fpoly_poisson(G, 0, F, 1, PSIF, CLMOF, ENCODEF)

    neg_GF = _zero_like(0)
    _fpoly_scale(GF, -1.0, neg_GF)

    _assert_array_close(FG, neg_GF, "antisymmetry failed")


def test_fpoly_poisson_linearity():
    """Linearity in first argument: {aF + bG, H} = a{F,H} + b{G,H}."""

    # Choose simple monomials
    F = _make_monomial(1, (1, 0, 0, 0, 0, 0), 1.0)  # I1
    G = _make_monomial(1, (0, 1, 0, 0, 0, 0), 1.0)  # I2
    H = _make_monomial(0, (0, 0, 0, 1, 0, 0), 1.0)  # e^{i θ1}

    a, b = 2.0, -3.0

    # aF + bG
    aF = _zero_like(1)
    bG = _zero_like(1)
    aF_bG = _zero_like(1)
    _fpoly_scale(F, a, aF)
    _fpoly_scale(G, b, bG)
    _fpoly_add(aF, bG, aF_bG)

    bracket_combined = _fpoly_poisson(aF_bG, 1, H, 0, PSIF, CLMOF, ENCODEF)

    bracket_FH = _fpoly_poisson(F, 1, H, 0, PSIF, CLMOF, ENCODEF)
    bracket_GH = _fpoly_poisson(G, 1, H, 0, PSIF, CLMOF, ENCODEF)

    # a{F,H} + b{G,H}
    a_FH = _zero_like(0)
    b_GH = _zero_like(0)
    sum_expected = _zero_like(0)
    _fpoly_scale(bracket_FH, a, a_FH)
    _fpoly_scale(bracket_GH, b, b_GH)
    _fpoly_add(a_FH, b_GH, sum_expected)

    _assert_array_close(bracket_combined, sum_expected, "linearity failed")


def test_fpoly_poisson_constant():
    """Poisson bracket with a constant must vanish."""

    const = _make_monomial(0, (0, 0, 0, 0, 0, 0), 1.0)  # 1
    F = _make_monomial(1, (1, 0, 0, 0, 0, 0), 1.0)     # I1

    bracket1 = _fpoly_poisson(const, 0, F, 1, PSIF, CLMOF, ENCODEF)
    bracket2 = _fpoly_poisson(F, 1, const, 0, PSIF, CLMOF, ENCODEF)

    zeros0 = _zero_like(0)
    _assert_array_close(bracket1, zeros0, "{1, F} ≠ 0")
    _assert_array_close(bracket2, zeros0, "{F, 1} ≠ 0")


def test_fpoly_poisson_canonical_relations():
    """Canonical relations: {I_i, I_j}=0, {θ_i, θ_j}=0, {I_i, e^{iθ_j}} = -1j δ_{ij} e^{iθ_j}."""

    # Loop over i, j in {0,1,2}
    for i in range(3):
        # Actions I_i
        n_tuple_i = [0, 0, 0]
        n_tuple_i[i] = 1
        I_i = _make_monomial(1, (n_tuple_i[0], n_tuple_i[1], n_tuple_i[2], 0, 0, 0), 1.0)

        for j in range(3):
            # Actions I_j (needed for I_i, I_j test)
            n_tuple_j = [0, 0, 0]
            n_tuple_j[j] = 1
            I_j = _make_monomial(1, (n_tuple_j[0], n_tuple_j[1], n_tuple_j[2], 0, 0, 0), 1.0)

            # Exponentials e^{i θ_j}
            k_tuple_j = [0, 0, 0]
            k_tuple_j[j] = 1
            Theta_j = _make_monomial(0, (0, 0, 0, k_tuple_j[0], k_tuple_j[1], k_tuple_j[2]), 1.0)

            # {I_i, I_j} should be zero
            bracket_II = _fpoly_poisson(I_i, 1, I_j, 1, PSIF, CLMOF, ENCODEF)
            _assert_array_close(bracket_II, _zero_like(1), f"{{I{i+1}, I{j+1}}} ≠ 0")

            # {Theta_i, Theta_j} should be zero
            Theta_i = _make_monomial(0, (0, 0, 0, (1 if i == 0 else 0), (1 if i == 1 else 0), (1 if i == 2 else 0)), 1.0)
            bracket_tt = _fpoly_poisson(Theta_i, 0, Theta_j, 0, PSIF, CLMOF, ENCODEF)
            _assert_array_close(bracket_tt, _zero_like(0), f"{{θ{i+1}, θ{j+1}}} ≠ 0")

            # {I_i, Theta_j}
            bracket_ITheta = _fpoly_poisson(I_i, 1, Theta_j, 0, PSIF, CLMOF, ENCODEF)

            if i == j:
                expected = _make_monomial(0, (0, 0, 0, k_tuple_j[0], k_tuple_j[1], k_tuple_j[2]), -1j * 1.0)
            else:
                expected = _zero_like(0)

            _assert_array_close(bracket_ITheta, expected, f"canonical relation failed for i={i}, j={j}")
