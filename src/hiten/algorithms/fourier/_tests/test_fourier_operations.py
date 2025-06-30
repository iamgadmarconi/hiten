import numpy as np
import pytest
from numba.typed import List

from hiten.algorithms.fourier.algebra import (_encode_fourier_index,
                                              _fpoly_add, _fpoly_diff_action,
                                              _fpoly_diff_angle, _fpoly_mul,
                                              _fpoly_poisson, _fpoly_scale)
from hiten.algorithms.fourier.base import (_create_encode_dict_fourier,
                                           _init_fourier_tables)
from hiten.algorithms.fourier.operations import (_fourier_evaluate,
                                                 _fourier_evaluate_with_grad,
                                                 _fourier_hessian,
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
    _assert_array_close(bracket, expected, "{I1, e^{itheta1}} bracket failed")


def test_fpoly_poisson_antisymmetry():
    """Verify antisymmetry: {F, G} = -{G, F}."""

    # F = I1  (degree-1)
    F = _make_monomial(1, (1, 0, 0, 0, 0, 0), 2.0)
    # G = exp(i theta1) (degree-0)
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
    H = _make_monomial(0, (0, 0, 0, 1, 0, 0), 1.0)  # e^{i theta1}

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
    """Canonical relations: {I_i, I_j}=0, {theta_i, theta_j}=0, {I_i, e^{itheta_j}} = -1j δ_{ij} e^{itheta_j}."""

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

            # Exponentials e^{i theta_j}
            k_tuple_j = [0, 0, 0]
            k_tuple_j[j] = 1
            Theta_j = _make_monomial(0, (0, 0, 0, k_tuple_j[0], k_tuple_j[1], k_tuple_j[2]), 1.0)

            # {I_i, I_j} should be zero
            bracket_II = _fpoly_poisson(I_i, 1, I_j, 1, PSIF, CLMOF, ENCODEF)
            _assert_array_close(bracket_II, _zero_like(1), f"{{I{i+1}, I{j+1}}} ≠ 0")

            # {Theta_i, Theta_j} should be zero
            Theta_i = _make_monomial(0, (0, 0, 0, (1 if i == 0 else 0), (1 if i == 1 else 0), (1 if i == 2 else 0)), 1.0)
            bracket_tt = _fpoly_poisson(Theta_i, 0, Theta_j, 0, PSIF, CLMOF, ENCODEF)
            _assert_array_close(bracket_tt, _zero_like(0), f"{{theta{i+1}, theta{j+1}}} ≠ 0")

            # {I_i, Theta_j}
            bracket_ITheta = _fpoly_poisson(I_i, 1, Theta_j, 0, PSIF, CLMOF, ENCODEF)

            if i == j:
                expected = _make_monomial(0, (0, 0, 0, k_tuple_j[0], k_tuple_j[1], k_tuple_j[2]), -1j * 1.0)
            else:
                expected = _zero_like(0)

            _assert_array_close(bracket_ITheta, expected, f"canonical relation failed for i={i}, j={j}")


def test_fourier_evaluate_single_monomial():
    deg = 2
    idx = (1, 1, 0, 1, -1, 0)
    coeff = 0.8 - 0.2j

    block = _make_monomial(deg, idx, coeff)

    coeffs_list = List()
    for d in range(MAX_DEG + 1):
        if d == deg:
            coeffs_list.append(block)
        else:
            coeffs_list.append(_zero_like(d))

    I_vals = np.array([0.5, 1.2, 0.8])
    theta_vals = np.array([0.3, -0.7, 1.1])

    val = _fourier_evaluate(coeffs_list, I_vals, theta_vals, CLMOF)

    n1, n2, n3, k1, k2, k3 = idx
    expected = coeff * (I_vals[0] ** n1) * (I_vals[1] ** n2) * (I_vals[2] ** n3) * np.exp(
        1j * (k1 * theta_vals[0] + k2 * theta_vals[1] + k3 * theta_vals[2])
    )

    assert np.allclose(val, expected)


def test_fourier_evaluate_with_grad_single_monomial():
    deg = 1
    idx = (1, 0, 0, 2, 0, 0)
    coeff = 2.0 + 1.0j

    block = _make_monomial(deg, idx, coeff)

    coeffs_list = List()
    for d in range(MAX_DEG + 1):
        if d == deg:
            coeffs_list.append(block)
        else:
            coeffs_list.append(_zero_like(d))

    I_vals = np.array([1.5, 0.7, 2.0])
    theta_vals = np.array([0.4, 1.2, -0.3])

    val, gI, gT = _fourier_evaluate_with_grad(coeffs_list, I_vals, theta_vals, CLMOF)

    n1, n2, n3, k1, k2, k3 = idx
    base_val = coeff * (I_vals[0] ** n1) * (I_vals[1] ** n2) * (I_vals[2] ** n3) * np.exp(
        1j * (k1 * theta_vals[0] + k2 * theta_vals[1] + k3 * theta_vals[2])
    )

    expected_gI = np.zeros(3, dtype=np.complex128)
    expected_gT = np.zeros(3, dtype=np.complex128)
    expected_gI[0] = base_val * n1 / I_vals[0]
    expected_gT[0] = 1j * k1 * base_val

    assert np.allclose(val, base_val)
    assert np.allclose(gI, expected_gI)
    assert np.allclose(gT, expected_gT)


def _analytic_hessian_single_monomial(idx, coeff, I_vals, theta_vals):
    """Compute expected 6x6 Hessian for a single Fourier monomial analytically."""
    n1, n2, n3, k1, k2, k3 = idx
    n = (n1, n2, n3)
    k = (k1, k2, k3)
    I1, I2, I3 = I_vals
    th1, th2, th3 = theta_vals

    base_val = coeff * (I1 ** n1) * (I2 ** n2) * (I3 ** n3) * np.exp(
        1j * (k1 * th1 + k2 * th2 + k3 * th3)
    )

    H = np.zeros((6, 6), dtype=np.complex128)

    I = (I1, I2, I3)

    # Action-action second derivatives
    for a in range(3):
        na = n[a]
        Ia = I[a]
        if na >= 2 and Ia != 0.0:
            H[a, a] += base_val * na * (na - 1) / (Ia * Ia)
        for b in range(a + 1, 3):
            nb = n[b]
            Ib = I[b]
            if na > 0 and nb > 0 and Ia != 0.0 and Ib != 0.0:
                contrib = base_val * na * nb / (Ia * Ib)
                H[a, b] += contrib
                H[b, a] += contrib

    # Action-angle mixed derivatives
    for a in range(3):
        na = n[a]
        Ia = I[a]
        if na == 0 or Ia == 0.0:
            continue
        factor_I = na / Ia
        for b in range(3):
            kb = k[b]
            if kb == 0:
                continue
            contrib = base_val * factor_I * 1j * kb
            H[a, 3 + b] += contrib
            H[3 + b, a] += contrib

    # Angle-angle second derivatives
    for a in range(3):
        ka = k[a]
        if ka == 0:
            continue
        for b in range(a, 3):
            kb = k[b]
            if kb == 0:
                continue
            contrib = -base_val * ka * kb
            H[3 + a, 3 + b] += contrib
            if b != a:
                H[3 + b, 3 + a] += contrib

    return H


def test_fourier_hessian_single_monomial():
    deg = 3
    idx = (2, 1, 0, 2, 1, 0)  # provides non-zero contributions in all blocks
    coeff = 1.3 - 0.4j

    block = _make_monomial(deg, idx, coeff)

    coeffs_list = List()
    for d in range(MAX_DEG + 1):
        if d == deg:
            coeffs_list.append(block)
        else:
            coeffs_list.append(_zero_like(d))

    I_vals = np.array([1.2, 0.8, 1.5])
    theta_vals = np.array([0.3, -0.2, 1.1])

    H_calculated = _fourier_hessian(coeffs_list, I_vals, theta_vals, CLMOF)

    H_expected = _analytic_hessian_single_monomial(idx, coeff, I_vals, theta_vals)

    _assert_array_close(H_calculated, H_expected, "Hessian single monomial mismatch")

    # Symmetry check
    _assert_array_close(H_calculated, H_calculated.T, "Hessian not symmetric")


def _finite_difference_hessian(coeffs_list, I_vals, theta_vals, eps=1e-6):
    """Compute Hessian numerically via central finite differences of gradients."""

    base_grad_I, base_grad_T = _fourier_evaluate_with_grad(coeffs_list, I_vals, theta_vals, CLMOF)[1:]
    g0 = np.hstack((base_grad_I, base_grad_T))

    H_fd = np.zeros((6, 6), dtype=np.complex128)

    # Concatenate variables for easier perturbation
    vars_vec = np.hstack((I_vals.copy(), theta_vals.copy()))

    for j in range(6):
        vars_plus = vars_vec.copy()
        vars_minus = vars_vec.copy()
        vars_plus[j] += eps
        vars_minus[j] -= eps

        I_plus = vars_plus[:3]
        th_plus = vars_plus[3:]
        I_minus = vars_minus[:3]
        th_minus = vars_minus[3:]

        grad_plus_I, grad_plus_T = _fourier_evaluate_with_grad(coeffs_list, I_plus, th_plus, CLMOF)[1:]
        grad_minus_I, grad_minus_T = _fourier_evaluate_with_grad(coeffs_list, I_minus, th_minus, CLMOF)[1:]

        grad_plus = np.hstack((grad_plus_I, grad_plus_T))
        grad_minus = np.hstack((grad_minus_I, grad_minus_T))

        # Central difference for column j
        H_fd[:, j] = (grad_plus - grad_minus) / (2 * eps)

    return H_fd


def _random_valid_index(max_deg, k_max):
    d = np.random.randint(0, max_deg + 1)
    n1 = np.random.randint(0, d + 1)
    n2 = np.random.randint(0, d - n1 + 1)
    n3 = d - n1 - n2
    k1 = np.random.randint(-k_max, k_max + 1)
    k2 = np.random.randint(-k_max, k_max + 1)
    k3 = np.random.randint(-k_max, k_max + 1)
    return (d, (n1, n2, n3, k1, k2, k3))


def test_fourier_hessian_finite_difference():
    """Compare analytical Hessian with finite-difference approximation."""

    # Build a small random polynomial with a few monomials across degrees
    coeffs_list = List()
    for d in range(MAX_DEG + 1):
        coeffs_list.append(_zero_like(d))

    rng = np.random.default_rng(seed=123)
    for _ in range(5):
        deg, idx = _random_valid_index(MAX_DEG, K_MAX)
        c = rng.normal() + 1j * rng.normal()
        pos = _encode_fourier_index(idx, deg, ENCODEF)
        coeffs_list[deg][pos] += c  # accumulate if repeats

    I_vals = np.array([1.1, 0.9, 0.8])
    theta_vals = np.array([0.2, -0.5, 1.0])

    H_analytic = _fourier_hessian(coeffs_list, I_vals, theta_vals, CLMOF)
    H_fd = _finite_difference_hessian(coeffs_list, I_vals, theta_vals)

    assert np.allclose(H_analytic, H_fd, rtol=1e-9, atol=1e-9), "Hessian finite-difference mismatch"

    # Ensure symmetry
    _assert_array_close(H_analytic, H_analytic.T, "Analytic Hessian not symmetric")
