import numpy as np
import pytest

from algorithms.center.polynomial import base
from algorithms.center.lie import _get_homogeneous_terms, _select_terms_for_elimination, _solve_homological_equation
from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.transforms import phys2rn, rn2cn
from algorithms.center.polynomial.base import decode_multiindex, encode_multiindex
from algorithms.center.polynomial.operations import polynomial_zero_list
from algorithms.center.polynomial.algebra import _poly_poisson
from system.libration import L1Point


@pytest.fixture
def cn_hamiltonian_data(request):
    max_deg = request.param

    # psi table needs to be large enough for n_missing in _get_homogeneous_terms tests
    psi_init_deg = max_deg + 2
    psi, clmo = base.init_index_tables(psi_init_deg)

    # Use a standard mu value (e.g., Earth-Moon L1)
    mu_earth_moon = 0.012150585609624
    point = L1Point(mu=mu_earth_moon)

    # The Hamiltonian itself is constructed up to max_deg.
    # The psi and clmo (initialized for psi_init_deg) are suitable as psi_init_deg >= max_deg.
    H_phys = build_physical_hamiltonian(point, max_deg)
    H_rn = phys2rn(point, H_phys, max_deg, psi, clmo)
    H_coeffs = rn2cn(H_rn, max_deg, psi, clmo)

    return H_coeffs, psi, clmo, max_deg


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_within_H_coeffs(cn_hamiltonian_data):
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

    n = 3  # Test for degree 3 terms
    if n > max_deg:
        # This block handles cases where n=3 but max_deg < 3 (e.g., max_deg=2).
        # If H_coeffs does not have H_coeffs[n], _get_homogeneous_terms should return zeros.
        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        assert np.all(Hn == 0), f"vector for n={n} (n > max_deg={max_deg}) is not zero"
        assert len(Hn) == psi[6, n], f"wrong length for zero vector for n={n} (n > max_deg={max_deg})"
    else:  # n <= max_deg (e.g. max_deg = 3, 4, or 6)
        Hn = _get_homogeneous_terms(H_coeffs, n, psi)
        expected_Hn = H_coeffs[n]
        assert np.array_equal(Hn, expected_Hn), "returned vector is not H_n"
        # must be a *copy*, not the original reference
        if Hn.size > 0:
            original_coeff_val = H_coeffs[n][0] # Save original value from H_coeffs
            Hn[0] += 1.0 # Modify the supposed copy
            assert H_coeffs[n][0] == original_coeff_val, "Original H_coeffs was modified!"
            assert Hn[0] != original_coeff_val, "Copy was not modified or not a proper copy."
        elif expected_Hn.size == 0:
            # Both are empty, this is fine. No copy modification to test on Hn[0].
            pass
        # No 'else' here, as an empty Hn and non-empty expected_Hn would be caught by np.array_equal


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_beyond_H_coeffs_degree(cn_hamiltonian_data):
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

    # H_coeffs extends up to max_deg. We test for a degree n_missing > max_deg.
    # but still within psi_init_deg (max_deg + 2)
    n_missing = max_deg + 1

    Hn_zero = _get_homogeneous_terms(H_coeffs, n_missing, psi)
    assert np.all(Hn_zero == 0), "vector for missing degree is not zero"
    # The length of Hn_zero should correspond to psi[6, n_missing]
    assert len(Hn_zero) == psi[6, n_missing], "wrong length for zero vector"


@pytest.mark.parametrize("cn_hamiltonian_data", [2, 3, 4, 6], indirect=True)
def test_get_homogeneous_terms_when_n_is_at_psi_table_edge(cn_hamiltonian_data):
    H_coeffs, psi, clmo, max_deg = cn_hamiltonian_data

    # This case tests access at psi_init_deg = max_deg + 2.
    n_at_psi_edge = max_deg + 2

    Hn_zero_psi_edge = _get_homogeneous_terms(H_coeffs, n_at_psi_edge, psi)
    assert np.all(Hn_zero_psi_edge == 0), "vector for missing degree (psi edge) is not zero"
    assert len(Hn_zero_psi_edge) == psi[6, n_at_psi_edge], "wrong length for zero vector (psi edge)"


@pytest.mark.parametrize("n", [3, 4, 6])
def test_select_terms_for_elimination(n):
    max_deg = n                       # lookup tables big enough
    psi, clmo = base.init_index_tables(max_deg)

    size = psi[6, n]
    rng  = np.random.default_rng(0)

    # random complex coefficients in [-1,1] + i[-1,1]
    Hn = (rng.uniform(-1, 1, size) + 1j*rng.uniform(-1, 1, size)).astype(np.complex128)

    # ---------- expected mask built in pure Python -----------------------
    expected = np.zeros_like(Hn)
    for pos, c in enumerate(Hn):
        k = decode_multiindex(pos, n, clmo)
        if k[0] != k[3]:          # q1-exponent vs p1-exponent
            expected[pos] = c

    # ---------- routine under test ---------------------------------------
    got = _select_terms_for_elimination(Hn, n, clmo)

    # element-wise equality (complex numbers, so use np.allclose(abs diff == 0))
    assert np.array_equal(got, expected)

    # make sure we didn't mutate the input in-place
    assert np.array_equal(Hn, Hn + 0)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 9])
def test_homological_property(n):
    max_deg = n
    psi, clmo = base.init_index_tables(max_deg)

    # pick arbitrary non-resonant frequencies
    lam, w1, w2 = 3.1, 2.4, 2.2
    eta = np.array([lam, 1j*w1, 1j*w2], dtype=np.complex128)

    # ---- fake degree-n polynomial with random 'bad' terms only ------------
    size = psi[6, n]
    rng  = np.random.default_rng(1234)
    Hn_bad = np.zeros(size, dtype=np.complex128)
    for pos in range(size):
        k = decode_multiindex(pos, n, clmo)
        if k[0] != k[3]:                     # k_q1 ≠ k_p1 → bad
            Hn_bad[pos] = rng.normal() + 1j*rng.normal()

    # ---- call the solver ---------------------------------------------------
    Gn = _solve_homological_equation(Hn_bad, n, eta, clmo)

    # ---- compute {H2,Gn} using Poisson bracket code -----------------------
    # Build H2 in coefficient-list format (degree 2)
    H2 = polynomial_zero_list(max_deg, psi)
    idx = encode_multiindex((1,0,0,1,0,0), 2, psi, clmo)   # q1 p1
    H2[2][idx] = lam
    idx = encode_multiindex((0,1,0,0,1,0), 2, psi, clmo)   # q2 p2
    H2[2][idx] = 1j*w1
    idx = encode_multiindex((0,0,1,0,0,1), 2, psi, clmo)   # q3 p3
    H2[2][idx] = 1j*w2

    # bracket restricted to degree n because both inputs are homogeneous
    # PB = poisson_bracket_degree2(H2[2], Gn, n, psi, clmo) # Old line
    
    # Use _poly_poisson for homogeneous inputs H2[2] (degree 2) and Gn (degree n)
    # Result is homogeneous of degree 2 + n - 2 = n
    PB_coeffs = _poly_poisson(H2[2], 2, Gn, n, psi, clmo)

    # ---- identity check ----------------------------------------------------
    # PB_coeffs must equal -Hn_bad *exactly* (same vector)
    assert np.allclose(PB_coeffs, -Hn_bad, atol=1e-14, rtol=1e-14)

    # bonus: Gn has zero on every "good" index
    for pos, g in enumerate(Gn):
        k = decode_multiindex(pos, n, clmo)
        if k[0] == k[3]:
            assert g == 0