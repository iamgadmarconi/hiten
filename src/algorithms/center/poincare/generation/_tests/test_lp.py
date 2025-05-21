import numpy as np
import pytest

from algorithms.center.poincare.generation.lindstedt_poincare import (
    build_mode_tables, coeff, build_LP)
from system.libration import L1Point


N_MAX = 5

REF = {
    ("omega", 0, 0):  2.086453564223108,
    ("nu",    0, 0):  2.015210662996640,
    ("omega", 2, 0): -1.720616528118309,
    ("nu",    2, 0):  0.2227430750989766,
    ("omega", 0, 2):  0.02581841437578153,
    ("nu",    0, 2): -0.1631915758176957,

    ("X", 1, 0, 1, 0):  1.0,
    ("Y", 1, 0, 1, 0): -3.229268251936296,
    ("Z", 0, 1, 0, 1):  1.0,
    ("X", 2, 0, 0, 0):  2.092695724506778,
    ("Y", 2, 0, 0, 0): -4.778922922033039,
    ("X", 2, 0, 2, 0): -0.9059648301914133,
    ("Y", 2, 0, 2, 0): -0.4924458783826869,
    ("X", 0, 2, 0, 0):  0.2482976576916407,
    ("X", 0, 2, 0, 2):  0.1108251822042930,
    ("Y", 0, 2, 0, 2): -0.06776373426177420,
}

offset_ij, _, _ = build_mode_tables(N_MAX)

def legendre_coeffs(n: int) -> np.ndarray:
    """Return the coefficients of the Legendre polynomial of degree n."""
    c_series = [L1Point(3.040423398444176e-6)._cn(i) for i in range(2, n + 1)]
    return np.array(c_series)

@pytest.fixture(scope="session")
def lp_series():
    c = legendre_coeffs(N_MAX + 2)
    return build_LP(c, N_MAX)

@pytest.mark.parametrize("label,i,j,k,m", [
    ("X", 1,0,1,0),
    ("Y", 1,0,1,0),
    ("Z", 0,1,0,1),
    ("X", 2,0,0,0),
    ("Y", 2,0,0,0),
    ("X", 2,0,2,0),
    ("Y", 2,0,2,0),
    ("X", 0,2,0,0),
    ("X", 0,2,0,2),
    ("Y", 0,2,0,2),
])
def test_xyz(lp_series, label, i, j, k, m):
    X, Y, Z, _, _ = lp_series
    arr = {"X": X, "Y": Y, "Z": Z}[label]
    assert np.allclose(coeff(arr, i, j, k, m, offset_ij), REF[(label, i, j, k, m)], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("label,i,j", [
    ("omega", 0,0),
    ("nu",    0,0),
    ("omega", 2,0),
    ("nu",    2,0),
    ("omega", 0,2),
    ("nu",    0,2),
])
def test_frequencies(lp_series, label, i, j):
    _, _, _, Omega_w, Omega_n = lp_series
    arr = {"omega": Omega_w, "nu": Omega_n}[label]
    assert np.allclose(arr[i, j], REF[(label, i, j)], rtol=1e-12, atol=1e-12)