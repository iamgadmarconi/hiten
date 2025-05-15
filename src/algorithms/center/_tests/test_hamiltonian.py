from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from algorithms.center.polynomial import base, operations as op, algebra
from algorithms.center.hamiltonian import build_physical_hamiltonian, _build_T_polynomials
from system.libration import L1Point  # noqa: F401 – used in fixture

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def point() -> L1Point:  # noqa: D401 – pytest fixture
    """Return an Earth–Moon L1 point (μ value taken from JPL DE-430)."""
    mu_earth_moon = 0.012150585609624
    return L1Point(mu=mu_earth_moon)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def sympy_reference(point: CollinearPoint, max_deg: int) -> sp.Expr:
    """Exact Hamiltonian expanded with SymPy up to *max_deg* total degree."""
    x, y, z, px, py, pz = sp.symbols("x y z px py pz")
    vars_tuple = (x, y, z, px, py, pz)

    rho2 = x**2 + y**2 + z**2
    rho = sp.sqrt(rho2)

    H = sp.Rational(1, 2) * (px**2 + py**2 + pz**2) + y * px - x * py

    for n in range(2, max_deg + 1):
        cn = point._cn(n)
        Pn_expr = sp.legendre(n, x / rho)
        # Each term cn * rho**n * Pn_expr should be a polynomial.
        # Expanding each term individually might help SymPy simplify it correctly.
        term_to_add = sp.simplify(cn * rho**n * Pn_expr)
        H -= term_to_add
    
    # Expand the full Hamiltonian expression
    expanded_H = sp.simplify(H)

    # Convert to Poly object and back to expression to ensure canonical polynomial form
    # and to catch errors if expanded_H is not a polynomial in vars_tuple.
    try:
        poly_obj = sp.Poly(expanded_H, *vars_tuple)
        return poly_obj.as_expr()
    except sp.PolynomialError as e:
        # This means expanded_H is not recognized by SymPy as a polynomial in vars_tuple.
        # This would be unexpected if the underlying mathematical formulation is correct
        # and implies an issue with how SymPy is simplifying the expression.
        error_msg = (
            f"Failed to convert SymPy expression to polynomial form in sympy_reference.\\n"
            f"Expression: {expanded_H}\\n"
            f"Error: {e}"
        )
        raise type(e)(error_msg) from e


def sympy_to_poly(sym_expr: sp.Expr, max_deg: int, psi, clmo):
    """Convert a SymPy polynomial into the coefficient‑list format used by the code base."""

    poly = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
    vars_syms = sp.symbols("x y z px py pz")

    for term in sym_expr.as_ordered_terms():
        coeff, monom = term.as_coeff_Mul()
        if monom == 1:
            exps = (0, 0, 0, 0, 0, 0)
        else:
            exps = sp.Poly(monom, *vars_syms).monoms()[0]

        deg = sum(exps)
        if deg > max_deg:
            continue  # truncated away in our polynomial representation

        # NOTE: adjust the next line if your encode_multiindex signature differs
        idx = algebra.encode_multiindex(exps, deg, psi, clmo)
        poly[deg][idx] = float(coeff)

    return poly


def evaluate_poly(poly, values, psi, clmo):
    """Brute‑force evaluation of a coefficient‑list polynomial at *values* (ℝ⁶)."""

    total = 0.0
    for deg, coeffs in enumerate(poly):
        if coeffs.size == 0:
            continue
        for pos, c in enumerate(coeffs):
            if c == 0.0:
                continue
            exps = algebra.decode_multiindex(pos, deg, clmo)
            total += c * np.prod(values**np.asarray(exps))
    return total


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_symbolic_identity(point, max_deg):
    """Coefficient arrays must match a SymPy ground‑truth for small degrees."""

    psi, clmo = base.init_index_tables(max_deg)
    bph_psi_config = (None, False)
    bph_clmo_deg = max_deg
    H_build = build_physical_hamiltonian(point, max_deg, bph_psi_config, bph_clmo_deg)

    H_sympy = sympy_reference(point, max_deg)
    H_ref = sympy_to_poly(H_sympy, max_deg, psi, clmo)

    for d in range(max_deg + 1):
        assert np.allclose(
            H_build[d], H_ref[d], atol=1e-12, rtol=1e-9
        ), f"Mismatch found in degree slice {d}.\\nBuild: {H_build[d]}\\nRef:   {H_ref[d]}"


@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_legendre_recursion(point, max_deg):
    """Internal `T[n]` sequence must satisfy Legendre three‑term recursion."""

    psi, clmo = base.init_index_tables(max_deg)
    x_poly, y_poly, z_poly, *_ = [
        op.polynomial_variable(i, max_deg, psi, complex_dtype=False) for i in range(6)
    ]

    # Call the refactored helper function directly
    T = _build_T_polynomials(x_poly, y_poly, z_poly, max_deg, psi, clmo, complex_dt=False)

    # The rest of the test logic for verification remains largely the same,
    # but it uses the T computed above.
    # We still need sum_sq for the RHS of the recursion check.
    sum_sq = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
    for var in (x_poly, y_poly, z_poly): # Use polynomial variables
        op.polynomial_add_inplace(sum_sq, op.polynomial_multiply(var, var, max_deg, psi, clmo), 1.0)
    
    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        # LHS is T[n] from the function call
        lhs = T[n]

        # RHS is constructed using the recursion formula
        # (a * x * T[n-1] - b * sum_sq * T[n-2])
        
        term1_mult = op.polynomial_multiply(x_poly, T[n - 1], max_deg, psi, clmo)
        term1 = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(term1, term1_mult, a)
        
        term2_mult = op.polynomial_multiply(sum_sq, T[n - 2], max_deg, psi, clmo)
        term2 = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(term2, term2_mult, -b) # -b factor

        rhs = op.polynomial_zero_list(max_deg, psi, complex_dtype=False)
        op.polynomial_add_inplace(rhs, term1, 1.0)
        op.polynomial_add_inplace(rhs, term2, 1.0)

        for d in range(max_deg + 1):
            assert np.array_equal(lhs[d], rhs[d]), f"Legendre recursion failed at n={n}, degree slice d={d}"


@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_numerical_evaluation(point, max_deg):
    """Evaluate both Hamiltonians at random points and compare numerically."""

    psi, clmo = base.init_index_tables(max_deg)
    H_poly = build_physical_hamiltonian(point, max_deg, (psi, np.float64), clmo)
    H_sym = sympy_reference(point, max_deg)

    rng = np.random.default_rng(42)
    vars_syms = sp.symbols("x y z px py pz")

    for _ in range(50):
        vals = rng.uniform(-0.1, 0.1, 6)
        H_num_poly = evaluate_poly(H_poly, vals, psi, clmo)
        H_num_sym = float(H_sym.subs(dict(zip(vars_syms, vals))))
        assert np.isclose(
            H_num_poly, H_num_sym, atol=1e-12
        ), "Numerical mismatch between polynomial and SymPy Hamiltonians"
