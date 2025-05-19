from __future__ import annotations

import numpy as np
import pytest
import sympy as sp
from numba import types
from numba.typed import Dict, List

from algorithms.center.hamiltonian import (_build_T_polynomials,
                                           build_physical_hamiltonian)
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.polynomial.conversion import sympy2poly
from algorithms.center.polynomial.operations import (polynomial_add_inplace,
                                                     polynomial_evaluate,
                                                     polynomial_multiply,
                                                     polynomial_variable,
                                                     polynomial_zero_list)
from system.libration import L1Point

_sympy_vars = sp.symbols("x y z px py pz")

@pytest.fixture()
def point() -> L1Point:
    """Return an Earth-Moon L1 point (mu value taken from JPL DE-430)."""
    mu_earth_moon = 0.012150585609624
    return L1Point(mu=mu_earth_moon)

@pytest.fixture(params=[4, 6])
def max_deg(request):
    return request.param

@pytest.fixture()
def psi_clmo(max_deg):
    psi, clmo = init_index_tables(max_deg)
    encode_dict = List.empty_list(types.DictType(types.int64, types.int32))
    for clmo_arr in clmo:
        d_map = Dict.empty(key_type=types.int64, value_type=types.int32)
        for i, packed_val in enumerate(clmo_arr):
            d_map[np.int64(packed_val)] = np.int32(i)
        encode_dict.append(d_map)
    return psi, clmo, encode_dict

def sympy_reference(point: L1Point, max_deg: int) -> sp.Expr:
    """Exact Hamiltonian expanded with SymPy up to *max_deg* total degree."""
    x, y, z, px, py, pz = _sympy_vars
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
            f"Failed to convert SymPy expression to polynomial form in sympy_reference.\n"
            f"Expression: {expanded_H}\n"
            f"Error: {e}"
        )
        raise type(e)(error_msg) from e


@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_symbolic_identity(point, max_deg):
    """Coefficient arrays must match a SymPy ground-truth for small degrees."""

    psi, clmo = init_index_tables(max_deg)
    encode_dict = _create_encode_dict_from_clmo(clmo)
    
    H_build = build_physical_hamiltonian(point, max_deg)

    H_sympy = sympy_reference(point, max_deg)
    H_ref = sympy2poly(H_sympy, _sympy_vars, psi, clmo, encode_dict)

    for d in range(max_deg + 1):
        assert np.allclose(
            H_build[d], H_ref[d], atol=1e-12, rtol=1e-9
        ), f"Mismatch found in degree slice {d}.\nBuild: {H_build[d]}\nRef:   {H_ref[d]}"

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_legendre_recursion(point, max_deg, psi_clmo):
    """Internal `T[n]` sequence must satisfy Legendre three-term recursion."""

    psi, clmo, encode_dict = psi_clmo
    x_poly, y_poly, z_poly, *_ = [
        polynomial_variable(i, max_deg, psi, clmo, encode_dict) for i in range(6)
    ]

    T = _build_T_polynomials(x_poly, y_poly, z_poly, max_deg, psi, clmo, encode_dict)

    sum_sq = polynomial_zero_list(max_deg, psi)
    for var in (x_poly, y_poly, z_poly):
        polynomial_add_inplace(sum_sq, polynomial_multiply(var, var, max_deg, psi, clmo, encode_dict), 1.0)
    
    for n in range(2, max_deg + 1):
        n_ = float(n)
        a = (2 * n_ - 1) / n_
        b = (n_ - 1) / n_

        lhs = T[n]

        term1_mult = polynomial_multiply(x_poly, T[n - 1], max_deg, psi, clmo, encode_dict)
        term1 = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(term1, term1_mult, a)
        
        term2_mult = polynomial_multiply(sum_sq, T[n - 2], max_deg, psi, clmo, encode_dict)
        term2 = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(term2, term2_mult, -b)

        rhs = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(rhs, term1, 1.0)
        polynomial_add_inplace(rhs, term2, 1.0)

        for d in range(max_deg + 1):
            assert np.array_equal(lhs[d], rhs[d]), f"Legendre recursion failed at n={n}, degree slice d={d}"

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_numerical_evaluation(point, max_deg, psi_clmo):
    """Evaluate both Hamiltonians at random points and compare numerically."""

    psi, clmo, _ = psi_clmo
    H_poly = build_physical_hamiltonian(point, max_deg) 
    H_sym = sympy_reference(point, max_deg)

    rng = np.random.default_rng(42)
    vars_syms = sp.symbols("x y z px py pz")

    for _ in range(50):
        vals = rng.uniform(-0.1, 0.1, 6)
        H_num_poly = polynomial_evaluate(H_poly, vals, clmo)
        H_num_sym = float(H_sym.subs(dict(zip(vars_syms, vals))))
        assert np.isclose(
            H_num_poly, H_num_sym, atol=1e-12
        ), "Numerical mismatch between polynomial and SymPy Hamiltonians"
