import numpy as np
import pytest
import sympy as sp
from numba import types
from numba.typed import Dict, List

from hiten.algorithms.center.hamiltonian import (
    _build_R_polynomials, _build_T_polynomials,
    build_lindstedt_poincare_rhs_polynomials, build_physical_hamiltonian)
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              init_index_tables)
from hiten.algorithms.polynomial.conversion import sympy2poly
from hiten.algorithms.polynomial.operations import (polynomial_add_inplace,
                                                    polynomial_evaluate,
                                                    polynomial_multiply,
                                                    polynomial_variable,
                                                    polynomial_zero_list)
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.system.libration.collinear import L1Point
from hiten.utils.constants import Constants

_sympy_vars = sp.symbols("x y z px py pz")

def _get_symbolic_lindstedt_poincare_rhs(point: L1Point, max_deg: int, x_s: sp.Symbol, y_s: sp.Symbol, z_s: sp.Symbol) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Helper function to symbolically build T_n, R_n and RHS of Lindstedt-Poincare EOMs."""
    rho2_s = x_s**2 + y_s**2 + z_s**2
    rho_s = sp.sqrt(rho2_s)

    T_n_sym_list = []
    if max_deg >= 0:
        t0_expr = sp.Integer(1)
        if rho_s != 0:
            try: t0_expr = sp.simplify(rho_s**0 * sp.legendre(0, x_s / rho_s))
            except Exception: t0_expr = sp.Integer(1)
        T_n_sym_list.append(t0_expr)
    if max_deg >= 1:
        t1_expr = x_s
        if rho_s != 0:
            try: t1_expr = sp.simplify(rho_s**1 * sp.legendre(1, x_s / rho_s))
            except Exception: t1_expr = x_s
        T_n_sym_list.append(t1_expr)
    for n_val in range(2, max_deg + 1):
        Tn_s_val = sp.Integer(0)
        if rho_s != 0:
            try: Tn_s_val = sp.simplify(rho_s**n_val * sp.legendre(n_val, x_s / rho_s))
            except Exception: Tn_s_val = sp.Integer(0) if n_val > 0 else sp.Integer(1)
        elif n_val == 0: Tn_s_val = sp.Integer(1)
        else: Tn_s_val = sp.Integer(0)
        T_n_sym_list.append(Tn_s_val)

    R_n_sym_list = []
    if max_deg >=0: R_n_sym_list.append(sp.Integer(-1))
    if max_deg >=1: R_n_sym_list.append(-3 * x_s)
    for n_val in range(2, max_deg + 1):
        term1_R = sp.Rational(2 * n_val + 3, n_val + 2) * x_s * R_n_sym_list[n_val - 1]
        term2_R = sp.Rational(2 * n_val + 2, n_val + 2) * T_n_sym_list[n_val]
        term3_R = sp.Rational(n_val + 1, n_val + 2) * rho2_s * R_n_sym_list[n_val - 2]
        Rn_s = sp.simplify(term1_R - term2_R - term3_R)
        R_n_sym_list.append(Rn_s)
    
    rhs_x_sym_expr = sp.Integer(0)
    for n_val in range(2, max_deg + 1):
        cn_plus_1 = point._cn(n_val + 1)
        rhs_x_sym_expr += sp.Float(cn_plus_1) * (n_val + 1) * T_n_sym_list[n_val]
    rhs_x_sym_expr = sp.expand(rhs_x_sym_expr)

    sum_term_yz_sym_expr = sp.Integer(0)
    for n_val in range(2, max_deg + 1):
        cn_plus_1 = point._cn(n_val + 1)
        if (n_val - 1) < len(R_n_sym_list):
            sum_term_yz_sym_expr += sp.Float(cn_plus_1) * R_n_sym_list[n_val - 1]
    rhs_y_sym_expr = sp.expand(y_s * sum_term_yz_sym_expr)
    rhs_z_sym_expr = sp.expand(z_s * sum_term_yz_sym_expr)
    
    return rhs_x_sym_expr, rhs_y_sym_expr, rhs_z_sym_expr

@pytest.fixture()
def system() -> System:
    """Return a system with Earth-Moon L1 point (mu value taken from JPL DE-430)."""
    primary = Body(
        "Earth",
        Constants.bodies["earth"]["mass"],
        Constants.bodies["earth"]["radius"],
        "blue",
    )
    secondary = Body(
        "Moon",
        Constants.bodies["moon"]["mass"],
        Constants.bodies["moon"]["radius"],
        "gray",
        primary,
    )
    system = System(
        systemConfig(
            primary,
            secondary,
            Constants.get_orbital_distance("Earth", "Moon"),
        )
    )
    return system

@pytest.fixture()
def point(system: System) -> L1Point:
    """Return an Earth-Moon L1 point (mu value taken from JPL DE-430)."""
    return system.get_libration_point(1)

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

def _get_symbolic_physical_hamiltonian(point: L1Point, max_deg: int) -> sp.Expr:
    """Exact Hamiltonian expanded with SymPy up to *max_deg* total degree."""
    x, y, z, px, py, pz = _sympy_vars
    vars_tuple = (x, y, z, px, py, pz)

    rho2 = x**2 + y**2 + z**2
    rho = sp.sqrt(rho2)

    H = sp.Rational(1, 2) * (px**2 + py**2 + pz**2) + y * px - x * py

    for n in range(2, max_deg + 1):
        cn = point._cn(n)
        Pn_expr = sp.legendre(n, x / rho)
        term_to_add = sp.simplify(cn * rho**n * Pn_expr)
        H -= term_to_add

    expanded_H = sp.simplify(H)

    try:
        poly_obj = sp.Poly(expanded_H, *vars_tuple)
        return poly_obj.as_expr()
    except sp.PolynomialError as e:
        error_msg = (
            f"Failed to convert SymPy expression to polynomial form in _get_symbolic_physical_hamiltonian.\n"
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

    H_sympy = _get_symbolic_physical_hamiltonian(point, max_deg)
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
def test_R_polynomial_recursion(point, max_deg, psi_clmo):
    """Internal `R[n]` sequence must satisfy its three-term recursion."""

    psi, clmo, encode_dict = psi_clmo
    x_poly, y_poly, z_poly, *_ = [
        polynomial_variable(i, max_deg, psi, clmo, encode_dict) for i in range(3) # Only x,y,z needed
    ]

    # Generate T_n polynomials (needed for R_n recurrence)
    T_n_list = _build_T_polynomials(x_poly, y_poly, z_poly, max_deg, psi, clmo, encode_dict)
    
    # Generate R_n polynomials
    R_n_list = _build_R_polynomials(x_poly, y_poly, z_poly, T_n_list, max_deg, psi, clmo, encode_dict)

    # Calculate rho_sq = x^2 + y^2 + z^2 polynomial
    rho_sq_poly = polynomial_zero_list(max_deg, psi)
    x_sq = polynomial_multiply(x_poly, x_poly, max_deg, psi, clmo, encode_dict)
    y_sq = polynomial_multiply(y_poly, y_poly, max_deg, psi, clmo, encode_dict)
    z_sq = polynomial_multiply(z_poly, z_poly, max_deg, psi, clmo, encode_dict)
    polynomial_add_inplace(rho_sq_poly, x_sq, 1.0)
    polynomial_add_inplace(rho_sq_poly, y_sq, 1.0)
    polynomial_add_inplace(rho_sq_poly, z_sq, 1.0)

    # The recurrence for R_n starts from n=2, using R_0 and R_1 as base cases.
    for n in range(2, max_deg + 1):
        n_ = float(n)

        # LHS: R_n from the generated list
        lhs = R_n_list[n]

        # RHS terms calculation based on the recurrence relation:
        # R_n = coeff1 * x * R_{n-1} + coeff2 * T_n + coeff3 * rho^2 * R_{n-2}
        
        # Term 1: ((2n+3)/(n+2)) * x * R_{n-1}
        coeff1 = (2.0 * n_ + 3.0) / (n_ + 2.0)
        term1_mult_x_Rnm1 = polynomial_multiply(x_poly, R_n_list[n - 1], max_deg, psi, clmo, encode_dict)
        term1_poly = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(term1_poly, term1_mult_x_Rnm1, coeff1)

        # Term 2: -((2n+2)/(n+2)) * T_n
        coeff2 = - (2.0 * n_ + 2.0) / (n_ + 2.0) # Note the negative sign here
        term2_poly = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(term2_poly, T_n_list[n], coeff2) # T_n_list[n] is T_n

        # Term 3: -((n+1)/(n+2)) * rho^2 * R_{n-2}
        coeff3 = - (n_ + 1.0) / (n_ + 2.0) # Note the negative sign here
        term3_mult_rhosq_Rnm2 = polynomial_multiply(rho_sq_poly, R_n_list[n - 2], max_deg, psi, clmo, encode_dict)
        term3_poly = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(term3_poly, term3_mult_rhosq_Rnm2, coeff3)
        
        # RHS: Sum of the three terms
        rhs = polynomial_zero_list(max_deg, psi)
        polynomial_add_inplace(rhs, term1_poly, 1.0)
        polynomial_add_inplace(rhs, term2_poly, 1.0)
        polynomial_add_inplace(rhs, term3_poly, 1.0)

        for d in range(max_deg + 1):
            assert np.array_equal(lhs[d], rhs[d]), (
                f"R_n recursion failed at n={n}, degree slice d={d}.\n"
                f"LHS ({R_n_list[n][d].shape}):\n{lhs[d]}\n"
                f"RHS ({rhs[d].shape}):\n{rhs[d]}\n"
                f"Term1 ({term1_poly[d].shape}):\n{term1_poly[d]}\n"
                f"Term2 ({term2_poly[d].shape}):\n{term2_poly[d]}\n"
                f"Term3 ({term3_poly[d].shape}):\n{term3_poly[d]}"
            )

@pytest.mark.parametrize("max_deg", [4, 6, 8])
def test_numerical_evaluation(point, max_deg, psi_clmo):
    """Evaluate both Hamiltonians at random points and compare numerically."""

    psi, clmo, _ = psi_clmo
    H_poly = build_physical_hamiltonian(point, max_deg) 
    H_sym = _get_symbolic_physical_hamiltonian(point, max_deg)

    rng = np.random.default_rng(42)
    vars_syms = sp.symbols("x y z px py pz")

    for _ in range(50):
        vals = rng.uniform(-0.1, 0.1, 6)
        H_num_poly = polynomial_evaluate(H_poly, vals, clmo)
        H_num_sym = float(H_sym.subs(dict(zip(vars_syms, vals))))
        assert np.isclose(
            H_num_poly, H_num_sym, atol=1e-12
        ), "Numerical mismatch between polynomial and SymPy Hamiltonians"

@pytest.mark.parametrize("max_deg", [4, 6])
def test_lindstedt_poincare_rhs_symbolic(point, max_deg, psi_clmo):
    """RHS of Lindstedt-Poincare EOMs must match SymPy ground-truth."""
    psi_table, clmo_table, encode_dict_list = psi_clmo

    x_s, y_s, z_s = sp.symbols("x y z")

    # Get symbolic RHS expressions from the helper function
    rhs_x_sym, rhs_y_sym, rhs_z_sym = _get_symbolic_lindstedt_poincare_rhs(point, max_deg, x_s, y_s, z_s)

    # Get computed polynomials
    rhs_x_calc, rhs_y_calc, rhs_z_calc = build_lindstedt_poincare_rhs_polynomials(point, max_deg)

    # Convert SymPy expressions to polynomial coefficient lists.
    rhs_x_ref = sympy2poly(rhs_x_sym, _sympy_vars, psi_table, clmo_table, encode_dict_list)
    rhs_y_ref = sympy2poly(rhs_y_sym, _sympy_vars, psi_table, clmo_table, encode_dict_list)
    rhs_z_ref = sympy2poly(rhs_z_sym, _sympy_vars, psi_table, clmo_table, encode_dict_list)

    # Compare
    for d in range(max_deg + 1):
        assert np.allclose(rhs_x_calc[d], rhs_x_ref[d], atol=1e-12, rtol=1e-9), \
            f"RHS_x mismatch at degree {d}. Calc:\n{rhs_x_calc[d]}\nRef:\n{rhs_x_ref[d]}"
        assert np.allclose(rhs_y_calc[d], rhs_y_ref[d], atol=1e-12, rtol=1e-9), \
            f"RHS_y mismatch at degree {d}. Calc:\n{rhs_y_calc[d]}\nRef:\n{rhs_y_ref[d]}"
        assert np.allclose(rhs_z_calc[d], rhs_z_ref[d], atol=1e-12, rtol=1e-9), \
            f"RHS_z mismatch at degree {d}. Calc:\n{rhs_z_calc[d]}\nRef:\n{rhs_z_ref[d]}"

@pytest.mark.parametrize("max_deg", [4, 6])
def test_lindstedt_poincare_rhs_numerical(point, max_deg, psi_clmo):
    """Numerically evaluate RHS of Lindstedt-Poincare EOMs and compare with SymPy."""
    psi_table, clmo_table, encode_dict_list = psi_clmo

    # Get computed polynomials from the function under test
    rhs_x_calc, rhs_y_calc, rhs_z_calc = build_lindstedt_poincare_rhs_polynomials(point, max_deg)

    # Symbolic variables for evaluation reference
    x_s, y_s, z_s = sp.symbols("x y z")
    symbolic_vars_for_eval = (x_s, y_s, z_s)
    
    # Get symbolic RHS expressions from the helper function
    rhs_x_sym, rhs_y_sym, rhs_z_sym = _get_symbolic_lindstedt_poincare_rhs(point, max_deg, x_s, y_s, z_s)

    rng = np.random.default_rng(seed=43)

    for _ in range(50):
        xyz_vals_array = rng.uniform(-0.1, 0.1, 3)
        sub_dict = dict(zip(symbolic_vars_for_eval, xyz_vals_array))

        # Numerical evaluation of calculated polynomials
        # polynomial_evaluate expects all 6 phase space variables. Fill px,py,pz with 0.
        full_eval_point = np.zeros(6)
        full_eval_point[0:3] = xyz_vals_array # x, y, z
        # full_eval_point[3:6] are already 0 for px, py, pz

        num_rhs_x_calc = polynomial_evaluate(rhs_x_calc, full_eval_point, clmo_table)
        num_rhs_y_calc = polynomial_evaluate(rhs_y_calc, full_eval_point, clmo_table)
        num_rhs_z_calc = polynomial_evaluate(rhs_z_calc, full_eval_point, clmo_table)

        # Numerical evaluation of SymPy reference expressions
        num_rhs_x_sym = float(rhs_x_sym.subs(sub_dict))
        num_rhs_y_sym = float(rhs_y_sym.subs(sub_dict))
        num_rhs_z_sym = float(rhs_z_sym.subs(sub_dict))

        # Compare results
        assert np.isclose(num_rhs_x_calc, num_rhs_x_sym, atol=1e-12), \
            f"Numerical mismatch for RHS_x at point {xyz_vals_array}. Calc: {num_rhs_x_calc}, Sym: {num_rhs_x_sym}"
        assert np.isclose(num_rhs_y_calc, num_rhs_y_sym, atol=1e-12), \
            f"Numerical mismatch for RHS_y at point {xyz_vals_array}. Calc: {num_rhs_y_calc}, Sym: {num_rhs_y_sym}"
        assert np.isclose(num_rhs_z_calc, num_rhs_z_sym, atol=1e-12), \
            f"Numerical mismatch for RHS_z at point {xyz_vals_array}. Calc: {num_rhs_z_calc}, Sym: {num_rhs_z_sym}"
