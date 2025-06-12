import numpy as np
from numba.typed import List
import sys
import os

# Ensure the project src directory is on the path
sys.path.append('src')

from algorithms.center.transforms import realify, complexify
from algorithms.center.coordinates import _realify_coordinates, _complexify_coordinates
from algorithms.center.polynomial.base import _create_encode_dict_from_clmo, init_index_tables
from algorithms.center.polynomial.operations import polynomial_variable


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

MAX_DEG = 1  # We only need degree-1 polynomials for linear-consistency checks
PSI, CLMO = init_index_tables(MAX_DEG)
ENCODE_DICT_LIST = _create_encode_dict_from_clmo(CLMO)
VARIABLE_NAMES = ["q1", "q2", "q3", "p1", "p2", "p3"]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_coordinate_transform_consistency():
    """realify/complexify (polynomial) must match coordinate versions for each variable."""
    for var_idx, var_name in enumerate(VARIABLE_NAMES):
        # 1. Build degree-1 polynomial for the single variable
        poly_var = polynomial_variable(var_idx, MAX_DEG, PSI, CLMO, ENCODE_DICT_LIST)

        # 2. Build the corresponding coordinate vector
        coord_var = np.zeros(6, dtype=np.complex128)
        coord_var[var_idx] = 1.0

        print(f"\n🔍 Variable {var_name} (index {var_idx})")
        print("-" * 40)
        print(f"Original polynomial coeffs (degree1): {poly_var[1]}")
        print(f"Coordinate vector: {coord_var}")

        # --- realify (complex → real) ----------------------------------------
        poly_realified = realify(poly_var, MAX_DEG, PSI, CLMO)
        coord_realified = _realify_coordinates(coord_var)
        print("\n1️⃣  REALIFY comparison:")
        print(f"   Polynomial result: {poly_realified[1]}")
        print(f"   Coordinate result: {coord_realified}")
        assert np.allclose(
            poly_realified[1], coord_realified, atol=1e-14
        ), f"realify mismatch for variable {var_name}"

        # --- complexify (real → complex) -------------------------------------
        poly_complexified = complexify(poly_var, MAX_DEG, PSI, CLMO)
        coord_complexified = _complexify_coordinates(coord_var)
        print("\n2️⃣  COMPLEXIFY comparison:")
        print(f"   Polynomial result: {poly_complexified[1]}")
        print(f"   Coordinate result: {coord_complexified}")
        assert np.allclose(
            poly_complexified[1], coord_complexified, atol=1e-14
        ), f"complexify mismatch for variable {var_name}"
        print("-" * 40)


def test_round_trip_identity():
    """Composition of the transforms should return identity (polynomial & coords)."""
    for var_idx, var_name in enumerate(VARIABLE_NAMES):
        poly_var = polynomial_variable(var_idx, MAX_DEG, PSI, CLMO, ENCODE_DICT_LIST)
        coord_var = np.zeros(6, dtype=np.complex128)
        coord_var[var_idx] = 1.0

        print(f"\n🔄 Round-trip checks for {var_name}")

        # Polynomial round-trips
        poly_real_complex = complexify(realify(poly_var, MAX_DEG, PSI, CLMO), MAX_DEG, PSI, CLMO)
        poly_complex_real = realify(complexify(poly_var, MAX_DEG, PSI, CLMO), MAX_DEG, PSI, CLMO)
        assert np.allclose(
            poly_real_complex[1], poly_var[1], atol=1e-14
        ), f"poly realify∘complexify failed for {var_name}"
        assert np.allclose(
            poly_complex_real[1], poly_var[1], atol=1e-14
        ), f"poly complexify∘realify failed for {var_name}"
        print("   Poly round-trip OK ✅")

        # Coordinate round-trips
        coord_real_complex = _complexify_coordinates(_realify_coordinates(coord_var))
        coord_complex_real = _realify_coordinates(_complexify_coordinates(coord_var))
        assert np.allclose(
            coord_real_complex, coord_var, atol=1e-14
        ), f"coord realify∘complexify failed for {var_name}"
        assert np.allclose(
            coord_complex_real, coord_var, atol=1e-14
        ), f"coord complexify∘realify failed for {var_name}"
        print("   Coord round-trip OK ✅")
