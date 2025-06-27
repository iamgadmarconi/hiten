from typing import Dict, Tuple

import numpy as np
from numba.typed import List

from hiten.algorithms.polynomial.base import _create_encode_dict_from_clmo
from hiten.algorithms.polynomial.operations import (
    _polynomial_clean,
    _polynomial_variable,
    _polynomial_multiply,
    _substitute_linear,
)
from hiten.algorithms.center.transforms import _M


def _poly2dict(poly_list: List[np.ndarray], *, tol: float) -> Dict[Tuple[int, int], complex]:
    """Return a sparse dictionary representation of a polynomial list.

    Entries with absolute value ≤ *tol* are discarded.

    Parameters
    ----------
    poly_list
        Polynomial in the packed-coefficient representation (list of NumPy
        arrays, one per homogeneous degree).
    tol
        Numerical tolerance for dropping negligible coefficients.

    Returns
    -------
    dict[(int,int), complex]
        Keys are ``(degree, position)`` pairs, values are complex
        coefficients.
    """
    sparse: Dict[Tuple[int, int], complex] = {}
    for d, coeffs_d in enumerate(poly_list):
        if coeffs_d.size == 0:
            continue
        for pos, coeff in enumerate(coeffs_d):
            if coeff != 0 and np.abs(coeff) > tol:
                sparse[(d, pos)] = complex(coeff)
    return sparse


def _realcenter2actionangle(
    poly_cm_real,
    max_deg: int,
    psi,
    clmo,
    *,
    tol: float = 1e-14,
) -> Tuple[
    Dict[Tuple[int, int], complex],
    Dict[Tuple[int, int], complex],
    Dict[Tuple[int, int], complex],
]:
    """Convert a real centre-manifold polynomial to (complex) action–angle form.

    The routine performs two tasks:

    1.  A *linear* change of variables from *real centre* coordinates
        :math:`(\hat q_1,\hat q_2,\hat q_3,\hat p_1,\hat p_2,\hat p_3)` to
        complex modal variables using the same matrix :math:`M` employed in
        the normal-form pipeline (see :pyfunc:`hiten.algorithms.center.transforms._M`).
        In complex coordinates each elliptic pair forms the usual
        harmonic-oscillator combination

        .. math:: z_j = \frac{1}{\sqrt 2}(q_j + i\,p_j), \qquad j=2,3.

        For the hyperbolic pair (:math:`j=1`) the same linear map is applied
        resulting in *real* variables that are convenient for subsequent
        analysis.

    2.  Construction of the three *action* invariants

        \[ I_1 = q_1 p_1, \qquad I_2 = q_2 p_2, \qquad I_3 = q_3 p_3. \]

        Each action is returned as a sparse dictionary mapping
        ``(degree, index)`` to its complex coefficient, using the same packed
        representation as the rest of the polynomial algebra back-end.

    Parameters
    ----------
    poly_cm_real
        Hamiltonian (or any polynomial) expressed in real centre-manifold
        coordinates.
    max_deg
        Truncation degree of *poly_cm_real*.
    psi, clmo
        Pre-computed combinatorial tables from
        :pyfunc:`hiten.algorithms.polynomial.base._init_index_tables`.
    tol
        Coefficient magnitude below which entries are discarded when
        generating the sparse output dictionaries.

    Returns
    -------
    (dict, dict, dict)
        Sparse dictionaries for *I₁*, *I₂* and *I₃* in that order.
    """

    # ------------------------------------------------------------------
    # 1) Linear transformation real → complex (proxy for action–angle)
    # ------------------------------------------------------------------
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    poly_cm_complex = _substitute_linear(
        poly_cm_real, _M(), max_deg, psi, clmo, encode_dict_list
    )

    poly_cm_complex = _polynomial_clean(poly_cm_complex, tol)

    # ------------------------------------------------------------------
    # 2) Build polynomial representations of the quadratic actions
    # ------------------------------------------------------------------
    I_polys: List[List[np.ndarray]] = List()

    # Variable indices in the canonical ordering: q1,q2,q3,p1,p2,p3
    pairs = ((0, 3), (1, 4), (2, 5))  # (q_idx, p_idx)

    for q_idx, p_idx in pairs:
        # Build actions as products q_i * p_i
        q_poly = _polynomial_variable(q_idx, max_deg, psi, clmo, encode_dict_list)
        p_poly = _polynomial_variable(p_idx, max_deg, psi, clmo, encode_dict_list)

        I_poly = _polynomial_multiply(
            q_poly, p_poly, max_deg, psi, clmo, encode_dict_list
        )
        I_polys.append(_polynomial_clean(I_poly, tol))

    # ------------------------------------------------------------------
    # 3) Convert polynomial lists to sparse dictionaries for lightweight use
    # ------------------------------------------------------------------
    I_dicts = tuple(_poly2dict(p, tol=tol) for p in I_polys)

    return I_dicts