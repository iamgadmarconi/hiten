from numba import njit, types
from numba.typed import Dict, List

from hiten.algorithms.polynomial.base import _combinations, _decode_multiindex
from hiten.algorithms.polynomial.fourier import (_create_encode_dict_fourier,
                                                 _encode_fourier_index,
                                                 _init_fourier_tables,
                                                 _make_fourier_poly)
from hiten.algorithms.utils.config import FASTMATH


@njit(fastmath=FASTMATH, cache=False)
def _hyperbolic_monomial(q_pow: int, p_pow: int):
    I1_pow = q_pow + p_pow
    k1 = q_pow - p_pow  # Fourier index (can be negative)
    return I1_pow, k1, 1.0 + 0.0j

@njit(fastmath=FASTMATH, cache=False)
def _elliptic_monomial_to_series(q_pow: int, p_pow: int):
    a = q_pow
    b = p_pow
    total = a + b

    I_pow = total // 2

    coeff_dict = Dict.empty(key_type=types.int64, value_type=types.complex128)
    scale_real = 2.0 ** (total / 2.0 - b)
    scale_complex = (0 - 0.5j) ** a  # (-i/2)^a
    scale = scale_real * scale_complex

    for j in range(a + 1):
        binom_a = _combinations(a, j)
        sign = -1 if (j & 1) else 1  # (-1)^j
        exp1 = a - 2 * j

        for l in range(b + 1):
            binom_b = _combinations(b, l)
            exp2 = b - 2 * l
            k = exp1 + exp2  # Fourier index

            coeff = scale * binom_a * binom_b * sign
            if k in coeff_dict:
                coeff_dict[k] = coeff_dict[k] + coeff
            else:
                coeff_dict[k] = coeff

    # Filter out zero coefficients that can arise from cancellation
    # of intermediate terms during the binomial expansion.
    final_dict = Dict.empty(key_type=types.int64, value_type=types.complex128)
    for k, v in coeff_dict.items():
        if abs(v.real) > 1e-9 or abs(v.imag) > 1e-9:
            final_dict[k] = v

    return I_pow, final_dict


@njit(fastmath=FASTMATH, cache=False)
def _populate_fourier(poly_nf_real, clmo, fourier_coeffs, encode_dictF):
    """Internal JIT kernel that fills *fourier_coeffs* in place."""
    tol = 1e-14
    max_deg = len(poly_nf_real) - 1

    for d in range(max_deg + 1):
        block = poly_nf_real[d]
        if block is None:
            continue
        size_block = block.shape[0]
        for pos in range(size_block):
            c0 = block[pos]
            if (c0.real == 0.0 and c0.imag == 0.0):
                continue

            k_vec = _decode_multiindex(pos, d, clmo)
            q1, q2, q3, p1, p2, p3 = k_vec[0], k_vec[1], k_vec[2], k_vec[3], k_vec[4], k_vec[5]

            # Hyperbolic contribution
            I1_pow, k1, coeff_h = _hyperbolic_monomial(q1, p1)

            # Elliptic pair 2
            total2 = q2 + p2
            if total2 & 1:
                # half-integer action -> cannot represent -> skip term
                continue
            I2_pow, dict_k2 = _elliptic_monomial_to_series(q2, p2)

            # Elliptic pair 3
            total3 = q3 + p3
            if total3 & 1:
                continue
            I3_pow, dict_k3 = _elliptic_monomial_to_series(q3, p3)

            # Combine Fourier indices
            for k2_idx in dict_k2.keys():
                c2 = dict_k2[k2_idx]
                for k3_idx in dict_k3.keys():
                    c3 = dict_k3[k3_idx]

                    deg_action = I1_pow + I2_pow + I3_pow
                    total_coeff = c0 * coeff_h * c2 * c3
                    if abs(total_coeff) < tol:
                        continue

                    idx_tuple = (I1_pow, I2_pow, I3_pow, k1, k2_idx, k3_idx)
                    posF = _encode_fourier_index(idx_tuple, deg_action, encode_dictF)
                    if posF == -1:
                        # out of table bounds; skip
                        continue

                    fourier_coeffs[deg_action][posF] += total_coeff


def _realcenter2actionangle(poly_nf_real, clmo, *, k_max: int | None = None):
    r"""
    Convert real centre-manifold Hamiltonian *poly_nf_real* to Fourier-Taylor
    coefficient arrays in action-angle variables.

    Parameters
    ----------
    poly_nf_real : List[np.ndarray]
        Coefficient arrays in variables (q1',q2',q3',p1',p2',p3').
    clmo : numba.typed.List
        Lookup table from `_init_index_tables` used with `_decode_multiindex`.
    k_max : int, optional
        Maximum Fourier index to keep (defaults to maximum polynomial degree).

    Returns
    -------
    fourier_coeffs : numba.typed.List[List[np.ndarray]]
        Coefficient arrays in the packed Fourier representation, one array per
        total action degree.
    psiF, clmoF, encode_dictF : objects
        The lookup tables required for further operations on the Fourier
        polynomials.
    """
    max_deg = len(poly_nf_real) - 1
    if k_max is None:
        k_max = max_deg

    # Build Fourier lookup
    psiF, clmoF = _init_fourier_tables(max_deg, k_max)
    encodeF = _create_encode_dict_fourier(clmoF)

    # Allocate coefficient arrays list
    fourier_coeffs = List()
    for d in range(max_deg + 1):
        fourier_coeffs.append(_make_fourier_poly(d, psiF))

    _populate_fourier(poly_nf_real, clmo, fourier_coeffs, encodeF)

    return fourier_coeffs, psiF, clmoF, encodeF