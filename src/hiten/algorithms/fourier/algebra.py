from __future__ import annotations

import numpy as np
from numba import njit

from hiten.algorithms.fourier.base import (_decode_fourier_index,
                                           _encode_fourier_index)
from hiten.algorithms.utils.config import FASTMATH




@njit(fastmath=FASTMATH, cache=False)
def _fpoly_add(p: np.ndarray, q: np.ndarray, out: np.ndarray) -> None:
    for i in range(p.shape[0]):
        out[i] = p[i] + q[i]


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_scale(p: np.ndarray, alpha, out: np.ndarray) -> None:
    for i in range(p.shape[0]):
        out[i] = alpha * p[i]


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_mul(
    p: np.ndarray,
    deg_p: int,
    q: np.ndarray,
    deg_q: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    deg_r = deg_p + deg_q
    out_len = psiF[deg_r]
    out = np.zeros(out_len, dtype=np.complex128)

    for i in range(p.shape[0]):
        ci = p[i]
        if ci == 0.0:
            continue
        n1a, n2a, n3a, k1a, k2a, k3a = _decode_fourier_index(clmoF[deg_p][i])

        for j in range(q.shape[0]):
            cj = q[j]
            if cj == 0.0:
                continue
            n1b, n2b, n3b, k1b, k2b, k3b = _decode_fourier_index(clmoF[deg_q][j])

            # combined exponents / indices
            n1c = n1a + n1b
            n2c = n2a + n2b
            n3c = n3a + n3b
            k1c = k1a + k1b
            k2c = k2a + k2b
            k3c = k3a + k3b

            idx_tuple = (n1c, n2c, n3c, k1c, k2c, k3c)
            pos = _encode_fourier_index(idx_tuple, deg_r, encodeF)
            if pos != -1:
                out[pos] += ci * cj
    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_diff_action(
    p: np.ndarray,
    deg_p: int,
    action_idx: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    if deg_p == 0:
        return np.zeros_like(p)

    out_deg = deg_p - 1
    out = np.zeros(psiF[out_deg], dtype=np.complex128)

    for i in range(p.shape[0]):
        coeff = p[i]
        if coeff == 0.0:
            continue
        n1, n2, n3, k1, k2, k3 = _decode_fourier_index(clmoF[deg_p][i])
        n = (n1, n2, n3)
        exp_val = n[action_idx]
        if exp_val == 0:
            continue
        n_list = [n1, n2, n3]
        n_list[action_idx] = exp_val - 1
        idx_tuple = (n_list[0], n_list[1], n_list[2], k1, k2, k3)
        pos = _encode_fourier_index(idx_tuple, out_deg, encodeF)
        if pos != -1:
            out[pos] += coeff * exp_val
    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_diff_angle(
    p: np.ndarray,
    deg_p: int,
    angle_idx: int,
    clmoF,
) -> np.ndarray:
    out = np.zeros_like(p)

    for i in range(p.shape[0]):
        coeff = p[i]
        if coeff == 0.0:
            continue

        _n1, _n2, _n3, k1, k2, k3 = _decode_fourier_index(clmoF[deg_p][i])

        k_tuple = (k1, k2, k3)
        k_val = k_tuple[angle_idx]
        if k_val == 0:
            continue

        out[i] = 1j * k_val * coeff

    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_poisson(
    p: np.ndarray,
    deg_p: int,
    q: np.ndarray,
    deg_q: int,
    psiF: np.ndarray,
    clmoF,
    encodeF,
) -> np.ndarray:
    if deg_p == 0 and deg_q == 0:
        return np.zeros(psiF[0], dtype=np.complex128)

    deg_r = deg_p + deg_q - 1  # derivative w.r.t I_j lowers deg by 1
    if deg_r >= psiF.shape[0]:
        # allocate on the fly if beyond current table (rare)
        return np.zeros(1, dtype=np.complex128)

    out = np.zeros(psiF[deg_r], dtype=np.complex128)

    for j in range(3):  # loop over 3 action-angle pairs
        dp_dtheta = _fpoly_diff_angle(p, deg_p, j, clmoF)
        dq_dI     = _fpoly_diff_action(q, deg_q, j, psiF, clmoF, encodeF)

        # Skip invalid combinations that would require negative degrees
        if deg_q > 0:
            term1 = _fpoly_mul(dp_dtheta, deg_p, dq_dI, deg_q - 1, psiF, clmoF, encodeF)
        else:
            term1 = np.zeros_like(out)

        dp_dI   = _fpoly_diff_action(p, deg_p, j, psiF, clmoF, encodeF)
        dq_dtheta = _fpoly_diff_angle(q, deg_q, j, clmoF)

        if deg_p > 0:
            term2 = _fpoly_mul(dp_dI, deg_p - 1, dq_dtheta, deg_q, psiF, clmoF, encodeF)
        else:
            term2 = np.zeros_like(out)

        # Addition/subtraction into out
        for idx in range(out.shape[0]):
            out[idx] += term1[idx] - term2[idx]

    return out


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_block_evaluate(
    coeffs_block: np.ndarray,
    degree: int,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    if coeffs_block.shape[0] == 0:
        return 0.0 + 0.0j

    packed_arr = clmoF[degree]
    total = 0.0 + 0.0j

    I1, I2, I3 = I_vals[0], I_vals[1], I_vals[2]
    th1, th2, th3 = theta_vals[0], theta_vals[1], theta_vals[2]

    for pos in range(coeffs_block.shape[0]):
        c = coeffs_block[pos]
        # Skip zero coefficients quickly
        if c.real == 0.0 and c.imag == 0.0:
            continue

        n1, n2, n3, k1, k2, k3 = _decode_fourier_index(packed_arr[pos])
        # Action powers
        term = c
        if n1:
            term *= I1 ** n1
        if n2:
            term *= I2 ** n2
        if n3:
            term *= I3 ** n3
        # Angle phase factor
        if k1 or k2 or k3:
            phase = np.exp(1j * (k1 * th1 + k2 * th2 + k3 * th3))
            term *= phase
        total += term
    return total




@njit(fastmath=FASTMATH, cache=False)
def _fpoly_block_gradient(
    coeffs_block: np.ndarray,
    degree: int,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    if coeffs_block.shape[0] == 0:
        return 0.0 + 0.0j, np.zeros(3, dtype=np.complex128), np.zeros(3, dtype=np.complex128)

    packed_arr = clmoF[degree]
    val = 0.0 + 0.0j
    gI = np.zeros(3, dtype=np.complex128)
    gT = np.zeros(3, dtype=np.complex128)

    I1, I2, I3 = I_vals[0], I_vals[1], I_vals[2]
    th1, th2, th3 = theta_vals[0], theta_vals[1], theta_vals[2]

    for pos in range(coeffs_block.shape[0]):
        c = coeffs_block[pos]
        if c.real == 0.0 and c.imag == 0.0:
            continue
        n1, n2, n3, k1, k2, k3 = _decode_fourier_index(packed_arr[pos])
        # Evaluate monomial
        I_term = 1.0
        if n1:
            I_term *= I1 ** n1
        if n2:
            I_term *= I2 ** n2
        if n3:
            I_term *= I3 ** n3
        phase = 1.0 + 0.0j
        if k1 or k2 or k3:
            phase = np.exp(1j * (k1 * th1 + k2 * th2 + k3 * th3))
        base_val = c * I_term * phase
        val += base_val

        # Gradients w.r.t actions
        if n1:
            gI[0] += base_val * n1 / I1 if I1 != 0.0 else 0.0
        if n2:
            gI[1] += base_val * n2 / I2 if I2 != 0.0 else 0.0
        if n3:
            gI[2] += base_val * n3 / I3 if I3 != 0.0 else 0.0
        # Gradients w.r.t angles
        if k1:
            gT[0] += 1j * k1 * base_val
        if k2:
            gT[1] += 1j * k2 * base_val
        if k3:
            gT[2] += 1j * k3 * base_val

    return val, gI, gT


@njit(fastmath=FASTMATH, cache=False)
def _fpoly_block_hessian(
    coeffs_block: np.ndarray,
    degree: int,
    I_vals: np.ndarray,
    theta_vals: np.ndarray,
    clmoF,
):
    H = np.zeros((6, 6), dtype=np.complex128)

    if coeffs_block.shape[0] == 0:
        return H

    packed_arr = clmoF[degree]

    I1, I2, I3 = I_vals[0], I_vals[1], I_vals[2]
    th1, th2, th3 = theta_vals[0], theta_vals[1], theta_vals[2]

    for pos in range(coeffs_block.shape[0]):
        c = coeffs_block[pos]
        if c.real == 0.0 and c.imag == 0.0:
            continue

        n1, n2, n3, k1, k2, k3 = _decode_fourier_index(packed_arr[pos])

        # Evaluate the base monomial value at the point (I, theta)
        base_val = c
        if n1:
            base_val *= I1 ** n1
        if n2:
            base_val *= I2 ** n2
        if n3:
            base_val *= I3 ** n3

        if k1 or k2 or k3:
            base_val *= np.exp(1j * (k1 * th1 + k2 * th2 + k3 * th3))

        # Convenience tuples for loop access
        n = (n1, n2, n3)
        I = (I1, I2, I3)
        k = (k1, k2, k3)

        # Action-action second derivatives ∂²H/∂I_a∂I_b
        for a in range(3):
            na = n[a]
            Ia = I[a]
            if na == 0 or Ia == 0.0:
                continue

            # Diagonal term first (a == b)
            if na >= 2:
                contrib = base_val * na * (na - 1) / (Ia * Ia)
                H[a, a] += contrib

            # Off-diagonal terms (a < b)
            for b in range(a + 1, 3):
                nb = n[b]
                Ib = I[b]
                if nb == 0 or Ib == 0.0:
                    continue
                contrib = base_val * na * nb / (Ia * Ib)
                H[a, b] += contrib
                H[b, a] += contrib  # symmetry

        # Action-angle mixed derivatives ∂²H/∂I_a∂theta_b
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
                H[3 + b, a] += contrib  # symmetry

        # Angle-angle second derivatives ∂²H/∂theta_a∂theta_b
        for a in range(3):
            ka = k[a]
            if ka == 0:
                continue
            for b in range(a, 3):
                kb = k[b]
                if kb == 0:
                    continue
                contrib = -base_val * ka * kb  # (i ka)(i kb) = - ka kb
                H[3 + a, 3 + b] += contrib
                if b != a:
                    H[3 + b, 3 + a] += contrib  # symmetry

    return H
