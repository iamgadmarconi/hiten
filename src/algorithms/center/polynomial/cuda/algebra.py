import numpy as np
from numba import cuda, int32, float64


@cuda.jit(device=True, inline=True)
def _add_encoded(exp_a: int32, exp_b: int32) -> int32: # type: ignore
    res: int32 = 0 # type: ignore
    for k in range(6):
        mask = int32(0x3F) # 6 bits
        shift = k * 6
        ea = (exp_a >> shift) & mask
        eb = (exp_b >> shift) & mask
        res |= ((ea + eb) & mask) << shift
    return res


@cuda.jit
def _poly_mul_kernel(
    p_coeff: float64[:], # type: ignore
    p_exp: int32[:], # type: ignore
    q_coeff: float64[:], # type: ignore
    q_exp: int32[:], # type: ignore
    out_coeff: float64[:], # type: ignore
    encode_dict: int32[:],  # maps packed exponent → position (‑1 if missing) # type: ignore
    idx_pairs: int32[:, :],  # (Npairs,2) – indices into p & q # type: ignore
):
    pair_idx: int32 = cuda.grid(1) # type: ignore
    if pair_idx >= idx_pairs.shape[0]:
        return

    i = idx_pairs[pair_idx, 0]
    j = idx_pairs[pair_idx, 1]

    coef = p_coeff[i] * q_coeff[j]
    exp_out = _add_encoded(p_exp[i], q_exp[j])

    pos = encode_dict[exp_out]
    # Safety: skip if monomial is not in the truncated output basis.
    if pos >= 0:
        cuda.atomic.add(out_coeff, pos, coef)
