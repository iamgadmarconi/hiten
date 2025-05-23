import numpy as np
from numba import cuda

from algorithms.center.polynomial.cuda.algebra import _poly_mul_kernel


_pair_list_cache = {}


def poly_mul_cuda(
    p_coeff: np.ndarray,
    p_exp: np.ndarray,
    q_coeff: np.ndarray,
    q_exp: np.ndarray,
    encode_dict_out: np.ndarray,
    n_valid: int,
    *,
    threads_per_block: int = 256,
):
    """Multiply two homogeneous blocks on the GPU.

    Parameters
    ----------
    p_coeff, q_coeff
        1-D arrays of coefficients (float64 or complex128 supported, but stay
        consistent).  Must already live on *host* memory; the helper copies
        them once per call.
    p_exp, q_exp
        1-D arrays of packed exponents, same length as the coefficient arrays.
    encode_dict_out
        1-D `int32` array mapping packed exponent → output position, as built
        by `ENCODE_DICT_GLOBAL[deg_out]`.  All positions must be valid (-1 for
        unmapped).  Will be copied to constant-style device memory once per
        call.
    n_valid
        Number of valid positions in encode_dict_out.
    threads_per_block
        CUDA kernel launch parameter.

    Returns
    -------
    out_coeff : np.ndarray
        Coefficient array for the degree *dₚ+d_q* block (on host).
    """
    # ------------------------------------------------------------------
    # Device copies
    # ------------------------------------------------------------------
    p_coeff_dev = cuda.to_device(p_coeff)
    q_coeff_dev = cuda.to_device(q_coeff)
    p_exp_dev = cuda.to_device(p_exp.astype(np.int32))
    q_exp_dev = cuda.to_device(q_exp.astype(np.int32))
    encode_dict_dev = cuda.to_device(encode_dict_out.astype(np.int32))

    # Build exhaustive pair list on host (baseline – O(n*m)). Cache per (n_p, n_q).
    n_p, n_q = p_coeff.size, q_coeff.size
    cache_key = (n_p, n_q)
    if cache_key in _pair_list_cache:
        idx_pairs = _pair_list_cache[cache_key]
    else:
        n_pairs = n_p * n_q
        idx_pairs = np.empty((n_pairs, 2), dtype=np.int32)
        idx = 0
        for i in range(n_p):
            idx_pairs[idx : idx + n_q, 0] = i
            idx_pairs[idx : idx + n_q, 1] = np.arange(n_q, dtype=np.int32)
            idx += n_q
        _pair_list_cache[cache_key] = idx_pairs
    idx_pairs_dev = cuda.to_device(idx_pairs)

    # Allocate output (zeros by default)
    out_coeff_dev = cuda.device_array(encode_dict_out.shape[0], dtype=p_coeff.dtype)

    # Launch
    blocks = (n_pairs + threads_per_block - 1) // threads_per_block
    _poly_mul_kernel[blocks, threads_per_block](
        p_coeff_dev,
        p_exp_dev,
        q_coeff_dev,
        q_exp_dev,
        out_coeff_dev,
        encode_dict_dev,
        idx_pairs_dev,
    )
    cuda.synchronize()

    out_coeff_host = out_coeff_dev.copy_to_host()
    # Only return the valid part:
    return out_coeff_host[:n_valid]