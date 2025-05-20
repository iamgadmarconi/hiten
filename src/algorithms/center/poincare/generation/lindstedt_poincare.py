import numpy as np
from numba import njit, prange


@njit(fastmath=True, cache=True)
def scale_series(series: np.ndarray, factor: float, out: np.ndarray) -> np.ndarray:
    """
    Multiply series by factor and store in out.
    
    Parameters
    ----------
    series : np.ndarray
        Input series
    factor : float
        Scale factor
    out : np.ndarray
        Output array to store result
        
    Returns
    -------
    np.ndarray
        Scaled series (out)
    """
    out[:] = series * factor
    return out


@njit(fastmath=True, cache=True)
def _omega0(c2: float) -> float:
    return np.sqrt((2-c2+np.sqrt(9*c2**2-8*c2))/2)


@njit(fastmath=True, cache=True)
def _nu0(c2: float) -> float:
    return np.sqrt(c2)


@njit(fastmath=True, cache=True)
def _kappa(c2: float, omega0: float) -> float:
    """Compute kappa analytically as in the image."""
    return -((omega0**2 + 1 + 2 * c2) / (2 * omega0))


@njit(fastmath=True, cache=True)
def _k_to_idx(i: int, k: int) -> int:
    return (k + i) // 2


@njit(fastmath=True, cache=True)
def _m_to_idx(j: int, m: int) -> int:
    return (m + j) // 2


@njit(fastmath=True, cache=True)
def linear_index(i: int, j: int, k_idx: int, m_idx: int, offset_ij: np.ndarray) -> int:
    return offset_ij[i, j] + k_idx * (j + 1) + m_idx


@njit(fastmath=True, cache=True)
def convolve_series(A, B, i_t, j_t, offset_ij, mode_list, out):
    """Compute the coefficient block (i_t, j_t, *, *) of
    the C = A★B convolution and write it into *out* (same size as A).
    Assumes *out* is already zeroed by the caller.
    """
    # loop over all decompositions (i1,j1)+(i2,j2)=(i_t,j_t)
    for i1 in range(i_t + 1):
        for j1 in range(j_t + 1):
            i2 = i_t - i1
            j2 = j_t - j1
            # (j2 is automatically ≥0 now)

            listA = mode_list[(i1, j1)]
            listB = mode_list[(i2, j2)]
            offA  = offset_ij[i1, j1]
            offB  = offset_ij[i2, j2]
            offC  = offset_ij[i_t, j_t]

            for a_idx in range(listA.shape[0]):
                k1, m1 = listA[a_idx, 4], listA[a_idx, 5]
                idxA   = offA + a_idx
                for b_idx in range(listB.shape[0]):
                    k2, m2 = listB[b_idx, 4], listB[b_idx, 5]
                    k = k1 + k2
                    m = m1 + m2
                    if abs(k) > i_t or abs(m) > j_t:            # outside admissible range
                        continue
                    if zero_coeff(i_t, j_t, k, m):              # killed by symmetry
                        continue
                    idxB   = offB + b_idx
                    # locate output index (using a simple scan; could memoise)
                    modesC = mode_list[(i_t, j_t)]
                    # small linear search – mode lists are short
                    for c_idx in range(modesC.shape[0]):
                        if modesC[c_idx, 4] == k and modesC[c_idx, 5] == m:
                            out[offC + c_idx] += A[idxA] * B[idxB]
                            break


@njit(fastmath=True, cache=True)
def zero_coeff(i, j, k, m):
    """
    Symmetry/parity: only allow Fourier-Taylor modes where
    (i - |k|) and (j - |m|) are even, ensuring the correct parity
    of Taylor vs Fourier indices.
    """
    return ((i - abs(k)) & 1) != 0 or ((j - abs(m)) & 1) != 0


@njit(fastmath=True, cache=True)
def build_mode_tables(N_max):
    offset_ij = np.zeros((N_max+1, N_max+1), dtype=np.int64)
    total = 0
    mode_list = {}
    for i in range(N_max+1):
        for j in range(N_max+1):
            offset_ij[i, j] = total
            modes = []
            for k in range(-i, i+1):
                for m in range(-j, j+1):
                    if not zero_coeff(i, j, k, m):
                        k_idx = _k_to_idx(i, k)
                        m_idx = _m_to_idx(j, m)
                        modes.append((i, j, k_idx, m_idx, k, m))
                        total += 1
            mode_list[(i, j)] = np.array(modes, dtype=np.int64)
    
    total_coeffs = offset_ij[N_max, N_max] + len(mode_list[(N_max, N_max)])
    return offset_ij, mode_list, total_coeffs


@njit(fastmath=True, cache=True)
def precompute_matrices(c2, omega0, nu0, N_max, mode_list, offset_ij):
    """
    Pre-factor the 3x3 matrices for each (k,m) to avoid re-computing entries.
    """
    # For each (i,j) we allocate storage for the matrix elements and determinants
    # Structure: [a11, a12, a21, a22, a33, det, det_inv]
    matrix_cache = {}
    for i in range(N_max+1):
        for j in range(N_max+1):
            modes = mode_list[(i, j)]
            matrices = np.zeros((len(modes), 7), dtype=np.float64)
            for m_idx in range(modes.shape[0]):
                k, m = modes[m_idx, 4], modes[m_idx, 5]  # Extract k, m from mode info
                ell = k*omega0 + m*nu0
                
                a11 = -(ell*ell + 1 + 2*c2)
                a12 = -2.0 * ell
                a21 = -2.0 * ell
                a22 = (c2 - 1 - ell*ell)
                a33 = (c2 - ell*ell)
                det = a11*a22 - a12*a21
                
                matrices[m_idx, 0] = a11
                matrices[m_idx, 1] = a12
                matrices[m_idx, 2] = a21
                matrices[m_idx, 3] = a22
                matrices[m_idx, 4] = a33
                matrices[m_idx, 5] = det
                if det != 0:
                    matrices[m_idx, 6] = 1.0 / det
                else:
                    matrices[m_idx, 6] = 0.0  # Handle singular case
            
            matrix_cache[(i, j)] = matrices
    
    return matrix_cache


@njit(fastmath=True, cache=True)
def build_rho2_series(i, j, X_arr, Y_arr, Z_arr, offset_ij, mode_list, tmp):
    """
    Build the Fourier-Taylor series of rho^2 = x^2 + y^2 + z^2 up to (i,j).
    """
    rho2 = np.zeros_like(X_arr)
    # x^2
    convolve_series(X_arr, X_arr, i, j, offset_ij, mode_list, tmp)
    rho2[:] = tmp
    # y^2
    tmp.fill(0.0)
    convolve_series(Y_arr, Y_arr, i, j, offset_ij, mode_list, tmp)
    rho2[:] += tmp
    # z^2
    tmp.fill(0.0)
    convolve_series(Z_arr, Z_arr, i, j, offset_ij, mode_list, tmp)
    rho2[:] += tmp
    return rho2


@njit(fastmath=True, cache=True)
def seed_series(omega0, nu0, kappa, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n):
    modes10 = mode_list[(1,0)]
    base10 = offset_ij[1,0]
    for idx_m in range(modes10.shape[0]):
        k, m = modes10[idx_m, 4], modes10[idx_m, 5]
        if k == 1 and m == 0:
            X_arr[base10 + idx_m] = 1.0
            Y_arr[base10 + idx_m] = kappa
    
    modes01 = mode_list[(0,1)]
    base01 = offset_ij[0,1]
    for idx_m in range(modes01.shape[0]):
        k, m = modes01[idx_m, 4], modes01[idx_m, 5]
        if k == 0 and m == 1:
            Z_arr[base01 + idx_m] = 1.0
    
    Omega_w[0,0] = omega0
    Omega_n[0,0] = nu0


@njit(fastmath=True, cache=True)
def compute_forcing_terms(i, j, c, Tn, Tn1, Tn2, Rn, Rn1, Rn2, Rn3, X_arr, Y_arr, Z_arr, offset_ij, mode_list, 
                          p, q, r, tmp):
    n = i + j
    # rotate old buffers
    for idx in range(Tn.size):
        Tn2[idx] = Tn1[idx]
        Tn1[idx] = Tn[idx]
        Rn3[idx] = Rn2[idx]
        Rn2[idx] = Rn1[idx]
        Rn1[idx] = Rn[idx]

    # compute rho^2 up to (i,j)
    rho2 = build_rho2_series(i, j, X_arr, Y_arr, Z_arr, offset_ij, mode_list, tmp)

    # T_n = ((2n-1)/n)*X·Tn1 - ((n-1)/n)*rho2·Tn2
    scale1 = (2.0*n - 1.0)/n
    tmp.fill(0.0)
    convolve_series(X_arr, Tn1, i, j, offset_ij, mode_list, tmp)
    scale_series(tmp, scale1, Tn)
    convolve_series(rho2, Tn2, i, j, offset_ij, mode_list, tmp)
    scale_series(tmp, -(n - 1.0)/n, tmp)
    Tn += tmp

    # p = c[n+1]*(n+1)*Tn
    scalar = c[n+1] * (n+1)
    scale_series(Tn, scalar, p)

    # R_{n-1} if available
    if n >= 1:
        # Rn1 = (2n+1)/(n+1)*X·Rn2 - 2n/(n+1)*Tn1 - n/(n+1)*rho2·Rn3
        scaleR1 = (2.0*n + 1.0)/(n+1)
        tmp.fill(0.0)
        convolve_series(X_arr, Rn2, i, j, offset_ij, mode_list, tmp)
        scale_series(tmp, scaleR1, Rn1)
        
        tmp.fill(0.0)
        convolve_series(Tn1, np.ones_like(X_arr), i, j, offset_ij, mode_list, tmp)
        scale_series(tmp, -(2.0*n)/(n+1), tmp)
        Rn1 += tmp
        
        tmp.fill(0.0)
        convolve_series(rho2, Rn3, i, j, offset_ij, mode_list, tmp)
        scale_series(tmp, -n/(n+1), tmp)
        Rn1 += tmp
        
        # q = scalar * (Y·Rn1)
        convolve_series(Y_arr, Rn1, i, j, offset_ij, mode_list, tmp)
        scale_series(tmp, scalar, q)
        # r = scalar * (Z·Rn1)
        convolve_series(Z_arr, Rn1, i, j, offset_ij, mode_list, tmp)
        scale_series(tmp, scalar, r)


@njit(parallel=True, fastmath=True, cache=True)
def compute_LHS_contrib(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, pbar, qbar, rbar, Omega_w, Omega_n):
    modes = mode_list[(i, j)]
    off = offset_ij[i, j]

    # Pre-compute the "known" truncated frequencies
    omega_eff = omega0
    nu_eff    = nu0

    # include Ω_w[a,b], Ω_n[a,b] with a+b < i+j
    for a in range(i+1):
        for b in range(j+1):
            if a == 0 and b == 0:
                continue
            if a + b >= i + j:          # same total degree → current, leave unknown
                continue
            omega_eff +=   Omega_w[a, b]
            nu_eff    +=   Omega_n[a, b]

    for m_idx in prange(modes.shape[0]):
        k, m = modes[m_idx, 4], modes[m_idx, 5]
        idx = off + m_idx
        # For the two resonant fundamentals we MUST exclude the yet-unknown
        # frequency correction ( Ω_w[i-1,j] or Ω_n[i,j-1] ) from ell:
        if   (k == 1 and m == 0):   # in-plane resonance
            ell = k*omega_eff + m*nu_eff          # == omega_eff
        elif (k == 0 and m == 1):   # out-of-plane resonance
            ell = k*omega_eff + m*nu_eff          # == nu_eff
        else:
            ell = k*omega_eff + m*nu_eff

        pbar[idx] = -ell*ell * X_arr[idx] - 2.0 * ell * Y_arr[idx] - (1.0 + 2.0*c2) * X_arr[idx]
        qbar[idx] = -ell*ell * Y_arr[idx] + 2.0 * (-ell * X_arr[idx]) + (c2 - 1.0) * Y_arr[idx]
        rbar[idx] = -ell*ell * Z_arr[idx] + c2 * Z_arr[idx]


@njit(fastmath=True, cache=True)
def solve_order(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n, 
                p, q, r, pbar, qbar, rbar, matrix_cache):
    modes = mode_list[(i, j)]
    off = offset_ij[i, j]
    matrices = matrix_cache[(i, j)]
    
    for m_idx in range(modes.shape[0]):
        k, m = modes[m_idx, 4], modes[m_idx, 5]
        idx = off + m_idx
        
        b1 = pbar[idx] - p[idx]
        b2 = qbar[idx] - q[idx]
        b3 = rbar[idx] - r[idx]
        
        # always use the pre-factored matrix row
        row = matrices[m_idx]
        a11, a12, a21, a22, a33, det, det_inv = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        
        if k == 1 and m == 0:
            X_arr[idx] = 0.0
            Y_arr[idx] = (a22*b1 - a12*b2) * det_inv
            Omega_w[i-1, j] = (a11*b2 - a21*b1) * det_inv / (2.0*omega0)
            if a33 != 0.0:
                Z_arr[idx] = b3 / a33
            else:                       # a33 = 0  ⇒  λ-block also resonant
                Z_arr[idx] = 0.0        # normalise the halo component
                # remove the secular term by letting ν-correction absorb it
                Omega_n[i, j-1] += b3 / (2.0*nu0)   # safe: ν0 ≠ 0
        elif k == 0 and m == 1:
            Z_arr[idx] = 0.0
            X_arr[idx] = (a22*b1 - a12*b2) * det_inv
            Y_arr[idx] = (a11*b2 - a21*b1) * det_inv
            Omega_n[i, j-1] = b3 / (2.0 * nu0)
        else:
            # --- generic (non-resonant) row: invert the full 2×2 block ---
            X_arr[idx] = (a22*b1 - a12*b2) * det_inv
            Y_arr[idx] = (a11*b2 - a21*b1) * det_inv
            Z_arr[idx] = b3 / a33


@njit(fastmath=True, cache=True)
def compute_LP_series(N_max, c2, c, omega0, nu0, kappa, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n,
                     Tn, Tn1, Tn2, Rn, Rn1, Rn2, Rn3, p, q, r, pbar, qbar, rbar, tmp):
    # Always pre-factor matrices for performance
    matrix_cache = precompute_matrices(c2, omega0, nu0, N_max, mode_list, offset_ij)
    
    seed_series(omega0, nu0, kappa, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n)
    
    # We cannot parallelize the outer loop because orders must be computed sequentially
    for total in range(1, N_max+1):
        # ---- keep the outer iteration serial: the scratch buffers are shared ----
        if total > 10:  
            for i in range(total+1):
                j = total - i
                compute_forcing_terms(i, j, c, Tn, Tn1, Tn2, Rn, Rn1, Rn2, Rn3, X_arr, Y_arr, Z_arr, offset_ij, mode_list, p, q, r, tmp)
                compute_LHS_contrib(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, pbar, qbar, rbar, Omega_w, Omega_n)
                solve_order(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n, 
                           p, q, r, pbar, qbar, rbar, matrix_cache)
        else:
            for i in range(total+1):
                j = total - i
                compute_forcing_terms(i, j, c, Tn, Tn1, Tn2, Rn, Rn1, Rn2, Rn3, X_arr, Y_arr, Z_arr, offset_ij, mode_list, p, q, r, tmp)
                compute_LHS_contrib(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, pbar, qbar, rbar, Omega_w, Omega_n)
                solve_order(i, j, c2, omega0, nu0, X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n, 
                           p, q, r, pbar, qbar, rbar, matrix_cache)
                
    return X_arr, Y_arr, Z_arr, Omega_w, Omega_n


@njit(fastmath=True, cache=True)
def build_LP(c, n_max):
    # Initialize tables and arrays
    offset_ij, mode_list, total_coeffs = build_mode_tables(n_max)
    
    # Coefficient arrays
    X_arr = np.zeros(total_coeffs, dtype=np.float64)
    Y_arr = np.zeros_like(X_arr)
    Z_arr = np.zeros_like(X_arr)
    Omega_w = np.zeros((n_max+1, n_max+1), dtype=np.float64)
    Omega_n = np.zeros_like(Omega_w)
    
    # Scratch buffers for recurrences and forcing
    Tn  = np.zeros_like(X_arr)
    Tn1 = np.zeros_like(X_arr)
    Tn2 = np.zeros_like(X_arr)
    Rn  = np.zeros_like(X_arr)
    Rn1 = np.zeros_like(X_arr)
    Rn2 = np.zeros_like(X_arr)
    Rn3 = np.zeros_like(X_arr)
    
    p   = np.zeros_like(X_arr)
    q   = np.zeros_like(X_arr)
    r   = np.zeros_like(X_arr)
    
    pbar = np.zeros_like(X_arr)
    qbar = np.zeros_like(X_arr)
    rbar = np.zeros_like(X_arr)
    
    tmp = np.zeros_like(X_arr)  # Scratch buffer for convolutions
    
    c2 = c[0] 

    omega0 = _omega0(c2)
    nu0 = _nu0(c2)
    kappa = _kappa(c2, omega0)
    
    return compute_LP_series(n_max, c2, c, omega0, nu0, kappa, 
                           X_arr, Y_arr, Z_arr, offset_ij, mode_list, Omega_w, Omega_n,
                           Tn, Tn1, Tn2, Rn, Rn1, Rn2, Rn3, 
                           p, q, r, pbar, qbar, rbar, tmp)


@njit(fastmath=True, cache=True)
def eval_lp(alpha, beta, X_arr, Y_arr, Z_arr, max_order):
    xp = yp = zp = 0.0
    idx = 0
    for i in range(max_order + 1):
        for j in range(max_order + 1 - i):
            amp = (alpha**i)*(beta**j)
            modes = (j + 1)*(i + 1)          # number of (k,m) pairs
            xp += np.sum(X_arr[idx:idx+modes]) * amp
            yp += np.sum(Y_arr[idx:idx+modes]) * amp
            zp += np.sum(Z_arr[idx:idx+modes]) * amp
            idx += modes
    return xp, yp, zp