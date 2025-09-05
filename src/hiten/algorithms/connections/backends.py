from typing import Tuple

import numpy as np
from numba import njit

from hiten.algorithms.connections.results import _ConnectionResult


@njit(cache=False)
def _pair_counts(query: np.ndarray, ref: np.ndarray, r2: float) -> np.ndarray:
    n_q = query.shape[0]
    n_r = ref.shape[0]
    counts = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        x = query[i, 0]
        y = query[i, 1]
        c = 0
        for j in range(n_r):
            dx = x - ref[j, 0]
            dy = y - ref[j, 1]
            if dx * dx + dy * dy <= r2:
                c += 1
        counts[i] = c
    return counts


@njit(cache=False)
def _exclusive_prefix_sum(a: np.ndarray) -> np.ndarray:
    n = a.size
    out = np.empty(n + 1, dtype=np.int64)
    out[0] = 0
    s = 0
    for i in range(n):
        s += int(a[i])
        out[i + 1] = s
    return out


@njit(cache=False)
def _radpair2d(query: np.ndarray, ref: np.ndarray, radius: float) -> np.ndarray:
    r2 = float(radius) * float(radius)
    counts = _pair_counts(query, ref, r2)
    offs = _exclusive_prefix_sum(counts)
    total = int(offs[-1])
    pairs = np.empty((total, 2), dtype=np.int64)

    n_q = query.shape[0]
    n_r = ref.shape[0]
    for i in range(n_q):
        write = offs[i]
        x = query[i, 0]
        y = query[i, 1]
        for j in range(n_r):
            dx = x - ref[j, 0]
            dy = y - ref[j, 1]
            if dx * dx + dy * dy <= r2:
                pairs[write, 0] = i
                pairs[write, 1] = j
                write += 1
    return pairs


def _radius_pairs_2d(query: np.ndarray, ref: np.ndarray, radius: float) -> np.ndarray:
    """Return pairs (i,j) where ||query[i]-ref[j]|| <= radius on a 2D plane.

    Parameters
    ----------
    query, ref : (N,2) and (M,2) float arrays
        2D plane coordinates.
    radius : float
        Match radius.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(ref, dtype=np.float64)
    return _radpair2d(q, r, float(radius))


@njit(cache=False)
def _nearest_neighbor_2d_numba(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        best = 1e300
        best_j = -1
        xi = points[i, 0]
        yi = points[i, 1]
        for j in range(n):
            if j == i:
                continue
            dx = xi - points[j, 0]
            dy = yi - points[j, 1]
            d2 = dx * dx + dy * dy
            if d2 < best:
                best = d2
                best_j = j
        out[i] = best_j
    return out


def _nearest_neighbor_2d(points: np.ndarray) -> np.ndarray:
    p = np.ascontiguousarray(points, dtype=np.float64)
    return _nearest_neighbor_2d_numba(p)


@njit(cache=False)
def _closest_points_on_segments_2d(a0x: float, a0y: float, a1x: float, a1y: float,
                                   b0x: float, b0y: float, b1x: float, b1y: float) -> Tuple[float, float, float, float, float, float]:
    ux = a1x - a0x
    uy = a1y - a0y
    vx = b1x - b0x
    vy = b1y - b0y
    wx = a0x - b0x
    wy = a0y - b0y

    A = ux * ux + uy * uy
    B = ux * vx + uy * vy
    C = vx * vx + vy * vy
    D = ux * wx + uy * wy
    E = vx * wx + vy * wy

    den = A * C - B * B
    s = 0.0
    t = 0.0
    if den > 0.0:
        s = (B * E - C * D) / den
        t = (A * E - B * D) / den

    # clamp and recompute as needed
    if s < 0.0:
        s = 0.0
        if C > 0.0:
            t = E / C
    elif s > 1.0:
        s = 1.0
        if C > 0.0:
            t = (E + B) / C

    if t < 0.0:
        t = 0.0
        if A > 0.0:
            s = -D / A
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
    elif t > 1.0:
        t = 1.0
        if A > 0.0:
            s = (B - D) / A
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0

    px = a0x + s * ux
    py = a0y + s * uy
    qx = b0x + t * vx
    qy = b0y + t * vy
    return s, t, px, py, qx, qy


def _refine_pairs_on_section(pu: np.ndarray, ps: np.ndarray, pairs: np.ndarray, nn_u: np.ndarray, nn_s: np.ndarray,
                             *, max_seg_len: float = 1e9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine matched pairs using closest points between local segments.

    Returns
    -------
    rstar : (m,2) ndarray refined common points (midpoint of segment closest points)
    u_idx0, u_idx1, s_idx0, s_idx1 : (m,) int arrays endpoints used
    sval, tval : (m,) float arrays interpolation parameters on U and S segments
    valid : (m,) bool mask for pairs where refinement was performed
    """
    m = pairs.shape[0]
    rstar = np.empty((m, 2), dtype=np.float64)
    u0 = np.empty(m, dtype=np.int64)
    u1 = np.empty(m, dtype=np.int64)
    s0 = np.empty(m, dtype=np.int64)
    s1 = np.empty(m, dtype=np.int64)
    sval = np.empty(m, dtype=np.float64)
    tval = np.empty(m, dtype=np.float64)
    valid = np.zeros(m, dtype=np.bool_)

    for k in range(m):
        i = int(pairs[k, 0]); j = int(pairs[k, 1])
        iu = int(nn_u[i]) if nn_u.size else -1
        js = int(nn_s[j]) if nn_s.size else -1
        if iu < 0 or js < 0 or iu == i or js == j:
            # fallback: keep original pairing point
            rstar[k, 0] = pu[i, 0]
            rstar[k, 1] = pu[i, 1]
            u0[k] = i; u1[k] = i
            s0[k] = j; s1[k] = j
            sval[k] = 0.0; tval[k] = 0.0
            valid[k] = False
            continue

        # reject overly long segments
        du = np.hypot(pu[iu, 0] - pu[i, 0], pu[iu, 1] - pu[i, 1])
        ds = np.hypot(ps[js, 0] - ps[j, 0], ps[js, 1] - ps[j, 1])
        if du > max_seg_len or ds > max_seg_len:
            rstar[k, 0] = pu[i, 0]
            rstar[k, 1] = pu[i, 1]
            u0[k] = i; u1[k] = i
            s0[k] = j; s1[k] = j
            sval[k] = 0.0; tval[k] = 0.0
            valid[k] = False
            continue

        s, t, px, py, qx, qy = _closest_points_on_segments_2d(
            pu[i, 0], pu[i, 1], pu[iu, 0], pu[iu, 1], ps[j, 0], ps[j, 1], ps[js, 0], ps[js, 1]
        )

        rstar[k, 0] = 0.5 * (px + qx)
        rstar[k, 1] = 0.5 * (py + qy)
        u0[k] = i; u1[k] = iu
        s0[k] = j; s1[k] = js
        sval[k] = s; tval[k] = t
        valid[k] = True

    return rstar, u0, u1, s0, s1, sval, tval, valid


class _ConnectionsBackend:
    """Encapsulates matching/refinement and ΔV computation for connections."""

    def solve(self, problem):
        # Lazy imports to avoid circulars at module import tim
        # 1) Build section hits on the provided synodic section
        sec_u = problem.source.to_section(problem.section, direction=problem.direction)
        sec_s = problem.target.to_section(problem.section, direction=problem.direction)

        pu = np.asarray(sec_u.points, dtype=float)
        ps = np.asarray(sec_s.points, dtype=float)
        Xu = np.asarray(sec_u.states, dtype=float)
        Xs = np.asarray(sec_s.states, dtype=float)

        if pu.size == 0 or ps.size == 0:
            return []

        # 2) Coarse 2D radius pairing
        eps = float(getattr(problem.search, "eps2d", 1e-4)) if problem.search else 1e-4
        dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
        bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8

        pairs_arr = _radius_pairs_2d(pu, ps, eps)
        if pairs_arr.size == 0:
            return []

        # 3) Mutual-nearest filtering among candidate pairs
        di = pu[pairs_arr[:, 0]] - ps[pairs_arr[:, 1]]
        d2 = np.sum(di * di, axis=1)
        best_for_i = {}
        best_for_j = {}
        for k in range(pairs_arr.shape[0]):
            i = int(pairs_arr[k, 0]); j = int(pairs_arr[k, 1]); val = float(d2[k])
            if (i not in best_for_i) or (val < best_for_i[i][0]):
                best_for_i[i] = (val, j)
            if (j not in best_for_j) or (val < best_for_j[j][0]):
                best_for_j[j] = (val, i)

        pairs: list[tuple[int, int]] = []
        for i, (vi, j) in best_for_i.items():
            vj, ii = best_for_j[j]
            if ii == i and vi == vj:
                pairs.append((i, j))

        if not pairs:
            return []

        # 4) On-section refinement using local segments; interpolate 6D and compute ΔV
        nn_u = _nearest_neighbor_2d(pu) if pu.shape[0] >= 2 else np.full(pu.shape[0], -1, dtype=int)
        nn_s = _nearest_neighbor_2d(ps) if ps.shape[0] >= 2 else np.full(ps.shape[0], -1, dtype=int)
        pairs_np = np.asarray(pairs, dtype=np.int64)
        rstar, u0, u1, s0, s1, sval, tval, valid = _refine_pairs_on_section(pu, ps, pairs_np, nn_u, nn_s)

        results: list[_ConnectionResult] = []
        for k in range(pairs_np.shape[0]):
            i = int(pairs_np[k, 0]); j = int(pairs_np[k, 1])
            if valid[k] and (u0[k] != u1[k]) and (s0[k] != s1[k]):
                Xu_seg = (1.0 - sval[k]) * Xu[u0[k]] + sval[k] * Xu[u1[k]]
                Xs_seg = (1.0 - tval[k]) * Xs[s0[k]] + tval[k] * Xs[s1[k]]
                vu = Xu_seg[3:6]
                vs = Xs_seg[3:6]
                dv = float(np.linalg.norm(vu - vs))
                if dv <= dv_tol:
                    kind = "ballistic" if dv <= bal_tol else "impulsive"
                    pt = (float(rstar[k, 0]), float(rstar[k, 1]))
                    results.append(_ConnectionResult(kind=kind, delta_v=dv, point2d=pt, state_u=Xu_seg.copy(), state_s=Xs_seg.copy(), index_u=int(i), index_s=int(j)))
            else:
                vu = Xu[i, 3:6]
                vs = Xs[j, 3:6]
                dv = float(np.linalg.norm(vu - vs))
                if dv <= dv_tol:
                    kind = "ballistic" if dv <= bal_tol else "impulsive"
                    pt = (float(pu[i, 0]), float(pu[i, 1]))
                    results.append(_ConnectionResult(kind=kind, delta_v=dv, point2d=pt, state_u=Xu[i].copy(), state_s=Xs[j].copy(), index_u=int(i), index_s=int(j)))

        results.sort(key=lambda r: r.delta_v)
        return results