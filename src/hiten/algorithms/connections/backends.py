"""Provide backend routines for discovering connections between synodic sections in CR3BP.

This module provides the computational backend for the connections algorithm,
which discovers ballistic and impulsive transfers between synodic sections in
the Circular Restricted Three-Body Problem (CR3BP).

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.connections.types`
    Result classes for connection data.
:mod:`~hiten.algorithms.connections.engine`
    High-level connection engine interface.
"""

from typing import Tuple

import numba
import numpy as np

from hiten.algorithms.connections.types import (
    ConnectionsBackendRequest,
    ConnectionsBackendResponse,
    _ConnectionResult,
)
from hiten.algorithms.types.core import _HitenBaseBackend


@numba.njit(cache=False)
def _pair_counts(query: np.ndarray, ref: np.ndarray, r2: float) -> np.ndarray:
    """Return for each query point the number of reference points within radius^2.

    Parameters
    ----------
    query : np.ndarray, shape (N, 2)
        2D coordinates of query points.
    ref : np.ndarray, shape (M, 2)
        2D coordinates of reference points.
    r2 : float
        Radius squared for distance comparison.

    Returns
    -------
    np.ndarray, shape (N,)
        For each query point, the count of reference points with distance^2 <= r2.

    Notes
    -----
    Used by :func:`~hiten.algorithms.connections.backends._radpair2d` to 
    efficiently allocate storage for pairs.
    """
    
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


@numba.njit(cache=False)
def _exclusive_prefix_sum(a: np.ndarray) -> np.ndarray:
    """Compute exclusive prefix sum of an integer array.

    Parameters
    ----------
    a : np.ndarray
        Input integer array of length N.

    Returns
    -------
    np.ndarray, shape (N+1,)
        Exclusive prefix sums where out[0] = 0 and out[i+1] = sum_{k=0}^{i} a[k].

    Notes
    -----
    Used by :func:`~hiten.algorithms.connections.backends._radpair2d` 
    to determine memory offsets for storing pairs.
    """
    n = a.size
    out = np.empty(n + 1, dtype=np.int64)
    out[0] = 0
    s = 0
    for i in range(n):
        s += int(a[i])
        out[i + 1] = s
    return out


@numba.njit(cache=False)
def _radpair2d(query: np.ndarray, ref: np.ndarray, radius: float) -> np.ndarray:
    """Find all pairs (i,j) where distance(query[i], ref[j]) <= radius in 2D.

    Parameters
    ----------
    query : np.ndarray, shape (N, 2)
        Query points in 2D.
    ref : np.ndarray, shape (M, 2)
        Reference points in 2D.
    radius : float
        Matching radius in the same units as query/ref coordinates.

    Returns
    -------
    np.ndarray, shape (total, 2)
        Each row is a pair (i, j) indicating a match between query[i] and ref[j].

    Notes
    -----
    Uses :func:`~hiten.algorithms.connections.backends._pair_counts` and 
    :func:`~hiten.algorithms.connections.backends._exclusive_prefix_sum` 
    to efficiently allocate and populate the output array.
    """
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
    query : ndarray, shape (N, 2)
        2D plane coordinates of query points.
    ref : ndarray, shape (M, 2)
        2D plane coordinates of reference points.
    radius : float
        Match radius in nondimensional CR3BP units.

    Returns
    -------
    ndarray, shape (total, 2)
        Each row is a pair (i, j) indicating a match between query[i] and ref[j].

    Notes
    -----
    This is the main entry point for 2D radius-based pairing. It prepares
    contiguous arrays and delegates to the numba-accelerated 
    :func:`~hiten.algorithms.connections.backends._radpair2d`.
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(ref, dtype=np.float64)
    return _radpair2d(q, r, float(radius))


@numba.njit(cache=False)
def _nearest_neighbor_2d_numba(points: np.ndarray) -> np.ndarray:
    """Find the nearest neighbor for each point in a 2D array (numba-accelerated).

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        2D coordinates of points.

    Returns
    -------
    np.ndarray, shape (N,)
        For each point i, the index j of its nearest neighbor (j != i).
        Returns -1 if no valid neighbor exists.

    Notes
    -----
    This is the numba-accelerated implementation used by 
    :func:`~hiten.algorithms.connections.backends._nearest_neighbor_2d`.
    """
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
    """Find the nearest neighbor for each point in a 2D array.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        2D coordinates of points.

    Returns
    -------
    np.ndarray, shape (N,)
        For each point i, the index j of its nearest neighbor (j != i).
        Returns -1 if no valid neighbor exists.

    Notes
    -----
    This function prepares data and delegates to the numba-accelerated
    :func:`~hiten.algorithms.connections.backends._nearest_neighbor_2d_numba`.
    """
    p = np.ascontiguousarray(points, dtype=np.float64)
    return _nearest_neighbor_2d_numba(p)


@numba.njit(cache=False)
def _closest_points_on_segments_2d(a0x: float, a0y: float, a1x: float, a1y: float,
                                   b0x: float, b0y: float, b1x: float, b1y: float) -> Tuple[float, float, float, float, float, float]:
    """Find the closest points between two 2D line segments.

    Parameters
    ----------
    a0x, a0y : float
        Start point of first segment.
    a1x, a1y : float
        End point of first segment.
    b0x, b0y : float
        Start point of second segment.
    b1x, b1y : float
        End point of second segment.

    Returns
    -------
    s : float
        Parameter along first segment (0 <= s <= 1).
    t : float
        Parameter along second segment (0 <= t <= 1).
    px, py : float
        Closest point on first segment.
    qx, qy : float
        Closest point on second segment.

    Notes
    -----
    Used by :func:`~hiten.algorithms.connections.backends._refine_pairs_on_section` 
    for geometric refinement of matched pairs between synodic sections.
    """
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


@numba.njit(cache=False)
def _refine_pairs_on_section(pu: np.ndarray, ps: np.ndarray, pairs: np.ndarray, nn_u: np.ndarray, nn_s: np.ndarray, max_seg_len: float = 1e9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine matched pairs using closest points between local segments.

    Parameters
    ----------
    pu : np.ndarray, shape (N, 2)
        2D points on the unstable (source) section.
    ps : np.ndarray, shape (M, 2)
        2D points on the stable (target) section.
    pairs : np.ndarray, shape (k, 2)
        Initial matched pairs as (i, j) indices.
    nn_u : np.ndarray, shape (N,)
        Nearest neighbor indices for unstable section points.
    nn_s : np.ndarray, shape (M,)
        Nearest neighbor indices for stable section points.
    max_seg_len : float, optional
        Maximum allowed segment length for refinement (default: 1e9).

    Returns
    -------
    rstar : np.ndarray, shape (k, 2)
        Refined common points (midpoint of segment closest points).
    u_idx0, u_idx1 : np.ndarray, shape (k,)
        Endpoint indices used on the unstable section.
    s_idx0, s_idx1 : np.ndarray, shape (k,)
        Endpoint indices used on the stable section.
    sval, tval : ndarray, shape (k,)
        Interpolation parameters on U and S segments.
    valid : ndarray, shape (k,)
        Boolean mask indicating pairs where refinement was performed.

    Notes
    -----
    Uses :func:`~hiten.algorithms.connections.backends._closest_points_on_segments_2d` 
    to find optimal intersection points between local segments formed by 
    nearest neighbors.
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


class _ConnectionsBackend(_HitenBaseBackend):
    """Encapsulate matching/refinement and Delta-V computation for connections.

    This backend orchestrates the end-to-end process for discovering
    ballistic/impulsive transfers between two synodic sections within the
    CR3BP. It builds coarse hits, applies radius-based pairing and mutual-nearest
    filtering, refines matched pairs using local segment geometry, and finally
    computes the Delta-V required for each candidate transfer.

    See Also
    --------
    :class:`~hiten.algorithms.connections.types._ConnectionResult`
        Result objects returned by the solve method.
    """

    def run(
        self,
        request: ConnectionsBackendRequest,
    ) -> ConnectionsBackendResponse:
        """Compute possible connections from precomputed section data.

        Parameters
        ----------
        pu : np.ndarray, shape (N, 2)
            2D points on the unstable/source section.
        ps : np.ndarray, shape (M, 2)
            2D points on the stable/target section.
        Xu : np.ndarray, shape (N, 6)
            6D states corresponding to ``pu``.
        Xs : ndarray, shape (M, 6)
            6D states corresponding to ``ps``.
        traj_indices_u : np.ndarray or None, shape (N,)
            Trajectory indices for source manifold intersections.
        traj_indices_s : ndarray or None, shape (M,)
            Trajectory indices for target manifold intersections.
        eps : float
            Pairing radius on the 2D section plane.
        dv_tol : float
            Maximum allowed Delta-V for accepting a connection.
        bal_tol : float
            Threshold for classifying a connection as ballistic.

        Returns
        -------
        list of :class:`~hiten.algorithms.connections.types._ConnectionResult`
            ConnectionPipeline results sorted by increasing delta_v (velocity change).

        Notes
        -----
        Steps:
        1. Coarse 2D radius pairing
        2. Mutual-nearest filtering
        3. Segment-based refinement
        4. Delta-V computation and classification
        """
        pu = request.points_u
        ps = request.points_s
        Xu = request.states_u
        Xs = request.states_s
        traj_indices_u = request.traj_indices_u
        traj_indices_s = request.traj_indices_s
        eps = request.eps
        dv_tol = request.dv_tol
        bal_tol = request.bal_tol

        if pu.size == 0 or ps.size == 0:
            return ConnectionsBackendResponse(results=[], metadata={})

        pairs_arr = _radius_pairs_2d(pu, ps, float(eps))
        if pairs_arr.size == 0:
            return ConnectionsBackendResponse(results=[], metadata={})

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
            return ConnectionsBackendResponse(results=[], metadata={})

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
                    traj_idx_u = int(traj_indices_u[i]) if traj_indices_u is not None else 0
                    traj_idx_s = int(traj_indices_s[j]) if traj_indices_s is not None else 0
                    results.append(_ConnectionResult(kind=kind, delta_v=dv, point2d=pt, state_u=Xu_seg.copy(), state_s=Xs_seg.copy(), index_u=int(i), index_s=int(j), trajectory_index_u=traj_idx_u, trajectory_index_s=traj_idx_s))
            else:
                vu = Xu[i, 3:6]
                vs = Xs[j, 3:6]
                dv = float(np.linalg.norm(vu - vs))
                if dv <= dv_tol:
                    kind = "ballistic" if dv <= bal_tol else "impulsive"
                    pt = (float(pu[i, 0]), float(pu[i, 1]))
                    traj_idx_u = int(traj_indices_u[i]) if traj_indices_u is not None else 0
                    traj_idx_s = int(traj_indices_s[j]) if traj_indices_s is not None else 0
                    results.append(_ConnectionResult(kind=kind, delta_v=dv, point2d=pt, state_u=Xu[i].copy(), state_s=Xs[j].copy(), index_u=int(i), index_s=int(j), trajectory_index_u=traj_idx_u, trajectory_index_s=traj_idx_s))

        results.sort(key=lambda r: r.delta_v)
        metadata = dict(request.metadata)
        metadata.update({
            "pairs_considered": pairs_arr.shape[0],
            "accepted": len(results),
        })
        return ConnectionsBackendResponse(results=results, metadata=metadata)

    def on_start(self, problem) -> None:  # Engine notifies before solving
        """Called by the engine before solving."""
        pass

    def on_success(self, results: list[_ConnectionResult]) -> None:  # Engine notifies after successful solve
        """Called by the engine after successful solve."""
        pass

    def on_failure(self, error: Exception) -> None:  # Engine notifies on failure
        """Called by the engine on failure."""
        pass
