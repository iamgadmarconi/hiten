from dataclasses import dataclass
from typing import List

import numpy as np

from hiten.algorithms.connections.section.base import _SectionAdapter


@dataclass
class _CandidatePair:
    idx_source: int
    idx_target: int
    distance2d: float


def _grid_indices(n: int, size: int) -> np.ndarray:
    n = max(1, int(n))
    if size <= 0:
        return np.empty((0,), dtype=int)
    # Uniformly spaced indices over [0, size-1]
    return np.unique(np.linspace(0, size - 1, num=min(n, size), dtype=int))


def generate_grid_pairs(
    src: _SectionAdapter,
    tgt: _SectionAdapter,
    *,
    grid_source: int,
    grid_target: int,
    radius: float,
    budget: int,
) -> List[_CandidatePair]:
    """Pair subsampled section points by 2D proximity.

    Returns up to 'budget' closest pairs within 'radius'.
    """
    ps = src.points2d()
    pt = tgt.points2d()
    if ps.size == 0 or pt.size == 0:
        return []

    isel = _grid_indices(grid_source, ps.shape[0])
    jsel = _grid_indices(grid_target, pt.shape[0])
    if isel.size == 0 or jsel.size == 0:
        return []

    cand: List[_CandidatePair] = []
    for i in isel.tolist():
        p = ps[i]
        d = pt[jsel] - p  # broadcast
        d2 = np.sum(d * d, axis=1)
        mask = d2 <= (radius * radius)
        if np.any(mask):
            for j, val in zip(jsel[mask].tolist(), d2[mask].tolist()):
                cand.append(_CandidatePair(i, j, float(val)))

    cand.sort(key=lambda c: c.distance2d)
    if budget > 0:
        cand = cand[: int(budget)]
    return cand


def generate_segment_pairs(
    src: _SectionAdapter,
    tgt: _SectionAdapter,
    *,
    step_src: int = 1,
    step_tgt: int = 1,
    radius: float = 1e-3,
    budget: int = 1000,
) -> List[_CandidatePair]:
    """Segment-segment proximity test on 2D section to capture crossings between samples.

    Approximates each consecutive pair as a segment and finds endpoints within 'radius'.
    This is a simple heuristic; a proper segment intersection can be added later.
    """
    ps = src.points2d()
    pt = tgt.points2d()
    out: List[_CandidatePair] = []
    if ps.shape[0] < 2 or pt.shape[0] < 2:
        return out

    def _nearest_endpoint(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> tuple[int, float]:
        # Check distances from both endpoints of q to segment p endpoints; pick smallest index
        d00 = float(np.sum((p0 - q0) ** 2))
        d01 = float(np.sum((p0 - q1) ** 2))
        d10 = float(np.sum((p1 - q0) ** 2))
        d11 = float(np.sum((p1 - q1) ** 2))
        best = min((0, d00), (1, d01), (0, d10), (1, d11), key=lambda x: x[1])
        return best[0], best[1]

    rsq = radius * radius
    for i in range(0, ps.shape[0] - 1, max(1, int(step_src))):
        p0, p1 = ps[i], ps[i + 1]
        for j in range(0, pt.shape[0] - 1, max(1, int(step_tgt))):
            q0, q1 = pt[j], pt[j + 1]
            end_idx, d2 = _nearest_endpoint(p0, p1, q0, q1)
            if d2 <= rsq:
                # choose the endpoint on target segment
                jj = j if end_idx == 0 else j + 1
                out.append(_CandidatePair(i, jj, float(d2)))
                if len(out) >= budget:
                    return out
    out.sort(key=lambda c: c.distance2d)
    return out


