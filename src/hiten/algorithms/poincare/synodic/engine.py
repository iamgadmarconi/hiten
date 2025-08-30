from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                                   _SurfaceEvent)
from hiten.algorithms.poincare.synodic.config import (_SynodicMapConfig,
                                                      _SynodicSectionConfig)


def _project_point(
    state: "np.ndarray",
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
) -> "tuple[float, float]":
    if callable(proj):
        u, v = proj(state)
        return float(u), float(v)
    # axis-name projection using _PlaneEvent mapping for indices
    i = int(_PlaneEvent._IDX_MAP[proj[0].lower()])
    j = int(_PlaneEvent._IDX_MAP[proj[1].lower()])
    return float(state[i]), float(state[j])


def _interp_linear(t0: float, x0: "np.ndarray", t1: float, x1: "np.ndarray", t: float) -> "np.ndarray":
    s = (t - t0) / (t1 - t0)
    return (1.0 - s) * x0 + s * x1


def detect_on_segment(
    t0: float,
    x0: "np.ndarray",
    t1: float,
    x1: "np.ndarray",
    *,
    event: "_SurfaceEvent",
    direction: Literal[1, -1, None],
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    cfg: "_SynodicMapConfig",
) -> "_SectionHit | None":
    g0 = event.value(x0)
    g1 = event.value(x1)

    # On-surface guard
    if abs(g0) < cfg.tol_on_surface:
        pt2 = _project_point(x0, proj)
        return _SectionHit(time=float(t0), state=x0.copy(), point2d=np.array(pt2, dtype=float))

    # Directional sign-change test
    if not event.is_crossing(g0, g1):
        return None

    # Linear refinement
    alpha = g0 / (g0 - g1)
    alpha = min(max(alpha, 0.0), 1.0)
    thit = float((1.0 - alpha) * t0 + alpha * t1)
    xhit = _interp_linear(t0, x0, t1, x1, thit)
    pt2 = _project_point(xhit, proj)
    return _SectionHit(time=thit, state=xhit.astype(float, copy=True), point2d=np.array(pt2, dtype=float))


def detect_on_trajectory(
    times: "np.ndarray",
    states: "np.ndarray",
    *,
    event: "_SurfaceEvent",
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    cfg: "_SynodicMapConfig",
) -> "list[_SectionHit]":
    hits: "list[_SectionHit]" = []

    for k in range(len(times) - 1):
        t0, t1 = float(times[k]), float(times[k + 1])
        x0, x1 = states[k], states[k + 1]

        hit = detect_on_segment(t0, x0, t1, x1, event=event, direction=event.direction, proj=proj, cfg=cfg)
        if hit is None:
            continue

        # Deduplicate against last stored hit
        if hits:
            prev = hits[-1]
            if abs(hit.time - prev.time) <= cfg.dedup_time_tol:
                continue
            du = hit.point2d[0] - prev.point2d[0]
            dv = hit.point2d[1] - prev.point2d[1]
            if (du * du + dv * dv) <= (cfg.dedup_point_tol * cfg.dedup_point_tol):
                continue

        hits.append(hit)

        if cfg.max_hits_per_traj is not None and len(hits) >= cfg.max_hits_per_traj:
            break

    return hits


def detect_batch(
    trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
    *,
    event: "_SurfaceEvent",
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    cfg: "_SynodicMapConfig",
) -> "list[list[_SectionHit]]":
    out: "list[list[_SectionHit]]" = []
    for (times, states) in trajectories:
        out.append(detect_on_trajectory(times, states, event=event, proj=proj, cfg=cfg))
    return out


class _SynodicEngine:
    """Thin orchestrator for synodic section detection on precomputed trajectories.

    This class mirrors the structure of other return-map engines but delegates
    all numerical work to interpolation-based detection functions.
    """

    def __init__(
        self,
        *,
        section_cfg: _SynodicSectionConfig,
        map_cfg: _SynodicMapConfig,
        n_workers: int | None = None,
    ) -> None:
        self._section_cfg = section_cfg
        self._detect_cfg = map_cfg
        self._n_workers = int(n_workers or 1)
        self._cache: _Section | None = None

    def compute_section(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: int | None = None,
        recompute: bool = False,
    ) -> _Section:
        if self._cache is not None and not recompute:
            return self._cache

        evt = self._section_cfg.build_event(direction=direction)
        proj = self._section_cfg.plane_coords

        if self._n_workers <= 1 or len(trajectories) <= 1:
            hits_lists = detect_batch(trajectories, event=evt, proj=proj, cfg=self._detect_cfg)
        else:
            # Parallelize by trajectory chunks
            chunks = np.array_split(np.arange(len(trajectories)), self._n_workers)

            def _worker(idx_arr: np.ndarray):
                subset = [trajectories[i] for i in idx_arr.tolist()]
                return detect_batch(subset, event=evt, proj=proj, cfg=self._detect_cfg)

            parts: list[list[list]] = []
            with ThreadPoolExecutor(max_workers=self._n_workers) as ex:
                futs = [ex.submit(_worker, idxs) for idxs in chunks if len(idxs)]
                for fut in as_completed(futs):
                    parts.append(fut.result())
            # Flatten preserving input order approximately
            hits_lists = [hits for part in parts for hits in part]

        # Assemble points and states stacked across all trajectories
        pts, sts = [], []
        for hits in hits_lists:
            for h in hits:
                pts.append(h.point2d)
                sts.append(h.state)

        pts_np = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
        sts_np = np.asarray(sts, dtype=float) if sts else np.empty((0, 6))

        self._cache = _Section(pts_np, sts_np, self._section_cfg.plane_coords)
        return self._cache
