from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                                   _SurfaceEvent)
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.core.strategies import _SeedingStrategyBase
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


class _NoOpBackend(_ReturnMapBackend):
    """Placeholder backend; never used by the synodic engine."""

    def __init__(self) -> None:
        # Pass inert placeholders; engine will not use backend.
        class _NullSurface(_SurfaceEvent):
            def value(self, state: np.ndarray) -> float:  # type: ignore[override]
                return 0.0

        super().__init__(dynsys=None, surface=_NullSurface())  # type: ignore[arg-type]

    def step_to_section(self, seeds: "np.ndarray", *, dt: float = 1e-2) -> tuple["np.ndarray", "np.ndarray"]:  # type: ignore[override]
        raise NotImplementedError("Synodic engine does not propagate seeds")


class _NoOpStrategy(_SeedingStrategyBase):
    def generate(self, *, h0, H_blocks, clmo_table, solve_missing_coord_fn, find_turning_fn):  # type: ignore[override]
        raise NotImplementedError("Synodic engine does not generate seeds")


class _SynodicEngineConfigAdapter:
    """Adapter providing dt/n_iter expected by the base engine."""

    def __init__(self, cfg: _SynodicMapConfig) -> None:
        self._cfg = cfg
        self.dt = 0.0
        self.n_iter = 1
        self.n_workers = cfg.n_workers
        # Satisfy _SeedingConfigLike for the no-op strategy
        self.n_seeds = 0

    def __repr__(self) -> str:
        return f"SynodicEngineConfigAdapter(n_workers={self.n_workers})"


class _SynodicEngine(_ReturnMapEngine):
    """Engine for synodic section detection on precomputed trajectories.

    Subclasses the generic engine to reuse worker/count plumbing and caching.
    """

    def __init__(
        self,
        *,
        backend: _ReturnMapBackend,
        seed_strategy: _SeedingStrategyBase,
        map_config: _SynodicEngineConfigAdapter,
        section_cfg: _SynodicSectionConfig,
    ) -> None:
        super().__init__(backend=backend, seed_strategy=seed_strategy, map_config=map_config)
        self._section_cfg = section_cfg
        self._detect_cfg: _SynodicMapConfig = map_config._cfg
        # Runtime state
        self._trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]" | None = None
        self._direction: int | None = None

    def set_trajectories(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: Literal[1, -1, None] | None = None,
    ) -> "_SynodicEngine":
        self._trajectories = trajectories
        self._direction = direction
        self.clear_cache()
        return self

    def compute_section(self, *, recompute: bool = False) -> _Section:  # type: ignore[override]
        if self._section_cache is not None and not recompute:
            return self._section_cache

        if self._trajectories is None:
            raise ValueError("No trajectories set. Call set_trajectories(...) first.")

        evt = self._section_cfg.build_event(direction=self._direction)
        proj = self._section_cfg.plane_coords

        if self._n_workers <= 1 or len(self._trajectories) <= 1:
            hits_lists = detect_batch(self._trajectories, event=evt, proj=proj, cfg=self._detect_cfg)
        else:
            chunks = np.array_split(np.arange(len(self._trajectories)), self._n_workers)

            def _worker(idx_arr: np.ndarray):
                subset = [self._trajectories[i] for i in idx_arr.tolist()]  # type: ignore[index]
                return detect_batch(subset, event=evt, proj=proj, cfg=self._detect_cfg)

            parts: list[list[list]] = []
            with ThreadPoolExecutor(max_workers=self._n_workers) as ex:
                futs = [ex.submit(_worker, idxs) for idxs in chunks if len(idxs)]
                for fut in as_completed(futs):
                    parts.append(fut.result())
            hits_lists = [hits for part in parts for hits in part]

        pts, sts = [], []
        for hits in hits_lists:
            for h in hits:
                pts.append(h.point2d)
                sts.append(h.state)

        pts_np = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
        sts_np = np.asarray(sts, dtype=float) if sts else np.empty((0, 6))

        self._section_cache = _Section(pts_np, sts_np, self._section_cfg.plane_coords)
        return self._section_cache
