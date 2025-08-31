from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                                   _SurfaceEvent)
from hiten.algorithms.poincare.synodic.config import (_SynodicMapConfig,
                                                      _SynodicSectionConfig)
from hiten.algorithms.poincare.utils import _hermite_der, _hermite_scalar


@dataclass
class _DetectionSettings:
    """Cached numerical/settings knobs to avoid repeated getattr lookups.

    These values are derived once from the map/section configuration and reused
    during detection/refinement routines.
    """

    use_cubic: bool
    segment_refine: int
    tol_on_surface: float
    dedup_time_tol: float
    dedup_point_tol: float
    max_hits_per_traj: int | None
    proj: "tuple[str, str]"
    newton_max_iter: int

    @classmethod
    def from_config(
        cls,
        *,
        map_cfg: "_SynodicMapConfig",
        plane_coords: "tuple[str, str]",
    ) -> "_DetectionSettings":
        return cls(
            use_cubic=(getattr(map_cfg, "interp_kind", "linear") == "cubic"),
            segment_refine=int(getattr(map_cfg, "segment_refine", 0)),
            tol_on_surface=float(getattr(map_cfg, "tol_on_surface", 1e-12)),
            dedup_time_tol=float(getattr(map_cfg, "dedup_time_tol", 1e-9)),
            dedup_point_tol=float(getattr(map_cfg, "dedup_point_tol", 1e-12)),
            max_hits_per_traj=getattr(map_cfg, "max_hits_per_traj", None),
            proj=(str(plane_coords[0]), str(plane_coords[1])),
            newton_max_iter=int(getattr(map_cfg, "newton_max_iter", 4)),
        )


def _project_batch(
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    x: "np.ndarray",
) -> "np.ndarray":
    if callable(proj):
        out_list = [tuple(map(float, proj(row))) for row in x]
        return np.asarray(out_list, dtype=float)
    i = int(_PlaneEvent._IDX_MAP[proj[0].lower()])
    j = int(_PlaneEvent._IDX_MAP[proj[1].lower()])
    return np.column_stack((x[:, i], x[:, j])).astype(float, copy=False)


def _compute_event_values(event: "_SurfaceEvent", states: "np.ndarray") -> "np.ndarray":
    ok, n_vec, c_off = _is_vectorizable_plane_event(event, states.shape[1])
    if ok and n_vec is not None and c_off is not None:
        return states @ n_vec.astype(float, copy=False) - float(c_off)
    return np.fromiter((float(event.value(states[k])) for k in range(states.shape[0])), dtype=float, count=states.shape[0])


def _is_vectorizable_plane_event(event: "_SurfaceEvent", n_cols: int) -> "tuple[bool, np.ndarray | None, float | None]":
    """Return True and cached (normal, offset) if event supports vectorised evaluation.

    We can vectorise when the event exposes a 1-D normal vector with length
    matching the state dimension, and a scalar offset.
    """
    n_vec = getattr(event, "normal", None)
    c_off = getattr(event, "offset", None)
    ok = (
        isinstance(n_vec, np.ndarray)
        and np.ndim(n_vec) == 1
        and n_vec.size == int(n_cols)
        and isinstance(c_off, (float, int))
    )
    if ok:
        return True, n_vec, float(c_off)
    return False, None, None


def _on_surface_indices(g_all: "np.ndarray", tol: float, direction: "Literal[1, -1, None]") -> "np.ndarray":
    g0 = g_all[:-1]
    g1 = g_all[1:]
    base = np.abs(g0) < tol
    if direction is None:
        return np.nonzero(base)[0]
    idxs = np.nonzero(base)[0]
    keep = []
    if direction == 1:
        for k in idxs:
            cond_next = g1[k] >= 0.0
            cond_prev = (k - 1 >= 0) and (g_all[k - 1] <= 0.0)
            keep.append(cond_next or cond_prev)
    else:
        for k in idxs:
            cond_next = g1[k] <= 0.0
            cond_prev = (k - 1 >= 0) and (g_all[k - 1] >= 0.0)
            keep.append(cond_next or cond_prev)
    mask = np.zeros_like(base)
    mask[idxs] = np.asarray(keep, dtype=bool)
    return np.nonzero(base & mask)[0]


def _crossing_indices_and_alpha(g0: "np.ndarray", g1: "np.ndarray", *, on_mask: "np.ndarray", direction: "Literal[1, -1, None]") -> tuple["np.ndarray", "np.ndarray"]:
    if direction is None:
        cross_mask = (g0 * g1 <= 0.0) & (g0 != g1)
    elif direction == 1:
        cross_mask = (g0 < 0.0) & (g1 >= 0.0)
    else:
        cross_mask = (g0 > 0.0) & (g1 <= 0.0)
    cross_mask &= ~on_mask
    cr_idx = np.nonzero(cross_mask)[0]
    if cr_idx.size:
        g0_sel = g0[cr_idx]
        g1_sel = g1[cr_idx]
        alpha = g0_sel / (g0_sel - g1_sel)
        alpha = np.minimum(1.0, np.maximum(0.0, alpha))
    else:
        alpha = np.empty((0,), dtype=float)
    return cr_idx, alpha


def _refine_hits_linear(t0: "np.ndarray", t1: "np.ndarray", x0: "np.ndarray", x1: "np.ndarray", cr_idx: "np.ndarray", alpha: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    if cr_idx.size == 0:
        return np.empty((0,), dtype=float), np.empty((0, x0.shape[1]), dtype=float)
    thit = (1.0 - alpha) * t0[cr_idx] + alpha * t1[cr_idx]
    xhit = x0[cr_idx] + alpha[:, None] * (x1[cr_idx] - x0[cr_idx])
    return thit.astype(float, copy=False), xhit.astype(float, copy=False)


def _refine_hits_cubic(
    times: "np.ndarray",
    states: "np.ndarray",
    g_all: "np.ndarray",
    cr_idx: "np.ndarray",
    alpha: "np.ndarray",
    *,
    max_iter: int,
) -> tuple["np.ndarray", "np.ndarray"]:
    N = times.shape[0]
    if cr_idx.size == 0:
        return np.empty((0,), dtype=float), np.empty((0, states.shape[1]), dtype=float)

    th_list: list[float] = []
    xh_list: list[np.ndarray] = []

    for pos, k in enumerate(cr_idx.tolist()):
        s_lin = float(alpha[pos])
        dt_seg = float(times[k + 1] - times[k])
        s_star = s_lin

        # Estimate g-derivatives (central where possible)
        if dt_seg > 0.0:
            if (k - 1) >= 0:
                d0 = (g_all[k + 1] - g_all[k - 1]) / (times[k + 1] - times[k - 1])
            else:
                d0 = (g_all[k + 1] - g_all[k]) / (times[k + 1] - times[k])
            if (k + 2) < N:
                d1 = (g_all[k + 2] - g_all[k]) / (times[k + 2] - times[k])
            else:
                d1 = (g_all[k + 1] - g_all[k]) / (times[k + 1] - times[k])

            # Newton refinement on s in [0,1]
            for _ in range(max_iter):
                f = _hermite_scalar(s_star, float(g_all[k]), float(g_all[k + 1]), float(d0), float(d1), dt_seg)
                df = _hermite_der(s_star, float(g_all[k]), float(g_all[k + 1]), float(d0), float(d1), dt_seg)
                if df == 0.0:
                    break
                s_star -= f / df
                if s_star < 0.0:
                    s_star = 0.0
                    break
                if s_star > 1.0:
                    s_star = 1.0
                    break

        th = (1.0 - s_star) * times[k] + s_star * times[k + 1]

        # State interpolation: cubic Hermite when neighbor points exist
        if dt_seg > 0.0 and (k - 1) >= 0 and (k + 2) < N:
            dxdt0 = (states[k + 1] - states[k - 1]) / (times[k + 1] - times[k - 1])
            dxdt1 = (states[k + 2] - states[k]) / (times[k + 2] - times[k])
            s = s_star
            h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
            h10 = s * (1.0 - s) ** 2
            h01 = s ** 2 * (3.0 - 2.0 * s)
            h11 = s ** 2 * (s - 1.0)
            xh = (
                h00 * states[k]
                + h10 * dxdt0 * dt_seg
                + h01 * states[k + 1]
                + h11 * dxdt1 * dt_seg
            )
        else:
            xh = states[k] + s_star * (states[k + 1] - states[k])

        th_list.append(float(th))
        xh_list.append(xh.astype(float, copy=True))

    return np.asarray(th_list, dtype=float), np.asarray(xh_list, dtype=float)


def _order_and_dedup_hits(
    cand_times: "list[float]",
    cand_states: "list[np.ndarray]",
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    seg_order: "np.ndarray",
    dedup_time_tol: float,
    dedup_point_tol: float,
    max_hits_per_traj: int | None,
) -> "list[_SectionHit]":
    if not cand_times:
        return []
    cand_states_np = np.asarray(cand_states, dtype=float)
    cand_pts_np = _project_batch(proj, cand_states_np)
    order = np.argsort(seg_order, kind="stable") if seg_order.size else np.arange(0)
    cand_times_np = np.asarray(cand_times, dtype=float)[order]
    cand_states_np = cand_states_np[order]
    cand_pts_np = cand_pts_np[order]

    hits: "list[_SectionHit]" = []
    for k in range(cand_times_np.shape[0]):
        th = float(cand_times_np[k])
        st = cand_states_np[k]
        pt = cand_pts_np[k]
        if hits:
            prev = hits[-1]
            if abs(th - prev.time) <= dedup_time_tol:
                continue
            du = pt[0] - prev.point2d[0]
            dv = pt[1] - prev.point2d[1]
            if (du * du + dv * dv) <= (dedup_point_tol * dedup_point_tol):
                continue
        hits.append(_SectionHit(time=th, state=st, point2d=pt))
        if max_hits_per_traj is not None and len(hits) >= max_hits_per_traj:
            break
    return hits


def _detect_with_segment_refine(
    times: "np.ndarray",
    states: "np.ndarray",
    g_all: "np.ndarray",
    *,
    event: "_SurfaceEvent",
    proj: "tuple[str, str] | Callable[[np.ndarray], tuple[float, float]]",
    settings: "_DetectionSettings",
) -> "list[_SectionHit]":
    N = times.shape[0]
    r = int(settings.segment_refine)
    if r <= 0 or N < 2:
        return []

    step = 1.0 / (r + 1)
    use_cubic = settings.use_cubic

    cand_times: list[float] = []
    cand_states: list[np.ndarray] = []
    seg_order_list: list[int] = []

    for k in range(N - 1):
        t0 = float(times[k])
        t1 = float(times[k + 1])
        dt = t1 - t0
        gk = float(g_all[k])
        gk1 = float(g_all[k + 1])

        # Optional cubic slopes for g
        if use_cubic and dt > 0.0:
            if (k - 1) >= 0:
                d0 = (g_all[k + 1] - g_all[k - 1]) / (times[k + 1] - times[k - 1])
            else:
                d0 = (g_all[k + 1] - g_all[k]) / (times[k + 1] - times[k])
            if (k + 2) < N:
                d1 = (g_all[k + 2] - g_all[k]) / (times[k + 2] - times[k])
            else:
                d1 = (g_all[k + 1] - g_all[k]) / (times[k + 1] - times[k])
            d0 = float(d0)
            d1 = float(d1)

        # Direction-aware on-surface at s=0
        accept_left = False
        if abs(gk) < settings.tol_on_surface:
            if event.direction is None:
                accept_left = True
            elif event.direction == 1:
                cond_next = (gk1 >= 0.0)
                cond_prev = (k - 1 >= 0) and (g_all[k - 1] <= 0.0)
                accept_left = cond_next or cond_prev
            else:
                cond_next = (gk1 <= 0.0)
                cond_prev = (k - 1 >= 0) and (g_all[k - 1] >= 0.0)
                accept_left = cond_next or cond_prev
            if accept_left:
                th = t0
                xh = states[k].astype(float, copy=True)
                cand_times.append(float(th))
                cand_states.append(xh)
                seg_order_list.append(k)

        # Iterate subintervals
        for m in range(r + 1):
            s_lo = m * step
            s_hi = (m + 1) * step
            if s_hi > 1.0 + 1e-15:
                break
            if accept_left and m == 0:
                continue

            # Evaluate g at s_lo and s_hi
            if use_cubic and dt > 0.0:
                g_lo = _hermite_scalar(s_lo, gk, gk1, d0, d1, dt)
                g_hi = _hermite_scalar(s_hi, gk, gk1, d0, d1, dt)
            else:
                g_lo = (1.0 - s_lo) * gk + s_lo * gk1
                g_hi = (1.0 - s_hi) * gk + s_hi * gk1

            # Directional crossing test
            if event.direction is None:
                crosses = (g_lo * g_hi <= 0.0) and (g_lo != g_hi)
            elif event.direction == 1:
                crosses = (g_lo < 0.0) and (g_hi >= 0.0)
            else:
                crosses = (g_lo > 0.0) and (g_hi <= 0.0)
            if not crosses:
                continue

            # Linear interpolation within the subsegment to locate root
            if g_lo == g_hi:
                s_star = 0.5 * (s_lo + s_hi)
            else:
                alpha_local = g_lo / (g_lo - g_hi)
                alpha_local = min(1.0, max(0.0, alpha_local))
                s_star = s_lo + alpha_local * (s_hi - s_lo)

            # Optional Newton refinement on the full base segment (cubic g)
            if use_cubic and dt > 0.0:
                for _ in range(settings.newton_max_iter):
                    f = _hermite_scalar(s_star, gk, gk1, d0, d1, dt)
                    df = _hermite_der(s_star, gk, gk1, d0, d1, dt)
                    if df == 0.0:
                        break
                    s_star -= f / df
                    if s_star < s_lo:
                        s_star = s_lo
                        break
                    if s_star > s_hi:
                        s_star = s_hi
                        break

            # Hit time
            th = (1.0 - s_star) * t0 + s_star * t1

            # Hit state on the base segment at s_star (cubic state if neighbors available)
            if use_cubic and dt > 0.0 and (k - 1) >= 0 and (k + 2) < N:
                dxdt0 = (states[k + 1] - states[k - 1]) / (times[k + 1] - times[k - 1])
                dxdt1 = (states[k + 2] - states[k]) / (times[k + 2] - times[k])
                s = s_star
                h00 = (1.0 + 2.0 * s) * (1.0 - s) ** 2
                h10 = s * (1.0 - s) ** 2
                h01 = s ** 2 * (3.0 - 2.0 * s)
                h11 = s ** 2 * (s - 1.0)
                xh = (
                    h00 * states[k]
                    + h10 * dxdt0 * dt
                    + h01 * states[k + 1]
                    + h11 * dxdt1 * dt
                )
            else:
                xh = states[k] + s_star * (states[k + 1] - states[k])

            cand_times.append(float(th))
            cand_states.append(xh.astype(float, copy=True))
            seg_order_list.append(k)

    seg_order = np.asarray(seg_order_list, dtype=int) if seg_order_list else np.empty((0,), dtype=int)
    return _order_and_dedup_hits(
        cand_times,
        cand_states,
        proj,
        seg_order,
        settings.dedup_time_tol,
        settings.dedup_point_tol,
        settings.max_hits_per_traj,
    )


class _SynodicDetectionBackend(_ReturnMapBackend):
    """Backend that encapsulates detection/refinement on precomputed trajectories.

    Unlike propagating backends, this backend does not integrate forward from
    seeds. It performs section detection on (time, state) samples supplied by
    the engine and returns crossings with refined hit states and 2-D projections.
    """

    def __init__(self, *, section_cfg: _SynodicSectionConfig, map_cfg: _SynodicMapConfig) -> None:
        # The stored surface is inert; we build an event per call to include direction.
        super().__init__(dynsys=None, surface=section_cfg.build_event(direction=None))
        self._section_cfg = section_cfg
        self._cfg = map_cfg
        # Cache frequently used numeric settings and projection axes
        self._settings = _DetectionSettings.from_config(map_cfg=map_cfg, plane_coords=section_cfg.plane_coords)

    # Required by abstract base, but unused for synodic
    def step_to_section(self, seeds: "np.ndarray", *, dt: float = 1e-2) -> tuple["np.ndarray", "np.ndarray"]:
        raise NotImplementedError("Synodic detection backend does not propagate seeds")

    def detect_on_trajectory(self, times: np.ndarray, states: np.ndarray, *, direction: Literal[1, -1, None] | None = None) -> "list[_SectionHit]":
        if times.size < 2:
            return []
        event = self._section_cfg.build_event(direction=direction)
        proj = self._settings.proj
        settings = self._settings

        g_all = _compute_event_values(event, states)

        # Segment refinement path (optional)
        if int(settings.segment_refine) > 0:
            return _detect_with_segment_refine(times, states, g_all, event=event, proj=proj, settings=settings)

        g0 = g_all[:-1]
        g1 = g_all[1:]
        t0 = times[:-1].astype(float, copy=False)
        t1 = times[1:].astype(float, copy=False)
        x0 = states[:-1]
        x1 = states[1:]

        on_idx = _on_surface_indices(g_all, settings.tol_on_surface, event.direction)
        on_mask = np.zeros_like(g0, dtype=bool)
        on_mask[on_idx] = True

        cand_times: list[float] = []
        cand_states: list[np.ndarray] = []
        if on_idx.size:
            cand_times.extend(t0[on_idx].tolist())
            cand_states.extend([row.copy() for row in x0[on_idx]])

        cr_idx, alpha = _crossing_indices_and_alpha(g0, g1, on_mask=on_mask, direction=event.direction)
        if cr_idx.size:
            use_cubic = settings.use_cubic
            if use_cubic:
                thit, xhit = _refine_hits_cubic(times, states, g_all, cr_idx, alpha, max_iter=settings.newton_max_iter)
            else:
                thit, xhit = _refine_hits_linear(t0, t1, x0, x1, cr_idx, alpha)
            cand_times.extend(thit.tolist())
            cand_states.extend([row.astype(float, copy=True) for row in xhit])

        if not cand_times:
            return []

        seg_order = np.concatenate((on_idx, cr_idx)) if on_idx.size or cr_idx.size else np.empty((0,), dtype=int)
        return _order_and_dedup_hits(
            cand_times,
            cand_states,
            proj,
            seg_order,
            settings.dedup_time_tol,
            settings.dedup_point_tol,
            settings.max_hits_per_traj,
        )

    def detect_batch(self, trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]", *, direction: Literal[1, -1, None] | None = None) -> "list[list[_SectionHit]]":
        out: "list[list[_SectionHit]]" = []
        for (times, states) in trajectories:
            out.append(self.detect_on_trajectory(times, states, direction=direction))
        return out

    @property
    def plane_coords(self) -> "tuple[str, str]":
        """Public accessor for the configured projection axes of the section.
        """
        return self._section_cfg.plane_coords

