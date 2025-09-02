from dataclasses import dataclass
from typing import List

import numpy as np

from hiten.algorithms.connections.config import _BallisticConfig
from hiten.algorithms.connections.endpoints import ManifoldRef, OrbitRef
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.connections.section.base import _SectionAdapter
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.utils.types import Trajectory


@dataclass
class MultipleShootingSolver:
    """Unified solver placeholder.

    This stub provides a refinement hook. The ballistic implementation with
    impulse_count=0 will be filled in subsequent commits.
    """

    def refine_ballistic(
        self,
        src: _SectionAdapter,
        tgt: _SectionAdapter,
        prelim: List[ConnectionResult],
        *,
        source_endpoint: OrbitRef | ManifoldRef,
        target_endpoint: OrbitRef | ManifoldRef,
        arc_time: float = 0.5,
        steps: int = 800,
        ballistic_cfg: _BallisticConfig | None = None,
    ) -> List[ConnectionResult]:
        refined: List[ConnectionResult] = []
        ps = src.points2d()
        pt = tgt.points2d()
        xs = src.states6d()
        xt = tgt.states6d()
        ts = src.times1d()
        tt = tgt.times1d()

        for r in prelim:
            mp = r.match_point2d
            if mp is None:
                refined.append(r)
                continue
            # Nearest indices to match point in each section
            di_s = np.sum((ps - mp) ** 2, axis=1)
            di_t = np.sum((pt - mp) ** 2, axis=1)
            i = int(np.argmin(di_s))
            j = int(np.argmin(di_t))

            # Integrate short arcs away from the section hits using the correct system
            sys_s = self._system_from_endpoint(source_endpoint)
            sys_t = self._system_from_endpoint(target_endpoint)

            # Initial meet estimate: use the actual source section point when close
            xmeet = xs[i].copy()

            # Newton refinement over (xmeet, tau_s, tau_t)
            cfg = ballistic_cfg or _BallisticConfig()
            tau_min = float(getattr(cfg, "tau_min", 1e-4))
            tau_max_eff = float(arc_time if getattr(cfg, "tau_max", None) is None else getattr(cfg, "tau_max"))
            tau_s = max(tau_min, 0.1 * tau_max_eff)
            tau_t = max(tau_min, 0.1 * tau_max_eff)
            max_iter = int(getattr(cfg, "newton_max_iters", 5))
            tol_newton = float(getattr(cfg, "newton_tol", 1e-10))
            # Optional: read from ballistic backend config if needed by passing in
            for _ in range(max_iter):
                # Short flows and STMs from the common meet state
                x_s, _, Phi_s, _ = _compute_stm(sys_s.var_dynsys, xmeet, tau_s, steps=max(50, steps//10), forward=-1)
                x_t, _, Phi_t, _ = _compute_stm(sys_t.var_dynsys, xmeet, tau_t, steps=max(50, steps//10), forward=1)
                xs_tau = x_s[-1]
                xt_tau = x_t[-1]
                resid = xs_tau - xt_tau  # 6D residual

                # Build plane-tangent projector
                if src.normal is not None and src.offset is not None:
                    n = src.normal.astype(float, copy=False).ravel()
                    nn = float(np.dot(n, n))
                    P = np.eye(6)
                    if nn > 0:
                        P -= np.outer(n, n) / nn
                else:
                    P = np.eye(6)

                # Time derivatives: dx/dtau = +/- f(x)
                f_s = sys_s.dynsys.rhs(0.0, xs_tau)
                f_t = sys_t.dynsys.rhs(0.0, xt_tau)
                ddt_s = -np.asarray(f_s, dtype=float).ravel()
                ddt_t =  np.asarray(f_t, dtype=float).ravel()

                # Jacobian w.r.t variables [dxmeet(6), dtau_s, dtau_t]
                Jx = (Phi_s - Phi_t)
                Jt_s = ddt_s.reshape(6, 1)
                Jt_t = ddt_t.reshape(6, 1)
                J = np.hstack([Jx, Jt_s, Jt_t])

                # Project into plane tangent space
                Jp = P @ J
                rp = P @ resid

                try:
                    delta, *_ = np.linalg.lstsq(Jp, rp, rcond=None)
                except Exception:
                    break

                dx = delta[:6]
                dt_s = float(delta[6])
                dt_t = float(delta[7])

                xmeet = xmeet - dx
                xmeet = self._project_onto_plane(xmeet, src.normal, src.offset)
                tau_s = float(np.clip(tau_s - dt_s, tau_min, tau_max_eff))
                tau_t = float(np.clip(tau_t - dt_t, tau_min, tau_max_eff))

                if float(np.linalg.norm(rp)) < tol_newton:
                    break

            # After Newton, evaluate final projected residual; discard if too large
            x_s, _, _, _ = _compute_stm(sys_s.var_dynsys, xmeet, tau_s, steps=max(50, steps//10), forward=-1)
            x_t, _, _, _ = _compute_stm(sys_t.var_dynsys, xmeet, tau_t, steps=max(50, steps//10), forward=1)
            xs_tau = x_s[-1]
            xt_tau = x_t[-1]
            resid = xs_tau - xt_tau
            if src.normal is not None and src.offset is not None:
                n = src.normal.astype(float, copy=False).ravel()
                nn = float(np.dot(n, n))
                P = np.eye(6)
                if nn > 0:
                    P -= np.outer(n, n) / nn
            else:
                P = np.eye(6)
            rp_final = P @ resid
            if float(np.linalg.norm(rp_final)) > float(getattr(cfg, "tol_refine", 1e-12)):
                # Skip spurious candidate that did not converge to a meet
                continue

            # Integrate from anchors to meet state with event stopping
            seg_s = self._integrate_from_anchor_to_meet(
                source_endpoint,
                meet_state=xmeet,
                plane_normal=src.normal,
                plane_offset=src.offset,
                forward=-1,
                t_max=tau_max_eff,
                steps=steps,
            )
            seg_t = self._integrate_from_anchor_to_meet(
                target_endpoint,
                meet_state=xmeet,
                plane_normal=tgt.normal,
                plane_offset=tgt.offset,
                forward=1,
                t_max=tau_max_eff,
                steps=steps,
            )

            tof_val = float(tau_s + tau_t)
            # Use refined meet state's projection as the match point
            mp2d_refined = self._project_state_to_section2d(xmeet, src.plane_coords)
            # Sanity check: ensure refined 2D point lies close to both source/target sections
            dsrc = float(np.min(np.sum((ps - mp2d_refined) ** 2, axis=1))) if ps.size else float('inf')
            dtgt = float(np.min(np.sum((pt - mp2d_refined) ** 2, axis=1))) if pt.size else float('inf')
            tol2 = float(getattr(cfg, "tol_intersection", 1e-3)) ** 2
            if not (dsrc <= tol2 and dtgt <= tol2):
                continue
            r2 = ConnectionResult(
                ballistic=True,
                dv_list=[],
                total_dv=0.0,
                tof=tof_val,
                tau_source=float(tau_s),
                tau_target=float(tau_t),
                match_point2d=mp2d_refined,
                transversality_angle=r.transversality_angle,
                source_leg=seg_s,
                target_leg=seg_t,
                section_labels=r.section_labels,
            )
            refined.append(r2)

        return refined

    @staticmethod
    def _project_state_to_section2d(x: np.ndarray, labels: tuple[str, str]) -> tuple[float, float]:
        index_map = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5}
        i = index_map.get(str(labels[0]).lower(), 0)
        j = index_map.get(str(labels[1]).lower(), 1)
        return (float(x[i]), float(x[j]))

    @staticmethod
    def _project_onto_plane(x: np.ndarray, normal: np.ndarray | None, offset: float | None) -> np.ndarray:
        if normal is None or offset is None:
            return x
        n = normal.astype(float, copy=False).ravel()
        c = float(offset)
        g = float(np.dot(n, x) - c)
        denom = float(np.dot(n, n))
        if denom == 0.0:
            return x
        return x - (g / denom) * n

    @staticmethod
    def _system_from_endpoint(ep: OrbitRef | ManifoldRef):
        if isinstance(ep, OrbitRef):
            return ep.orbit.system
        # ManifoldRef
        return ep.manifold.generating_orbit.system

    @staticmethod
    def _g_value(state: np.ndarray, normal: np.ndarray | None, offset: float | None) -> float | None:
        if normal is None or offset is None:
            return None
        return float(np.dot(normal.astype(float, copy=False).ravel(), state.astype(float, copy=False).ravel()) - float(offset))

    def _integrate_to_next_section_crossing(
        self,
        *,
        system,
        x0: np.ndarray,
        normal: np.ndarray | None,
        offset: float | None,
        forward: int,
        t_max: float,
        steps: int,
    ) -> Trajectory | None:
        # If no geometry provided, fall back to fixed arc integration
        if normal is None or offset is None:
            t, x = system.propagate(x0, tf=t_max, steps=steps, method="scipy", order=8, forward=forward)
            return Trajectory(times=t, states=x)

        t, x = system.propagate(x0, tf=t_max, steps=steps, method="scipy", order=8, forward=forward)
        g = np.array([self._g_value(row, normal, offset) for row in x])
        # Find first crossing after index 0 in the requested forward direction
        sgn = np.sign(g)
        idx = None
        for k in range(0, len(g) - 1):
            if g[k] is None or g[k + 1] is None:
                continue
            if (g[k] == 0.0):
                idx = k
                break
            if g[k] * g[k + 1] <= 0.0 and g[k] != g[k + 1]:
                idx = k
                break
        if idx is None:
            return Trajectory(times=t, states=x)

        # Linear time interpolation between k and k+1
        g0 = float(g[idx])
        g1 = float(g[idx + 1])
        if g1 == g0:
            s = 0.5
        else:
            s = g0 / (g0 - g1)
            s = float(max(0.0, min(1.0, s)))
        thit = (1.0 - s) * t[idx] + s * t[idx + 1]
        xhit = x[idx] + s * (x[idx + 1] - x[idx])

        # Return truncated segment including the hit point
        if forward >= 0:
            t_seg = np.concatenate([t[: idx + 1], np.array([thit])])
            x_seg = np.vstack([x[: idx + 1], xhit])
        else:
            t_seg = np.concatenate([np.array([thit]), t[idx + 1 :]])
            x_seg = np.vstack([xhit, x[idx + 1 :]])
        return Trajectory(times=t_seg, states=x_seg)

    def _integrate_from_anchor_to_meet(
        self,
        endpoint: OrbitRef | ManifoldRef,
        *,
        meet_state: np.ndarray,
        plane_normal: np.ndarray | None,
        plane_offset: float | None,
        forward: int,
        t_max: float,
        steps: int,
    ) -> Trajectory:
        system = self._system_from_endpoint(endpoint)
        # Choose anchor: for orbits, current initial_state; for manifolds, nearest trajectory state
        if isinstance(endpoint, OrbitRef):
            x_anchor = endpoint.orbit.initial_state
        else:
            # Take first state of first manifold trajectory as a simple anchor
            res = endpoint.manifold.manifold_result
            if res is None or not res.states_list:
                x_anchor = meet_state
            else:
                x_anchor = np.asarray(res.states_list[0][0], dtype=float)

        # Integrate toward meet; if no plane provided, do fixed arc
        if plane_normal is None or plane_offset is None:
            t, x = system.propagate(x_anchor, tf=t_max, steps=steps, method="scipy", order=8, forward=forward)
            return Trajectory(times=t, states=x)

        # Basic secant shooting: integrate and cut at the nearest approach to the meet plane
        t, x = system.propagate(x_anchor, tf=t_max, steps=steps, method="scipy", order=8, forward=forward)
        g = np.array([self._g_value(row, plane_normal, plane_offset) for row in x])
        # Choose segment reaching the plane; refine linearly
        idx = None
        for k in range(0, len(g) - 1):
            if g[k] is None or g[k + 1] is None:
                continue
            if g[k] * g[k + 1] <= 0.0 and g[k] != g[k + 1]:
                idx = k
                break
        if idx is None:
            return Trajectory(times=t, states=x)
        g0 = float(g[idx])
        g1 = float(g[idx + 1])
        s = 0.5 if g1 == g0 else float(np.clip(g0 / (g0 - g1), 0.0, 1.0))
        thit = (1.0 - s) * t[idx] + s * t[idx + 1]
        xhit = x[idx] + s * (x[idx + 1] - x[idx])
        if forward >= 0:
            t_seg = np.concatenate([t[: idx + 1], np.array([thit])])
            x_seg = np.vstack([x[: idx + 1], xhit])
        else:
            t_seg = np.concatenate([np.array([thit]), t[idx + 1 :]])
            x_seg = np.vstack([xhit, x[idx + 1 :]])
        return Trajectory(times=t_seg, states=x_seg)


