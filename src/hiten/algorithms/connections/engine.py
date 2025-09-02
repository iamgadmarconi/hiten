from dataclasses import dataclass
from typing import Optional

import numpy as np

from hiten.algorithms.connections.backends import BallisticBackend
from hiten.algorithms.connections.config import (_BallisticConfig,
                                                 _ConnectionEngineConfig,
                                                 _ImpulsiveConfig,
                                                 _SearchConfig,
                                                 _SectionUseConfig)
from hiten.algorithms.connections.endpoints import LPRef, ManifoldRef, OrbitRef
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.connections.section.base import _SectionAdapter
from hiten.algorithms.connections.seeds import (generate_grid_pairs,
                                                generate_segment_pairs)
from hiten.algorithms.connections.solvers import MultipleShootingSolver


@dataclass
class ConnectionProblem:
    source: OrbitRef | ManifoldRef | LPRef
    target: OrbitRef | ManifoldRef | LPRef
    section: _SectionUseConfig
    engine: _ConnectionEngineConfig
    search: _SearchConfig
    ballistic: Optional[_BallisticConfig] = None
    impulsive: Optional[_ImpulsiveConfig] = None


class ConnectionEngine:
    """Stub orchestrator for connection-finding.

    For now this returns a placeholder result so downstream modules can import
    and the scaffolding compiles. Subsequent iterations will implement the
    full pipeline.
    """

    def solve(self, problem: ConnectionProblem) -> list[ConnectionResult]:
        # Build sections from endpoints; support offset sweep if provided
        offsets = problem.section.offset_sweep if problem.section.offset_sweep is not None else self._auto_offsets(problem)
        collected: list[ConnectionResult] = []
        for off in offsets:
            sect_cfg = _SectionUseConfig(
                mode=problem.section.mode,
                plane_coords=problem.section.plane_coords,
                direction=problem.section.direction,
                axis=problem.section.axis,
                normal=problem.section.normal,
                offset=float(off),
                offset_sweep=None,
            )

            src_sec = self._to_section(problem.source, sect_cfg)
            tgt_sec = self._to_section(problem.target, sect_cfg)

            # Simple geometry check (projection labels)
            if not src_sec.geometry_matches(tgt_sec):
                continue

            # Seeding (grid + proximity)
            pairs = generate_grid_pairs(
                src_sec,
                tgt_sec,
                grid_source=problem.search.seed_grid,
                grid_target=problem.search.seed_grid,
                radius=problem.search.neighborhood_radius,
                budget=problem.search.budget,
            )
            if problem.search.use_segments:
                seg_pairs = generate_segment_pairs(
                    src_sec,
                    tgt_sec,
                    step_src=problem.search.segment_step_source,
                    step_tgt=problem.search.segment_step_target,
                    radius=problem.search.neighborhood_radius,
                    budget=max(0, problem.search.budget - len(pairs)),
                )
                pairs.extend(seg_pairs)
            cand_idx = [(c.idx_source, c.idx_target) for c in pairs]

        # Ballistic minimal backend by default when dv=0
            ballistic_mode = (problem.impulsive is None or (problem.impulsive.max_total_dv == 0.0))
            if ballistic_mode:
                backend = BallisticBackend(problem.ballistic or _BallisticConfig())
                prelim = backend.pre_refine(src_sec, tgt_sec, cand_idx)
                solver = MultipleShootingSolver()
                results = solver.refine_ballistic(
                    src_sec,
                    tgt_sec,
                    prelim,
                    source_endpoint=problem.source,
                    target_endpoint=problem.target,
                    arc_time=0.5,
                    steps=800,
                    ballistic_cfg=(problem.ballistic or _BallisticConfig()),
                )
                if results:
                    collected.extend(results)
                continue

        # Rank and return top-k across all offsets
        if not collected:
            return []
        collected.sort(key=lambda r: ((r.tof or 1e9), -float(r.transversality_angle or 0.0)))
        # Deduplicate based on 2D meet location
        deduped: list[ConnectionResult] = []
        seen: list[tuple[float, float]] = []
        tol = float((problem.search.neighborhood_radius or 1e-6))
        for r in collected:
            if r.match_point2d is None:
                continue
            y = float(r.match_point2d[0])
            z = float(r.match_point2d[1])
            if any((abs(y - sy) <= tol and abs(z - sz) <= tol) for sy, sz in seen):
                continue
            seen.append((y, z))
            deduped.append(r)
            if len(deduped) >= max(1, int(problem.search.top_k)):
                break
        return deduped

    @staticmethod
    def _to_section(endpoint: OrbitRef | ManifoldRef | LPRef, section_cfg: _SectionUseConfig) -> _SectionAdapter:
        if isinstance(endpoint, OrbitRef):
            return endpoint.to_section(section_cfg)
        if isinstance(endpoint, ManifoldRef):
            return endpoint.to_section(section_cfg)
        raise NotImplementedError("LPRef requires generating an orbit; pass OrbitRef/ManifoldRef instead")

    def _auto_offsets(self, problem: ConnectionProblem) -> list[float]:
        sect = problem.section
        # If normal provided, use projection n·x; otherwise use axis index
        def _collect_states(ep: OrbitRef | ManifoldRef | LPRef) -> np.ndarray:
            if isinstance(ep, ManifoldRef):
                res = ep.manifold.manifold_result
                if res and res.states_list:
                    return np.vstack([np.asarray(s, dtype=float) for s in res.states_list if s is not None and len(s)])
                return np.empty((0, 6), dtype=float)
            if isinstance(ep, OrbitRef):
                if ep.orbit.trajectory is not None:
                    return np.asarray(ep.orbit.trajectory, dtype=float)
                # fallback to initial_state
                return np.asarray(ep.orbit.initial_state, dtype=float).reshape(1, 6)
            # LPRef unsupported
            return np.empty((0, 6), dtype=float)

        xs = _collect_states(problem.source)
        xt = _collect_states(problem.target)
        if xs.size == 0 and xt.size == 0:
            return [float(sect.offset)]

        vals = []
        if sect.normal is not None:
            n = np.asarray(sect.normal, dtype=float).ravel()
            if xs.size:
                vals.append(xs @ n)
            if xt.size:
                vals.append(xt @ n)
        else:
            # axis mapping
            idx_map = {"x":0, "y":1, "z":2, "vx":3, "vy":4, "vz":5}
            axis = sect.axis if isinstance(sect.axis, str) else {0:"x",1:"y",2:"z",3:"vx",4:"vy",5:"vz"}.get(int(sect.axis or 0), "x")
            k = idx_map.get(str(axis).lower(), 0)
            if xs.size:
                vals.append(xs[:, k])
            if xt.size:
                vals.append(xt[:, k])

        if not vals:
            return [float(sect.offset)]
        vals_all = np.concatenate(vals)
        vmin = float(np.min(vals_all))
        vmax = float(np.max(vals_all))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return [float(sect.offset)]
        # Number of offsets: derive from budget; minimum 21
        n_offsets = max(21, int(problem.search.budget // 50) if problem.search.budget else 21)
        return np.linspace(vmin, vmax, n_offsets).tolist()


