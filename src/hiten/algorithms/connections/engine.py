from dataclasses import dataclass
from typing import Literal

import numpy as np

from hiten.algorithms.connections.backends import _radius_pairs_2d
from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig


@dataclass
class _ConnectionProblem:
    source: _ManifoldInterface
    target: _ManifoldInterface
    section: _SynodicMapConfig
    direction: Literal[1, -1, None] | None
    search: _SearchConfig


class _ConnectionEngine:
    """Stub orchestrator for connection-finding.

    For now this returns a placeholder result so downstream modules can import
    and the scaffolding compiles. Subsequent iterations will implement the
    full pipeline.
    """

    def solve(self, problem: _ConnectionProblem) -> list[ConnectionResult]:
        # Extract hits for both manifolds on the user-provided synodic section
        sec_u = problem.source.to_section(problem.section, direction=problem.direction)
        sec_s = problem.target.to_section(problem.section, direction=problem.direction)

        pu = np.asarray(sec_u.points, dtype=float)
        ps = np.asarray(sec_s.points, dtype=float)
        Xu = np.asarray(sec_u.states, dtype=float)
        Xs = np.asarray(sec_s.states, dtype=float)

        if pu.size == 0 or ps.size == 0:
            return []

        # 2D radius pairing (numba)
        eps = float(getattr(problem.search, "eps2d", 1e-4)) if problem.search else 1e-4
        dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
        bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8

        pairs_arr = _radius_pairs_2d(pu, ps, eps)
        if pairs_arr.size == 0:
            return []

        # Mutual nearest filter within eps to reduce duplicates/false matches
        di = pu[pairs_arr[:, 0]] - ps[pairs_arr[:, 1]]
        d2 = np.sum(di * di, axis=1)

        # For each i, j pick the minimal distance pairs
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

        results: list[ConnectionResult] = []
        for i, j in pairs:
            vu = Xu[i, 3:6]
            vs = Xs[j, 3:6]
            dv = float(np.linalg.norm(vu - vs))
            if dv <= dv_tol:
                kind = "ballistic" if dv <= bal_tol else "impulsive"
                pt = (float(pu[i, 0]), float(pu[i, 1]))
                results.append(ConnectionResult(kind=kind, delta_v=dv, point2d=pt, state_u=Xu[i].copy(), state_s=Xs[j].copy(), index_u=int(i), index_s=int(j)))

        # Optionally sort by ΔV
        results.sort(key=lambda r: r.delta_v)
        return results