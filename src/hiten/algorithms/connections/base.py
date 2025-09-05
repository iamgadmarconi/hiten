from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.engine import (_ConnectionEngine,
                                                 _ConnectionProblem)
from hiten.system.manifold import Manifold
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.utils.plots import plot_poincare_connections_map


@dataclass
class Connections:
    """User-facing facade for connection discovery and plotting.

    Wraps `ConnectionEngine` and provides convenience plotting helpers.
    """
    # User-provided single section configuration and direction
    section: _SynodicMapConfig
    direction: Literal[1, -1, None] | None = None

    # Optional search config
    search_cfg: _SearchConfig | None = None

    # Internal cache for plot convenience
    _last_source: Manifold | None = None
    _last_target: Manifold | None = None
    _last_results: list[ConnectionResult] | None = None

    def solve(self, source: Manifold, target: Manifold) -> list[ConnectionResult]:
        # Convert to internal interfaces
        from hiten.algorithms.connections.interfaces import _ManifoldInterface  # local import keeps type internal
        src_if = _ManifoldInterface(manifold=source)
        tgt_if = _ManifoldInterface(manifold=target)

        problem = _ConnectionProblem(
            source=src_if,
            target=tgt_if,
            section=self.section,
            direction=self.direction,
            search=self.search_cfg,
        )
        results = _ConnectionEngine().solve(problem)
        self._last_source = source
        self._last_target = target
        self._last_results = results
        return results

    def plot(self, **kwargs):
        # Use cached artifacts; user should call solve() first
        if self._last_source is None or self._last_target is None:
            raise ValueError("Nothing to plot: call solve(source, target) first.")
        from hiten.algorithms.connections.interfaces import _ManifoldInterface  # internal
        src_if = _ManifoldInterface(manifold=self._last_source)
        tgt_if = _ManifoldInterface(manifold=self._last_target)

        # Build section hits for both manifolds on the configured synodic section
        sec_u = src_if.to_section(self.section, direction=self.direction)
        sec_s = tgt_if.to_section(self.section, direction=self.direction)

        pts_u = np.asarray(sec_u.points, dtype=float)
        pts_s = np.asarray(sec_s.points, dtype=float)
        labels = tuple(sec_u.labels)

        # Use cached results
        res_list = self._last_results or []

        if res_list:
            match_pts = np.asarray([r.point2d for r in res_list], dtype=float)
            match_vals = np.asarray([r.delta_v for r in res_list], dtype=float)
        else:
            match_pts = None
            match_vals = None

        return plot_poincare_connections_map(
            points_src=pts_u,
            points_tgt=pts_s,
            labels=labels,
            match_points=match_pts,
            match_values=match_vals,
            ballistic=False,
            **kwargs,
        )
