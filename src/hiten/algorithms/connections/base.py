from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hiten.algorithms.connections.config import (_BallisticConfig,
                                                 _ConnectionEngineConfig,
                                                 _ImpulsiveConfig,
                                                 _SearchConfig,
                                                 _SectionUseConfig)
from hiten.algorithms.connections.endpoints import LPRef, ManifoldRef, OrbitRef
from hiten.algorithms.connections.engine import ConnectionEngine, ConnectionProblem
from hiten.algorithms.connections.results import ConnectionResult
from hiten.utils.plots import plot_poincare_connections_map


@dataclass
class Connections:
    """User-facing facade for connection discovery and plotting.

    Wraps `ConnectionEngine` and provides convenience plotting helpers.
    """

    engine_cfg: _ConnectionEngineConfig = field(default_factory=_ConnectionEngineConfig)
    section_cfg: _SectionUseConfig = field(default_factory=_SectionUseConfig)
    search_cfg: _SearchConfig = field(default_factory=_SearchConfig)
    ballistic_cfg: _BallisticConfig | None = None
    impulsive_cfg: _ImpulsiveConfig | None = None

    def solve(self, source: OrbitRef | ManifoldRef | LPRef, target: OrbitRef | ManifoldRef | LPRef) -> list[ConnectionResult]:
        problem = ConnectionProblem(
            source=source,
            target=target,
            section=self.section_cfg,
            engine=self.engine_cfg,
            search=self.search_cfg,
            ballistic=self.ballistic_cfg,
            impulsive=self.impulsive_cfg,
        )
        return ConnectionEngine().solve(problem)

    def plot_poincare(self, source: OrbitRef | ManifoldRef | LPRef, target: OrbitRef | ManifoldRef | LPRef, results: ConnectionResult | Sequence[ConnectionResult], **kwargs):
        # Build sections for both endpoints
        engine = ConnectionEngine()
        problem = ConnectionProblem(
            source=source,
            target=target,
            section=self.section_cfg,
            engine=self.engine_cfg,
            search=self.search_cfg,
            ballistic=self.ballistic_cfg,
            impulsive=self.impulsive_cfg,
        )
        src_sec = source.to_section(self.section_cfg) if isinstance(source, (OrbitRef, ManifoldRef)) else None
        tgt_sec = target.to_section(self.section_cfg) if isinstance(target, (OrbitRef, ManifoldRef)) else None
        if src_sec is None or tgt_sec is None:
            raise NotImplementedError("LPRef requires generating an orbit; pass OrbitRef/ManifoldRef instead")

        ps = src_sec.points2d()
        pt = tgt_sec.points2d()
        labels = src_sec.labels()

        res_list = list(results) if isinstance(results, Sequence) and not isinstance(results, ConnectionResult) else [results]  # type: ignore
        mpts = []
        vals = []
        for r in res_list:
            if r.match_point2d is None:
                continue
            mpts.append(r.match_point2d)
            vals.append(float(r.transversality_angle) if r.ballistic and (r.transversality_angle is not None) else float(r.total_dv))

        mp = np.asarray(mpts, dtype=float) if mpts else np.empty((0, 2), dtype=float)
        mv = np.asarray(vals, dtype=float) if vals else None
        return plot_poincare_connections_map(ps, pt, labels, match_points=mp, match_values=mv, ballistic=(res_list[0].ballistic if res_list else True), **kwargs)


