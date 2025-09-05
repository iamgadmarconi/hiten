from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from hiten.algorithms.connections.config import (_SearchConfig,
                                                 _SectionUseConfig)
from hiten.algorithms.connections.endpoints import ManifoldRef
from hiten.algorithms.connections.engine import ConnectionEngine, ConnectionProblem
from hiten.algorithms.connections.results import ConnectionResult
from hiten.utils.plots import plot_poincare_connections_map


@dataclass
class Connections:
    """User-facing facade for connection discovery and plotting.

    Wraps `ConnectionEngine` and provides convenience plotting helpers.
    """


    def solve(self, source: ManifoldRef, target: ManifoldRef) -> list[ConnectionResult]:
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

    def plot(self, source: ManifoldRef, target: ManifoldRef, results: ConnectionResult | Sequence[ConnectionResult], **kwargs):
        pass
