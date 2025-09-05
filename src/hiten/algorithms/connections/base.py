from dataclasses import dataclass, field
from typing import Sequence, Literal


from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.engine import _ConnectionEngine, _ConnectionProblem
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.connections.config import _SearchConfig
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

    def solve(self, source: _ManifoldInterface, target: _ManifoldInterface) -> list[ConnectionResult]:
        problem = _ConnectionProblem(
            source=source,
            target=target,
            section=self.section,
            direction=self.direction,
            search=self.search_cfg,
        )
        return _ConnectionEngine().solve(problem)

    def plot(self, source: _ManifoldInterface, target: _ManifoldInterface, results: ConnectionResult | Sequence[ConnectionResult], **kwargs):
        pass
