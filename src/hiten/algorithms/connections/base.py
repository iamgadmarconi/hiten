from dataclasses import dataclass, field
from typing import Sequence


from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.engine import _ConnectionEngine, _ConnectionProblem
from hiten.algorithms.connections.results import ConnectionResult
from hiten.utils.plots import plot_poincare_connections_map


@dataclass
class Connections:
    """User-facing facade for connection discovery and plotting.

    Wraps `ConnectionEngine` and provides convenience plotting helpers.
    """


    def solve(self, source: _ManifoldInterface, target: _ManifoldInterface) -> list[ConnectionResult]:
        problem = _ConnectionProblem(
            source=source,
            target=target,
            search=self.search_cfg,
        )
        return _ConnectionEngine().solve(problem)

    def plot(self, source: _ManifoldInterface, target: _ManifoldInterface, results: ConnectionResult | Sequence[ConnectionResult], **kwargs):
        pass
