from dataclasses import dataclass

from hiten.algorithms.connections.config import _SearchConfig
from hiten.algorithms.connections.interfaces import _ManifoldInterface
from hiten.algorithms.connections.results import ConnectionResult


@dataclass
class _ConnectionProblem:
    source: _ManifoldInterface
    target: _ManifoldInterface
    search: _SearchConfig


class _ConnectionEngine:
    """Stub orchestrator for connection-finding.

    For now this returns a placeholder result so downstream modules can import
    and the scaffolding compiles. Subsequent iterations will implement the
    full pipeline.
    """

    def solve(self, problem: _ConnectionProblem) -> list[ConnectionResult]:
        ...