from dataclasses import dataclass

from hiten.algorithms.connections.config import (_SearchConfig,
                                                 _SectionUseConfig)
from hiten.algorithms.connections.endpoints import ManifoldRef
from hiten.algorithms.connections.results import ConnectionResult


@dataclass
class ConnectionProblem:
    source: ManifoldRef
    target: ManifoldRef
    section: _SectionUseConfig
    search: _SearchConfig


class ConnectionEngine:
    """Stub orchestrator for connection-finding.

    For now this returns a placeholder result so downstream modules can import
    and the scaffolding compiles. Subsequent iterations will implement the
    full pipeline.
    """

    def solve(self, problem: ConnectionProblem) -> list[ConnectionResult]:
        ...