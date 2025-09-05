from dataclasses import dataclass
from typing import Literal


@dataclass
class _SearchConfig:
    """Planner/search configuration that generates/filters candidates."""

    strategy: Literal[""]
    delta_v_tol: float
    ballistic_tol: float
    eps2d: float

class ConnectionConfig(_SearchConfig):
    n_workers: int = 1
