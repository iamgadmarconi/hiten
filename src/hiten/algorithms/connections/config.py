from dataclasses import dataclass
from typing import Literal


@dataclass
class _SearchConfig:
    """Planner/search configuration that generates/filters candidates."""

    # Accept if ||ΔV|| <= delta_v_tol
    delta_v_tol: float = 1e-3
    # Classify ballistic if ||ΔV|| <= ballistic_tol
    ballistic_tol: float = 1e-8
    # Pairing radius on the section plane
    eps2d: float = 1e-4

class ConnectionConfig(_SearchConfig):
    n_workers: int = 1
