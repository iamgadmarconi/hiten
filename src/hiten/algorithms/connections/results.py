from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np


@dataclass
class ConnectionResult:
    kind: Literal["impulsive", "ballistic"]
    delta_v: float
    point2d: Tuple[float, float]
    state_u: np.ndarray
    state_s: np.ndarray
    index_u: int
    index_s: int