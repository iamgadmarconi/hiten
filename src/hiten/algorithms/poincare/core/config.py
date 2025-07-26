from abc import ABC
from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class _ReturnMapConfig(ABC):
    dt: float = 1e-2
    method: Literal["scipy", "rk", "symplectic", "adaptive"] = "rk"
    order: int = 4
    c_omega_heuristic: float = 20.0

    n_seeds: int = 20
    n_iter: int = 40

    compute_on_init: bool = False


class _SectionConfig(ABC):

    section_coord: str
    plane_coords: Tuple[str, str]