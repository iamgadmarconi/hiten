from dataclasses import dataclass
from typing import List, Optional, Tuple

from hiten.algorithms.utils.types import Trajectory


@dataclass
class ConnectionResult:
    ballistic: bool
    dv_list: List[float]
    total_dv: float
    tof: Optional[float]
    match_point2d: Optional[Tuple[float, float]]
    transversality_angle: Optional[float]
    tau_source: Optional[float] = None
    tau_target: Optional[float] = None

    source_leg: Optional[Trajectory] = None
    target_leg: Optional[Trajectory] = None

    section_labels: Optional[Tuple[str, str]] = None

    def as_dict(self) -> dict:
        return {
            "ballistic": bool(self.ballistic),
            "dv_list": list(self.dv_list),
            "total_dv": float(self.total_dv),
            "tof": None if self.tof is None else float(self.tof),
            "tau_source": None if self.tau_source is None else float(self.tau_source),
            "tau_target": None if self.tau_target is None else float(self.tau_target),
            "match_point2d": None if self.match_point2d is None else tuple(self.match_point2d),
            "transversality_angle": None if self.transversality_angle is None else float(self.transversality_angle),
            "section_labels": None if self.section_labels is None else tuple(self.section_labels),
        }


