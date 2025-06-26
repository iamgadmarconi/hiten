r"""
hiten.algorithms.poincare.config
===============================

Configuration for Poincaré sections of the centre manifold of the spatial
circular restricted three body problem.

The module exposes a lightweight dataclass :pyclass:`_PoincareSectionConfig`
that encapsulates the configuration of a Poincaré section.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class _PoincareSectionConfig:

    _TABLE: dict[str, dict[str, object]] = {
        "q3": dict(
            section_index=2, section_value=0.0,
            plane_coords=("q2", "p2"), plane_indices=(1, 4),
            missing_coord="p3", missing_index=5,
            momentum_check_index=5, momentum_check_sign=+1.0,
            deriv_index=2,
            other_coords=("q3", "p3"), other_indices=(2, 5),
        ),
        "p3": dict(
            section_index=5, section_value=0.0,
            plane_coords=("q2", "p2"), plane_indices=(1, 4),
            missing_coord="q3", missing_index=2,
            momentum_check_index=2, momentum_check_sign=+1.0,
            deriv_index=5,
            other_coords=("q3", "p3"), other_indices=(2, 5),
        ),
        "q2": dict(
            section_index=1, section_value=0.0,
            plane_coords=("q3", "p3"), plane_indices=(2, 5),
            missing_coord="p2", missing_index=4,
            momentum_check_index=4, momentum_check_sign=+1.0,
            deriv_index=1,
            other_coords=("q2", "p2"), other_indices=(1, 4),
        ),
        "p2": dict(
            section_index=4, section_value=0.0,
            plane_coords=("q3", "p3"), plane_indices=(2, 5),
            missing_coord="q2", missing_index=1,
            momentum_check_index=1, momentum_check_sign=+1.0,
            deriv_index=4,
            other_coords=("q2", "p2"), other_indices=(1, 4),
        ),
    }

    def __init__(self, section_coord: str) -> None:
        try:
            cfg = self._TABLE[section_coord]
        except KeyError as exc:
            raise ValueError(f"Unsupported section_coord: {section_coord}") from exc

        # copy into attributes (they are read-only by convention)
        self.section_coord: str = section_coord
        for k, v in cfg.items():
            setattr(self, k, v)

    def get_section_value(self, state: np.ndarray) -> float:
        return float(state[self.section_index])

    def extract_plane_coords(self, state: np.ndarray) -> Tuple[float, float]:
        i, j = self.plane_indices 
        return float(state[i]), float(state[j])

    def extract_other_coords(self, state: np.ndarray) -> Tuple[float, float]:
        i, j = self.other_indices
        return float(state[i]), float(state[j])

    def build_state(
        self,
        plane_vals: Tuple[float, float],
        other_vals: Tuple[float, float],
    ) -> Tuple[float, float, float, float]:
        q2 = p2 = q3 = p3 = 0.0
        if self.plane_coords == ("q2", "p2"):
            q2, p2 = plane_vals
            q3, p3 = other_vals
        else:
            q3, p3 = plane_vals
            q2, p2 = other_vals

        if self.section_coord == "q2":
            q2 = self.section_value
        elif self.section_coord == "p2":    
            p2 = self.section_value
        elif self.section_coord == "q3":
            q3 = self.section_value
        else:  # "p3"
            p3 = self.section_value
        return q2, p2, q3, p3

    def build_constraint_dict(self, **kwargs) -> dict[str, float]:
        out: dict[str, float] = {self.section_coord: self.section_value} 
        for k, v in kwargs.items():
            if k in {"q1", "q2", "q3", "p1", "p2", "p3"}:
                out[k] = float(v)
        return out


_SECTION_CACHE: dict[str, _PoincareSectionConfig] = {
    name: _PoincareSectionConfig(name) for name in ("q2", "p2", "q3", "p3")
}

def _get_section_config(section_coord: str) -> _PoincareSectionConfig:
    try:
        return _SECTION_CACHE[section_coord]
    except KeyError as exc:
        raise ValueError(f"Unsupported section_coord: {section_coord}") from exc
