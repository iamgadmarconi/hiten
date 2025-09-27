"""
Interface classes for synodic Poincare map computation.

This module provides interface classes that abstract synodic Poincare
map computation for the synodic module. These interfaces handle the
conversion between synodic map configuration and the engine interface.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Sequence, Tuple

import numpy as np

from hiten.algorithms.poincare.core.interfaces import (_PoincareBaseInterface,
                                                       _SectionInterface)
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.events import _AffinePlaneEvent
from hiten.algorithms.poincare.synodic.types import (
    SynodicMapResults,
    _SynodicMapProblem,
)
from hiten.algorithms.types.states import SynodicState
from hiten.algorithms.types.core import _BackendCall


@dataclass(frozen=True)
class _SynodicEngineConfig:
    """Configuration adapter for synodic Poincare engine."""

    dt: float
    n_iter: int
    n_workers: int | None
    n_seeds: int
    section_interface: "_SynodicSectionInterface"
    config: _SynodicMapConfig

    @classmethod
    def from_config(
        cls,
        cfg: _SynodicMapConfig,
        section_iface: "_SynodicSectionInterface",
    ) -> "_SynodicEngineConfig":
        return cls(
            dt=0.0,
            n_iter=1,
            n_workers=cfg.n_workers,
            n_seeds=0,
            section_interface=section_iface,
            config=cfg,
        )

    def __repr__(self) -> str:
        return f"_SynodicEngineConfig(n_workers={self.n_workers})"


@dataclass(frozen=True)
class _SynodicSectionInterface(_SectionInterface):
    """Section interface for synodic affine hyperplanes."""

    section_coord: str | None  # not a single coord; keep None
    plane_coords: tuple[str, str]
    normal: np.ndarray
    offset: float

    @staticmethod
    def axis_normal(axis: str | int) -> np.ndarray:
        normal = np.zeros(6, dtype=float)
        if isinstance(axis, str):
            idx = int(SynodicState[axis.upper()])
        else:
            idx = int(axis)
        normal[idx] = 1.0
        return normal

    @staticmethod
    def from_normal(normal: Sequence[float], offset: float, plane_coords: Tuple[str, str]) -> "_SynodicSectionInterface":
        n_arr = np.asarray(normal, dtype=float)
        if n_arr.ndim != 1 or n_arr.size != 6 or not np.all(np.isfinite(n_arr)):
            raise ValueError("normal must be a finite 1-D array of length 6")
        return _SynodicSectionInterface(section_coord=None, plane_coords=(str(plane_coords[0]), str(plane_coords[1])), normal=n_arr, offset=float(offset))

    def build_event(self, *, direction: Literal[1, -1, None] = None) -> _AffinePlaneEvent:
        return _AffinePlaneEvent(normal=self.normal, offset=self.offset, direction=direction)

    def coordinate_index(self, axis: str) -> int:
        try:
            return int(SynodicState[axis.upper()])
        except KeyError as exc:
            raise ValueError(f"Unsupported synodic axis '{axis}'") from exc


class _SynodicInterface(
    _PoincareBaseInterface[
        _SynodicMapConfig, 
        _SynodicMapProblem, 
        SynodicMapResults, 
        Tuple[np.ndarray, np.ndarray, np.ndarray | None]
    ]
):

    def __init__(self) -> None:
        super().__init__()

    def create_problem(
        self,
        *,
        config: _SynodicMapConfig,
        plane_coords: Tuple[str, str],
        normal: np.ndarray,
        offset: float,
        direction: Literal[1, -1, None] | None,
        trajectories: Sequence[tuple[np.ndarray, np.ndarray]] | None,
    ) -> _SynodicMapProblem:
        from hiten.algorithms.poincare.synodic.config import _SynodicSectionConfig
        
        section_cfg = _SynodicSectionConfig(
            normal=normal,
            offset=offset,
            plane_coords=plane_coords
        )
        
        return _SynodicMapProblem(
            plane_coords=plane_coords,
            direction=direction,
            n_workers=config.n_workers or 1,
            section_cfg=section_cfg,
            map_cfg=config,
            trajectories=trajectories,
        )

    def to_backend_inputs(self, problem: _SynodicMapProblem):
        return _BackendCall(kwargs={"trajectories": problem.trajectories, "direction": problem.direction})

    def to_results(self, outputs, *, problem: _SynodicMapProblem) -> SynodicMapResults:
        points, states, times, trajectory_indices = outputs
        plane_coords = problem.plane_coords
        return SynodicMapResults(points, states, plane_coords, times, trajectory_indices)