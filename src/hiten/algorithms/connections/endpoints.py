from dataclasses import dataclass
from typing import Literal, Optional

from hiten.algorithms.poincare.synodic.base import SynodicMap
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.system.libration.base import LibrationPoint
from hiten.system.manifold import Manifold
from hiten.system.orbits.base import PeriodicOrbit

from hiten.algorithms.connections.config import _SectionUseConfig
from hiten.algorithms.connections.section.base import _SectionAdapter


@dataclass
class OrbitRef:
    orbit: PeriodicOrbit
    phase: float | None = None  # future use; not required for section extraction

    def to_section(self, section_cfg: _SectionUseConfig) -> _SectionAdapter:
        if self.orbit.times is None or self.orbit.trajectory is None:
            raise ValueError("Orbit must be propagated before building a section")

        if section_cfg.mode != "synodic":
            raise NotImplementedError("Only synodic sections are supported at this stage")

        map_cfg = _SynodicMapConfig(
            section_axis=section_cfg.axis if section_cfg.normal is None else None,
            section_offset=section_cfg.offset,
            section_normal=section_cfg.normal,
            plane_coords=section_cfg.plane_coords,
        )
        pmap = SynodicMap(map_cfg)
        sec = pmap.from_orbit(self.orbit, direction=section_cfg.direction)
        return _SectionAdapter(
            section=sec,
            plane_coords=section_cfg.plane_coords,
            normal=section_cfg.normal,
            offset=section_cfg.offset,
            axis=section_cfg.axis if section_cfg.normal is None else None,
        )


@dataclass
class ManifoldRef:
    manifold: Manifold

    def to_section(self, section_cfg: _SectionUseConfig) -> _SectionAdapter:
        result = self.manifold.manifold_result
        if result is None or not result.states_list:
            raise ValueError("Manifold must be computed before building a section")

        if section_cfg.mode != "synodic":
            raise NotImplementedError("Only synodic sections are supported at this stage")

        map_cfg = _SynodicMapConfig(
            section_axis=section_cfg.axis if section_cfg.normal is None else None,
            section_offset=section_cfg.offset,
            section_normal=section_cfg.normal,
            plane_coords=section_cfg.plane_coords,
        )
        pmap = SynodicMap(map_cfg)
        sec = pmap.from_manifold(result, direction=section_cfg.direction)
        return _SectionAdapter(
            section=sec,
            plane_coords=section_cfg.plane_coords,
            normal=section_cfg.normal,
            offset=section_cfg.offset,
            axis=section_cfg.axis if section_cfg.normal is None else None,
        )


@dataclass
class LPRef:
    point: LibrationPoint
    # Optional generator parameters to spawn an orbit if needed later
    family: Optional[str] = None
    amplitude: Optional[float] = None
    zenith: Optional[Literal["northern", "southern"]] = None

    def to_section(self, *args, **kwargs) -> _SectionAdapter:
        raise NotImplementedError("LPRef requires generating an orbit; use OrbitRef or ManifoldRef after creation")


