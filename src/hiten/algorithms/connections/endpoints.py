from dataclasses import dataclass

from hiten.system.manifold import Manifold

from hiten.algorithms.connections.config import _SectionUseConfig


@dataclass
class ManifoldRef:
    manifold: Manifold

    def to_section(self, section_cfg: _SectionUseConfig):
        pass

