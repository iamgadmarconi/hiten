from dataclasses import dataclass
from typing import Literal

from hiten.system.manifold import Manifold
from hiten.algorithms.poincare.synodic.base import SynodicMap
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.core.base import _Section


@dataclass
class _ManifoldInterface:
    manifold: Manifold

    def to_section(
        self,
        config: _SynodicMapConfig | None = None,
        *,
        direction: Literal[1, -1, None] | None = None,
    ) -> _Section:
        """Return synodic section hits for this manifold.

        Parameters
        ----------
        config : _SynodicMapConfig, optional
            Geometry and detection settings for the synodic section. If not
            provided, defaults from `_SynodicMapConfig()` are used.
        direction : {1, -1, None}, optional
            Crossing direction filter passed to the detector. ``None`` accepts
            both directions.

        Returns
        -------
        _Section
            Cached section object containing 2-D points, 6-D states and hit
            times suitable for pairing in connections.
        """

        if self.manifold.manifold_result is None:
            raise ValueError("Manifold must be computed before extracting section hits")

        cfg = config or _SynodicMapConfig()
        syn = SynodicMap(cfg)
        return syn.from_manifold(self.manifold, direction=direction)

