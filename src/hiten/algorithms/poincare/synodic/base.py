from typing import Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.core.base import _ReturnMapBase, _Section
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.config import (_get_section_config,
                                                      _SynodicMapConfig,
                                                      _SynodicSectionConfig)
from hiten.algorithms.poincare.synodic.engine import (
    _SynodicEngine, _SynodicEngineConfigAdapter)
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.plots import plot_poincare_map


class SynodicMap(_ReturnMapBase):
    """Driver for synodic section detection on precomputed trajectories.

    This facade mirrors the API of other return-map modules and caches a `_Section`.
    It does not propagate trajectories; callers supply them explicitly.
    """

    def __init__(self, map_cfg: Optional[_SynodicMapConfig] = None) -> None:
        cfg = map_cfg or _SynodicMapConfig()
        super().__init__(cfg)  # stores in self.config
        self._section_cfg = self._build_section_config(cfg)
        self._engine = self._build_engine()

    # These are unused but required by the abstract base; we do not build a propagation backend/strategy.
    def _build_backend(self, section_coord: str):
        raise NotImplementedError("SynodicMap does not use a propagation backend")

    def _build_seeding_strategy(self, section_coord: str):
        raise NotImplementedError("SynodicMap does not use a seeding strategy")

    def _build_section_config(self, cfg: _SynodicMapConfig) -> _SynodicSectionConfig:
        # Translate user-facing geometry fields into a cached section config
        if cfg.section_normal is not None:
            normal = np.asarray(cfg.section_normal, dtype=float)
        else:
            from hiten.algorithms.poincare.core.events import _PlaneEvent
            if isinstance(cfg.section_axis, str):
                idx = int(_PlaneEvent._IDX_MAP[cfg.section_axis.lower()])
            else:
                idx = int(cfg.section_axis)
            normal = np.zeros(6, dtype=float)
            normal[idx] = 1.0
        return _get_section_config(normal=normal, offset=cfg.section_offset, plane_coords=cfg.plane_coords)

    def _build_engine(self) -> _SynodicEngine:
        adapter = _SynodicEngineConfigAdapter(self.config)
        backend = _SynodicDetectionBackend(section_cfg=self._section_cfg, map_cfg=adapter._cfg)
        strategy = _NoOpStrategy(self._section_cfg, adapter)
        return _SynodicEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=adapter,
        )

    def from_trajectories(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: Literal[1, -1, None] = None,
        recompute: bool = False,
    ) -> _Section:
        sec = self._engine.set_trajectories(trajectories, direction=direction).compute_section(recompute=recompute)
        # Bridge into the base-class caching for consistent downstream APIs
        key = self._section_key()
        self._sections[key] = sec
        self._section = sec
        return sec

    def from_orbit(self, orbit: PeriodicOrbit, *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> _Section:
        if orbit.times is None or orbit.trajectory is None:
            raise ValueError("Orbit must be propagated before extracting trajectories")
        traj = [(np.asarray(orbit.times, dtype=np.float64), np.asarray(orbit.trajectory, dtype=np.float64))]
        return self.from_trajectories(traj, direction=direction, recompute=recompute)

    def from_manifold(self, manifold, *, direction: Literal[1, -1, None] = None, recompute: bool = False) -> _Section:
        manifold_result = manifold.manifold_result
        trajs = []
        for times, states in zip(getattr(manifold_result, "times_list", []), getattr(manifold_result, "states_list", [])):
            if times is None or states is None:
                continue
            t_arr = np.asarray(times, dtype=np.float64)
            x_arr = np.asarray(states, dtype=np.float64)
            if t_arr.ndim == 1 and x_arr.ndim == 2 and len(t_arr) == len(x_arr) and x_arr.shape[1] == 6:
                trajs.append((t_arr, x_arr))

        if not trajs:
            raise ValueError("Manifold result contains no valid trajectories")
        return self.from_trajectories(trajs, direction=direction, recompute=recompute)

    def _section_key(self) -> str:
        """Stable cache key mirroring CM's `section_coord` semantics.

        For synodic maps we derive a canonical name from the geometry.
        """
        n = self._section_cfg.normal
        n_key = ",".join(f"{float(v):.12g}" for v in n.tolist())
        c = float(self._section_cfg.offset)
        i, j = self._section_cfg.plane_coords
        return f"synodic[{i},{j}]_c={c:.12g}_n=({n_key})"

    def plot(
        self,
        *,
        axes: Sequence[str] | None = None,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        **kwargs,
    ):
        """Render a 2-D Poincaré map for the last computed synodic section.

        Requires that `from_orbit` or `from_manifold` has been
        called to populate the cached section.
        """
        if self._section is None:
            raise ValueError("No synodic section cached. Compute from orbit or manifold first.")

        section = self._section

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            # Build projection using either stored plane points or full states
            from hiten.algorithms.poincare.core.events import _PlaneEvent
            cols = []
            for ax in axes:
                if ax in section.labels:
                    idx = section.labels.index(ax)
                    cols.append(section.points[:, idx])
                else:
                    idx = int(_PlaneEvent._IDX_MAP[ax.lower()])
                    cols.append(section.states[:, idx])
            pts = np.column_stack(cols)
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )
