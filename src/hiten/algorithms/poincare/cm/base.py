from typing import Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.cm import _make_strategy
from hiten.algorithms.poincare.cm.config import (_CenterManifoldMapConfig,
                                                 _get_section_config)
from hiten.algorithms.poincare.cm.engine import _CenterManifoldEngine
from hiten.algorithms.poincare.core.base import _ReturnMapBase
from hiten.system.center import CenterManifold
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io.map import (load_poincare_map, load_poincare_map_inplace,
                                save_poincare_map)
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive


class CenterManifoldMap(_ReturnMapBase):
    """Poincaré return map restricted to the centre manifold of a collinear L-point."""

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[_CenterManifoldMapConfig] = None,
    ) -> None:
        self.cm: CenterManifold = cm
        self._energy: float = float(energy)

        # If caller does not supply a config, fall back to defaults.
        cfg = config or _CenterManifoldMapConfig()

        super().__init__(cfg)

    @property
    def energy(self) -> float:
        return self._energy

    def _build_backend(self, section_coord: str):
        """Return (and cache) a CM backend via the owning `CenterManifold`."""

        return self.cm._get_or_create_backend(
            self._energy,
            section_coord,
            method=self.config.method,
            order=self.config.order,
            c_omega_heuristic=self.config.c_omega_heuristic,
        )

    def _build_seeding_strategy(self, section_coord: str):
        """Return a seeding strategy configured for *section_coord*."""

        sec_cfg = _get_section_config(section_coord)

        strategy_kwargs: dict[str, object] = {}
        if self.config.seed_strategy == "single":
            strategy_kwargs["seed_axis"] = self.config.seed_axis

        strategy = _make_strategy(
            self.config.seed_strategy,
            sec_cfg,
            self.config,
            **strategy_kwargs,
        )

        return strategy

    def _build_engine(self, backend, strategy):
        """Instantiate the concrete engine for centre-manifold maps."""
        return _CenterManifoldEngine(
            backend=backend,
            seed_strategy=strategy,
            map_config=self.config,
        )

    def ic(self, pt: np.ndarray, *, section_coord: str | None = None) -> np.ndarray:
        key = section_coord or self.config.section_coord
        return self.cm.ic(pt, self._energy, section_coord=key)

    def _propagate_from_point(
        self,
        cm_point,
        energy,
        *,
        steps=1000,
        method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy",
        order=6,
    ):
        ic = self.cm.ic(cm_point, energy, section_coord=self.config.section_coord)
        logger.info("Initial conditions: %s", ic)
        orbit = GenericOrbit(self.cm.point, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    def plot(
        self,
        section_coord: str | None = None,
        *,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        axes: Sequence[str] | None = None,
        **kwargs,
    ):
        # Determine section
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self.compute(section_coord=section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self.compute()
            section = self._section

        # Decide projection
        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )

    def plot_interactive(
        self,
        *,
        steps=1000,
        method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy",
        order=6,
        frame="rotating",
        dark_mode: bool = True,
        axes: Sequence[str] | None = None,
        section_coord: str | None = None,
    ):
        # Ensure section exists
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self.compute(section_coord=section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self.compute()
            section = self._section

        def _on_select(pt_np: np.ndarray):
            if axes is None:
                section_pt = pt_np
            else:
                prev_sec = self._section
                self._section = section
                try:
                    proj_pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
                finally:
                    self._section = prev_sec
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                section_pt = section.points[np.argmin(distances)]

            orbit = self._propagate_from_point(
                section_pt,
                self.energy,
                steps=steps,
                method=method,
                order=order,
            )

            orbit.plot(frame=frame, dark_mode=dark_mode, block=False, close_after=False)

            return orbit

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map_interactive(
            points=pts,
            labels=lbls,
            on_select=_on_select,
            dark_mode=dark_mode,
        )

    def get_points(
        self,
        *,
        section_coord: str | None = None,
        axes: tuple[str, str] | None = None,
    ) -> np.ndarray:
        """Return 2-D projection, allowing any CM axis combination.

        The base implementation only knows about the two coordinates that span
        the section plane.  Here we extend it to permit projections mixing the
        plane coordinates with the *missing* coordinate (solved on the
        section) by falling back to the stored 4-D centre-manifold states.
        """

        if axes is None:
            return super().get_points(section_coord=section_coord)

        key = section_coord or self.config.section_coord

        # Compute on-demand if missing
        if key not in self._sections:
            self.compute(section_coord=key)

        sec = self._sections[key]

        # Mapping for full 4-D CM state stored in `sec.states`
        state_map = {"q2": 0, "p2": 1, "q3": 2, "p3": 3}

        cols = []
        for ax in axes:
            if ax in sec.labels:
                idx = sec.labels.index(ax)
                cols.append(sec.points[:, idx])
            elif ax in state_map:
                cols.append(sec.states[:, state_map[ax]])
            else:
                raise ValueError(
                    f"Axis '{ax}' not recognised; allowed are q2, p2, q3, p3"
                )

        # Stack the two 1-D arrays column-wise into shape (n, 2)
        return np.column_stack(cols)

    def save(self, filepath: str, **kwargs) -> None:
        save_poincare_map(self, filepath, **kwargs)

    def load_inplace(self, filepath: str, **kwargs) -> None:
        load_poincare_map_inplace(self, filepath, **kwargs)

    @classmethod
    def load(
        cls,
        filepath: str,
        cm: CenterManifold,
        **kwargs,
    ) -> "CenterManifoldMap":
        return load_poincare_map(filepath, cm, **kwargs)
