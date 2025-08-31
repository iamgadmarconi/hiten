from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from hiten.algorithms.poincare.centermanifold.backend import \
    _CenterManifoldBackend
from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.seeding import \
    _CenterManifoldSeedingBase
from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.utils.log_config import logger


class _CenterManifoldEngine(_ReturnMapEngine):
    """Driver for centre-manifold return-map generation."""

    def __init__(
        self,
        backend: _CenterManifoldBackend,
        seed_strategy: _CenterManifoldSeedingBase,
        map_config: _CenterManifoldMapConfig,
    ) -> None:
        super().__init__(backend, seed_strategy, map_config)

    def compute_section(self, *, recompute: bool = False) -> _Section:
        if self._section_cache is not None and not recompute:
            return self._section_cache

        logger.info("Generating Poincaré map: seeds=%d, iterations=%d, workers=%d",
                    self._strategy.n_seeds, self._n_iter, self._n_workers)

        plane_pts = self._strategy.generate(
            h0=self._backend._h0,
            H_blocks=self._backend._H_blocks,
            clmo_table=self._backend._clmo_table,
            solve_missing_coord_fn=self._backend._solve_missing_coord,
            find_turning_fn=self._backend._find_turning,
        )

        seeds0 = [self._backend._lift_plane_point(p) for p in plane_pts]
        seeds0 = np.asarray([s for s in seeds0 if s is not None], dtype=np.float64)

        if seeds0.size == 0:
            raise RuntimeError("Seed strategy produced no valid points inside Hill boundary")

        chunks = np.array_split(seeds0, self._n_workers)

        def _worker(chunk: np.ndarray):
            pts_accum, states_accum, times_accum = [], [], []
            seeds = chunk
            for _ in range(self._n_iter):
                pts, states, times, flags = self._backend.step_to_section(seeds, dt=self._dt)
                if pts.size == 0:
                    break
                pts_accum.append(pts)
                states_accum.append(states)
                times_accum.append(times)
                seeds = states  # feed back
            if pts_accum:
                return np.vstack(pts_accum), np.vstack(states_accum), np.concatenate(times_accum)
            return np.empty((0, 2)), np.empty((0, 4)), np.empty((0,))

        pts_list, states_list, times_list = [], [], []
        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            futures = [executor.submit(_worker, c) for c in chunks if c.size]
            for fut in as_completed(futures):
                p, s, t = fut.result()
                if p.size:
                    pts_list.append(p)
                    states_list.append(s)
                    times_list.append(t)

        pts_np = np.vstack(pts_list) if pts_list else np.empty((0, 2))
        cms_np = np.vstack(states_list) if states_list else np.empty((0, 4))
        times_np = np.concatenate(times_list) if times_list else None

        self._section_cache = _Section(
            pts_np, cms_np, self._backend._section_cfg.plane_coords, times_np
        )
        return self._section_cache

