"""Center manifold Poincare map computation engine.

This module provides the computation engine for generating Poincare maps
restricted to center manifolds of collinear libration points in the Circular
Restricted Three-Body Problem (CR3BP). The engine coordinates the seeding
strategy, numerical integration, and parallel processing to efficiently
compute return maps.

The main class :class:`~hiten.algorithms.poincare.centermanifold.engine._CenterManifoldEngine` 
extends the base return map engine with center manifold-specific functionality and parallel 
processing capabilities.

References
----------
Szebehely, V. (1967). *Theory of Orbits*. Academic Press.

Jorba, A. & Masdemont, J. (1999). Dynamics in the center manifold
of the collinear points of the restricted three body problem.
*Physica D*, 132(1-2), 189-213.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from hiten.algorithms.poincare.centermanifold.backend import \
    _CenterManifoldBackend
from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.interfaces import \
    _CenterManifoldInterface
from hiten.algorithms.poincare.centermanifold.seeding import \
    _CenterManifoldSeedingBase
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldMapResults, _CenterManifoldMapProblem)
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.utils.exceptions import EngineError
from hiten.utils.log_config import logger


class _CenterManifoldEngine(_ReturnMapEngine):
    """Engine for center manifold Poincare map computation.

    This engine coordinates the computation of Poincare maps restricted to
    center manifolds in the CR3BP. It manages the seeding strategy, numerical
    integration, and parallel processing to efficiently generate return maps.

    Parameters
    ----------
    backend : :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend`
        Backend providing numerical integration and section crossing detection.
    seed_strategy : :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase`
        Strategy for generating initial conditions on the center manifold.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config._CenterManifoldMapConfig`
        Configuration specifying computation parameters.

    Notes
    -----
    The engine uses parallel processing to efficiently compute multiple
    trajectories and their intersections with the Poincare section. It
    iteratively applies the Poincare map to generate the return map data.

    The computation process:
    1. Generate initial seeds using the seeding strategy
    2. Lift plane points to center manifold states
    3. Iteratively apply the Poincare map using parallel workers
    4. Collect and combine results from all workers
    5. Cache the computed section for reuse

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    def __init__(
        self,
        backend: _CenterManifoldBackend,
        seed_strategy: _CenterManifoldSeedingBase,
        map_config: _CenterManifoldMapConfig,
        *,
        interface: _CenterManifoldInterface,
    ) -> None:
        super().__init__(backend, seed_strategy, map_config)
        self._interface = interface

    def solve(
        self,
        *,
        dt: float | None = None,
        n_iter: int | None = None,
        n_workers: int | None = None,
    ) -> CenterManifoldMapResults:
        """Compute the Poincare section for the center manifold.

        This method generates the Poincare map by iteratively applying the
        return map to initial seeds. It uses parallel processing to efficiently
        compute multiple trajectories and their intersections with the section.

        Parameters
        ----------
        recompute : bool, default=False
            If True, force recomputation even if cached results exist.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            Computed Poincare section containing points, states, and metadata.

        Raises
        ------
        RuntimeError
            If the seeding strategy produces no valid points inside the
            Hill boundary.

        Notes
        -----
        The computation process involves:
        1. Generating initial seeds using the configured seeding strategy
        2. Lifting plane points to center manifold states using the backend
        3. Iteratively applying the Poincare map using parallel workers
        4. Collecting and combining results from all workers
        5. Caching the computed section for future use

        The method uses ThreadPoolExecutor for parallel processing, with the
        number of workers determined by the configuration.
        """
        dt_ = float(self._dt if dt is None else dt)
        n_iter_ = int(self._n_iter if n_iter is None else n_iter)
        n_workers_ = int(self._n_workers if n_workers is None else n_workers)

        problem = _CenterManifoldMapProblem(
            section_coord=self._backend._section_cfg.section_coord,
            energy=self._backend._h0,
            dt=dt_,
            n_iter=n_iter_,
            n_workers=n_workers_,
        )

        logger.info("Generating Poincare map: seeds=%d, iterations=%d, workers=%d",
                    self._strategy.n_seeds, problem.n_iter, problem.n_workers)

        # Provide interface-bound helpers matching strategy signatures
        solve_missing_coord_fn = lambda varname, fixed_vals: self._interface.solve_missing_coord(  # noqa: E731
            varname,
            fixed_vals,
            h0=self._backend._h0,
            H_blocks=self._backend._H_blocks,
            clmo_table=self._backend._clmo_table,
        )
        find_turning_fn = lambda name: self._interface.find_turning(  # noqa: E731
            name,
            h0=self._backend._h0,
            H_blocks=self._backend._H_blocks,
            clmo_table=self._backend._clmo_table,
        )

        plane_pts = self._strategy.generate(
            h0=self._backend._h0,
            H_blocks=self._backend._H_blocks,
            clmo_table=self._backend._clmo_table,
            solve_missing_coord_fn=solve_missing_coord_fn,
            find_turning_fn=find_turning_fn,
        )

        section_coord = self._backend._section_cfg.section_coord
        seeds0 = [
            self._interface.lift_plane_point(
                p,
                section_coord=section_coord,
                h0=self._backend._h0,
                H_blocks=self._backend._H_blocks,
                clmo_table=self._backend._clmo_table,
            )
            for p in plane_pts
        ]
        seeds0 = np.asarray([s for s in seeds0 if s is not None], dtype=np.float64)

        if seeds0.size == 0:
            raise EngineError("Seed strategy produced no valid points inside Hill boundary")

        chunks = np.array_split(seeds0, problem.n_workers)

        def _worker(chunk: np.ndarray):
            pts_accum, states_accum, times_accum = [], [], []
            seeds = chunk
            for it in range(problem.n_iter):
                # Hook: iteration start
                try:
                    self._backend.on_iteration(it, seeds)
                except Exception:
                    pass

                pts, states, times, flags = self._backend.step_to_section(seeds, dt=problem.dt)
                if pts.size == 0:
                    try:
                        self._backend.on_failure(it)
                    except Exception:
                        pass
                    break

                try:
                    self._backend.on_success(it, pts, states, times)
                except Exception:
                    pass

                pts_accum.append(pts)
                states_accum.append(states)
                times_accum.append(times)
                seeds = states  # feed back
            if pts_accum:
                return np.vstack(pts_accum), np.vstack(states_accum), np.concatenate(times_accum)
            return np.empty((0, 2)), np.empty((0, 4)), np.empty((0,))

        pts_list, states_list, times_list = [], [], []
        with ThreadPoolExecutor(max_workers=problem.n_workers) as executor:
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

        try:
            self._backend.on_accept(pts_np, cms_np, times_np)
        except Exception:
            pass

        return CenterManifoldMapResults(
            pts_np,
            cms_np,
            self._backend._section_cfg.plane_coords,
            times_np,
        )

