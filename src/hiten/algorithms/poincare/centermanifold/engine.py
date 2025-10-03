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

from hiten.algorithms.poincare.centermanifold.backend import (
    _CenterManifoldBackend,
)
from hiten.algorithms.poincare.centermanifold.config import (
    CenterManifoldMapConfig,
)
from hiten.algorithms.poincare.centermanifold.interfaces import (
    _CenterManifoldInterface,
)
from hiten.algorithms.poincare.centermanifold.seeding import (
    _CenterManifoldSeedingBase,
)
from hiten.algorithms.poincare.centermanifold.types import (
    CenterManifoldBackendRequest,
    CenterManifoldBackendResponse,
    CenterManifoldMapResults,
    _CenterManifoldMapProblem,
)
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.types.core import _BackendCall
from hiten.algorithms.types.exceptions import EngineError
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
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
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
        *,
        backend: _CenterManifoldBackend,
        seed_strategy: _CenterManifoldSeedingBase,
        map_config: CenterManifoldMapConfig,
        interface: _CenterManifoldInterface,
    ) -> None:
        super().__init__(backend=backend, seed_strategy=seed_strategy, map_config=map_config, interface=interface)

    def solve(self, problem: _CenterManifoldMapProblem) -> CenterManifoldMapResults:
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
        logger.info("Generating Poincare map: seeds=%d, iterations=%d, workers=%d",
                    self._strategy.n_seeds, problem.n_iter, problem.n_workers)

        plane_pts = self._strategy.generate(
            h0=problem.energy,
            H_blocks=problem.H_blocks,
            clmo_table=problem.clmo_table,
            solve_missing_coord_fn=lambda varname, fixed_vals: problem.solve_missing_coord_fn(varname, fixed_vals),
            find_turning_fn=lambda name: problem.find_turning_fn(name),
        )

        section_coord = problem.section_coord
        seeds0 = [
            self._interface.lift_plane_point(
                p,
                section_coord=section_coord,
                h0=problem.energy,
                H_blocks=problem.H_blocks,
                clmo_table=problem.clmo_table,
            )
            for p in plane_pts
        ]
        seeds0 = np.asarray([s for s in seeds0 if s is not None], dtype=np.float64)

        if seeds0.size == 0:
            raise EngineError("Seed strategy produced no valid points inside Hill boundary")

        n_workers_eff = max(1, int(problem.n_workers))
        chunks = np.array_split(seeds0, n_workers_eff)

        call = self._interface.to_backend_inputs(problem)
        template_request = call.request
        if not isinstance(template_request, CenterManifoldBackendRequest):
            raise EngineError("Interface must return CenterManifoldBackendRequest")

        def _worker(chunk: np.ndarray):
            states_accum, times_accum = [], []
            seeds = chunk
            for it in range(problem.n_iter):
                # Clone and populate request for this iteration
                request = CenterManifoldBackendRequest(
                    seeds=seeds,
                    dt=template_request.dt,
                    jac_H=template_request.jac_H,
                    clmo_table=template_request.clmo_table,
                    section_coord=template_request.section_coord,
                    forward=template_request.forward,
                    max_steps=template_request.max_steps,
                    method=template_request.method,
                    order=template_request.order,
                    c_omega_heuristic=template_request.c_omega_heuristic,
                    metadata={"iteration": it},
                )
                response = self._backend.run(request)
                if response.states.size == 0:
                    break

                states = self._interface.enforce_section_coordinate(response.states, section_coord=section_coord)
                states_accum.append(states)
                times_accum.append(response.times)
                seeds = states  # feed back
            if states_accum:
                return np.vstack(states_accum), np.concatenate(times_accum)
            return np.empty((0, 4)), np.empty((0,))

        states_list, times_list = [], []
        with ThreadPoolExecutor(max_workers=n_workers_eff) as executor:
            futures = [executor.submit(_worker, c) for c in chunks if c.size]
            for fut in as_completed(futures):
                s, t = fut.result()
                if s.size:
                    states_list.append(s)
                    times_list.append(t)

        cms_np = np.vstack(states_list) if states_list else np.empty((0, 4))
        times_np = np.concatenate(times_list) if times_list else None

        cms_np = self._interface.enforce_section_coordinate(cms_np, section_coord=section_coord)
        pts_np = self._interface.plane_points_from_states(cms_np, section_coord=section_coord)

        # Create proper backend response
        response = CenterManifoldBackendResponse(
            states=cms_np,
            times=times_np,
            flags=np.zeros(len(cms_np), dtype=int),  # No flags in iterative solve
            metadata={"n_workers": n_workers_eff, "n_iterations": problem.n_iter}
        )
        
        return self._interface.to_results(
            response,
            problem=problem,
        )

