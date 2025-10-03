"""Engine classes for synodic Poincare section detection.

This module provides the engine classes that coordinate the detection
and refinement of synodic Poincare sections on precomputed trajectories.
It implements parallel processing capabilities for efficient batch
detection across multiple trajectories.

The implementation provides high-accuracy detection using advanced
numerical techniques including cubic Hermite interpolation and
Newton refinement for precise crossing detection.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.interfaces import _SynodicInterface
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.algorithms.poincare.synodic.types import (
    SynodicBackendRequest,
    SynodicMapResults,
    _SynodicMapProblem,
)
from hiten.algorithms.types.exceptions import EngineError


class _SynodicEngine(_ReturnMapEngine):
    """Engine for synodic Poincare section detection on precomputed trajectories.

    This engine coordinates the detection and refinement of synodic Poincare
    sections across multiple precomputed trajectories. It extends the base
    return map engine to provide specialized functionality for synodic sections
    while reusing the worker management and caching infrastructure.

    Parameters
    ----------
    backend : :class:`~hiten.algorithms.poincare.synodic.backend._SynodicDetectionBackend`
        The detection backend for synodic sections.
    seed_strategy : :class:`~hiten.algorithms.poincare.synodic.strategies._NoOpStrategy`
        The seeding strategy (no-op for synodic maps).
    map_config : :class:`~hiten.algorithms.poincare.synodic.interfaces._SynodicEngineInterface`
        The configuration adapter for the engine.

    Attributes
    ----------
    _trajectories : sequence of tuple[ndarray, ndarray] or None
        The precomputed trajectories to analyze.
    _direction : int or None
        The crossing direction filter for detection.

    Notes
    -----
    This engine provides parallel processing capabilities for efficient
    batch detection across multiple trajectories. It automatically
    chooses between serial and parallel processing based on the number
    of workers and trajectories.

    The engine caches computed sections to avoid redundant computation
    and provides a fluent interface for setting trajectories and
    computing sections.

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(
        self,
        *,
        backend: _SynodicDetectionBackend,
        seed_strategy: _NoOpStrategy,
        map_config,
        interface: _SynodicInterface,
    ) -> None:
        super().__init__(backend=backend, seed_strategy=seed_strategy, map_config=map_config, interface=interface)

    def solve(self, problem: _SynodicMapProblem) -> SynodicMapResults:
        """Compute the synodic Poincare section from the composed problem."""
        trajectories = problem.trajectories or []
        if not trajectories:
            raise EngineError("No trajectories provided to synodic engine")

        n_workers = problem.n_workers
        
        # Build template request from problem
        call = self._interface.to_backend_inputs(problem)
        template_request = call.request
        if not isinstance(template_request, SynodicBackendRequest):
            raise EngineError("Interface must return SynodicBackendRequest")

        # Delegate detection to backend passed in at construction
        if n_workers <= 1 or len(trajectories) <= 1:
            # For consistency, always pass trajectory indices even in single-worker mode
            request = SynodicBackendRequest(
                trajectories=trajectories,
                normal=template_request.normal,
                trajectory_indices=list(range(len(trajectories))),
                offset=template_request.offset,
                plane_coords=template_request.plane_coords,
                interp_kind=template_request.interp_kind,
                segment_refine=template_request.segment_refine,
                tol_on_surface=template_request.tol_on_surface,
                dedup_time_tol=template_request.dedup_time_tol,
                dedup_point_tol=template_request.dedup_point_tol,
                max_hits_per_traj=template_request.max_hits_per_traj,
                newton_max_iter=template_request.newton_max_iter,
                direction=template_request.direction,
            )
            response = self._backend.run(request)
            hits_lists = response.hits
        else:
            chunks = np.array_split(np.arange(len(trajectories)), n_workers)

            def _worker(idx_arr: np.ndarray):
                subset = [trajectories[i] for i in idx_arr.tolist()]
                # Pass the original trajectory indices so backend labels them correctly
                request = SynodicBackendRequest(
                    trajectories=subset,
                    normal=template_request.normal,
                    trajectory_indices=idx_arr.tolist(),
                    offset=template_request.offset,
                    plane_coords=template_request.plane_coords,
                    interp_kind=template_request.interp_kind,
                    segment_refine=template_request.segment_refine,
                    tol_on_surface=template_request.tol_on_surface,
                    dedup_time_tol=template_request.dedup_time_tol,
                    dedup_point_tol=template_request.dedup_point_tol,
                    max_hits_per_traj=template_request.max_hits_per_traj,
                    newton_max_iter=template_request.newton_max_iter,
                    direction=template_request.direction,
                )
                return self._backend.run(request).hits

            parts: list[list[list]] = []
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_worker, idxs) for idxs in chunks if len(idxs)]
                for fut in as_completed(futs):
                    parts.append(fut.result())
            hits_lists = [hits for part in parts for hits in part]

        pts, sts, ts, traj_indices = [], [], [], []
        for hits in hits_lists:
            for h in hits:
                pts.append(h.point2d)
                sts.append(h.state)
                ts.append(h.time)
                traj_indices.append(h.trajectory_index)

        pts_np = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
        sts_np = np.asarray(sts, dtype=float) if sts else np.empty((0, 6))
        ts_np = np.asarray(ts, dtype=float) if ts else None
        traj_indices_np = np.asarray(traj_indices, dtype=int) if traj_indices else np.empty((0,), dtype=int)

        # Create proper backend response with aggregated data
        from hiten.algorithms.poincare.synodic.types import SynodicBackendResponse
        response = SynodicBackendResponse(
            hits=hits_lists,  # Keep raw hits for potential debugging
            points=pts_np,
            states=sts_np,
            times=ts_np,
            trajectory_indices=traj_indices_np,
            metadata={"n_points": len(pts), "n_trajectories": len(set(traj_indices)) if traj_indices else 0}
        )
        
        return self._interface.to_results(response, problem=problem)