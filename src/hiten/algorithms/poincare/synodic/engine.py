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
from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.interfaces import _SynodicEngineInterface
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.algorithms.poincare.synodic.types import (SynodicMapResults,
                                                     _SynodicMapProblem)





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
        backend: _SynodicDetectionBackend,
        seed_strategy: _NoOpStrategy,
        map_config: _SynodicEngineInterface,
    ) -> None:
        super().__init__(backend, seed_strategy, map_config)
        self._trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]" | None = None
        self._direction: int | None = None

    def set_trajectories(
        self,
        trajectories: "Sequence[tuple[np.ndarray, np.ndarray]]",
        *,
        direction: Literal[1, -1, None] | None = None,
    ) -> "_SynodicEngine":
        """Set the trajectories to analyze and return self for chaining.

        Parameters
        ----------
        trajectories : sequence of tuple[ndarray, ndarray]
            Sequence of (times, states) tuples for each trajectory.
            Each tuple contains:
            - times: ndarray, shape (n,) - Time points (nondimensional units)
            - states: ndarray, shape (n, 6) - State vectors at each time point
        direction : {1, -1, None}, optional
            Crossing direction filter. If None, uses the default
            direction from the section configuration.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.engine._SynodicEngine`
            Self for method chaining.

        Notes
        -----
        This method sets the trajectories to analyze and clears any
        cached results. It provides a fluent interface for chaining
        method calls.

        The method automatically clears the section cache when new
        trajectories are set to ensure fresh computation.
        """
        self._trajectories = trajectories
        self._direction = direction
        self.clear_cache()
        return self

    def solve(self, problem: _SynodicMapProblem) -> SynodicMapResults:
        """Compute the synodic Poincare section from the composed problem."""
        self.set_trajectories(problem.trajectories or [], direction=problem.direction)
        if self._section_cache is not None:
            return self._section_cache

        if self._trajectories is None:
            raise ValueError("No trajectories set. Call set_trajectories(...) first.")

        # Keep local aliases for clarity
        plane_coords = self._backend.plane_coords
        n_workers = self._n_workers

        # Delegate detection to backend passed in at construction
        if n_workers <= 1 or len(self._trajectories) <= 1:  # type: ignore[arg-type]
            hits_lists = self._backend.detect_batch(self._trajectories, direction=self._direction)  # type: ignore[arg-type]
        else:
            chunks = np.array_split(np.arange(len(self._trajectories)), n_workers)  # type: ignore[arg-type]

            def _worker(idx_arr: np.ndarray):
                subset = [self._trajectories[i] for i in idx_arr.tolist()]  # type: ignore[index]
                return self._backend.detect_batch(subset, direction=self._direction)

            parts: list[list[list]] = []
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futs = [ex.submit(_worker, idxs) for idxs in chunks if len(idxs)]
                for fut in as_completed(futs):
                    parts.append(fut.result())
            hits_lists = [hits for part in parts for hits in part]

        pts, sts, ts = [], [], []
        for hits in hits_lists:
            for h in hits:
                pts.append(h.point2d)
                sts.append(h.state)
                ts.append(h.time)

        pts_np = np.asarray(pts, dtype=float) if pts else np.empty((0, 2))
        sts_np = np.asarray(sts, dtype=float) if sts else np.empty((0, 6))
        ts_np = np.asarray(ts, dtype=float) if ts else None

        labels = plane_coords
        self._section_cache = SynodicMapResults(pts_np, sts_np, labels, ts_np)
        return self._section_cache