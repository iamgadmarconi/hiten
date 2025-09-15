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

from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy


class _SynodicEngineConfigAdapter:
    """Configuration adapter for synodic Poincare engine.

    This adapter class provides the interface expected by the base
    return map engine while adapting the synodic map configuration
    to the required format. It handles the translation between
    synodic-specific parameters and the generic engine interface.

    Parameters
    ----------
    cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The synodic map configuration to adapt.

    Attributes
    ----------
    _cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The original synodic map configuration.
    dt : float
        Time step (set to 0.0 for synodic maps since they use precomputed trajectories).
    n_iter : int
        Number of iterations (set to 1 for synodic maps).
    n_workers : int
        Number of parallel workers for batch processing.
    n_seeds : int
        Number of seeds (set to 0 for synodic maps since they use precomputed trajectories).

    Notes
    -----
    This adapter is necessary because synodic Poincare maps operate on
    precomputed trajectories rather than integrating from initial conditions.
    The adapter provides the interface expected by the base engine while
    setting appropriate values for the synodic use case.

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self, cfg: _SynodicMapConfig) -> None:
        self._cfg = cfg
        self.dt = 0.0
        self.n_iter = 1
        self.n_workers = cfg.n_workers
        # Satisfy _SeedingConfigLike for the no-op strategy
        self.n_seeds = 0

    def __repr__(self) -> str:
        return f"SynodicEngineConfigAdapter(n_workers={self.n_workers})"


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
    map_config : :class:`~hiten.algorithms.poincare.synodic.engine._SynodicEngineConfigAdapter`
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
        map_config: _SynodicEngineConfigAdapter,
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

    def compute_section(self, *, recompute: bool = False) -> _Section:
        """Compute the synodic Poincare section from the set trajectories.

        Parameters
        ----------
        recompute : bool, default False
            Whether to recompute the section even if it's already cached.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.core.base._Section`
            The computed synodic Poincare section containing:
            - points: 2D projected coordinates
            - states: Full 6D state vectors
            - times: Crossing times
            - labels: Coordinate labels for the projection

        Raises
        ------
        ValueError
            If no trajectories have been set.

        Notes
        -----
        This method computes the synodic Poincare section from the
        previously set trajectories. It uses parallel processing when
        beneficial and caches the result for future use.

        The method automatically chooses between serial and parallel
        processing based on the number of workers and trajectories.
        For small trajectory sets or single-worker configurations,
        it uses serial processing for efficiency.

        The computed section is cached and returned on subsequent
        calls unless recompute is True.

        All time units are in nondimensional units.
        """  
        if self._section_cache is not None and not recompute:
            return self._section_cache

        if self._trajectories is None:
            raise ValueError("No trajectories set. Call set_trajectories(...) first.")

        # Delegate detection to backend passed in at construction
        if self._n_workers <= 1 or len(self._trajectories) <= 1:
            hits_lists = self._backend.detect_batch(self._trajectories, direction=self._direction)
        else:
            chunks = np.array_split(np.arange(len(self._trajectories)), self._n_workers)

            def _worker(idx_arr: np.ndarray):
                subset = [self._trajectories[i] for i in idx_arr.tolist()]
                return self._backend.detect_batch(subset, direction=self._direction)

            parts: list[list[list]] = []
            with ThreadPoolExecutor(max_workers=self._n_workers) as ex:
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

        labels = self._backend.plane_coords
        self._section_cache = _Section(pts_np, sts_np, labels, ts_np)
        return self._section_cache
