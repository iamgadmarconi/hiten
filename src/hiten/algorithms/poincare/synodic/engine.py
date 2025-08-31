from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.core.engine import _ReturnMapEngine
from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy


class _SynodicEngineConfigAdapter:
    """Adapter providing dt/n_iter expected by the base engine."""

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
    """Engine for synodic section detection on precomputed trajectories.

    Subclasses the generic engine to reuse worker/count plumbing and caching.
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
        self._trajectories = trajectories
        self._direction = direction
        self.clear_cache()
        return self

    def compute_section(self, *, recompute: bool = False) -> _Section:  
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
