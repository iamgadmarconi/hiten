from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
from hiten.algorithms.poincare.backends import _ReturnMapBackend
from hiten.algorithms.poincare.crossing import find_crossing
from hiten.algorithms.poincare.events import _SectionHit, _SurfaceEvent
from hiten.algorithms.poincare.seeding.generators import _SeedGenerator
from hiten.utils.log_config import logger


class _Section:
    def __init__(self, pts, st):
        self.points = pts
        self.states = st
        self.labels = ("coord1", "coord2")

class _ReturnMapEngine(_ReturnMapBackend):
    """Generic CPU return-map builder (implements backend interface)."""

    def __init__(
        self,
        *,
        dynsys: _DynamicalSystemProtocol,
        surface: _SurfaceEvent,
        seeder: _SeedGenerator,
        n_seeds: int = 20,
        n_iter: int = 40,
        forward: int = 1,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
        order: int = 4,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
    ) -> None:
        self._dynsys = dynsys
        self._surface = surface
        self._seeder = seeder
        self._n_seeds = int(n_seeds)
        self._n_iter = int(n_iter)
        self._forward = 1 if forward >= 0 else -1

        # numeric controls
        self._method = method
        self._order = int(order)
        self._pre_steps = int(pre_steps)
        self._refine_steps = int(refine_steps)
        self._bracket_dx = float(bracket_dx)
        self._max_expand = int(max_expand)

        # storage
        self._hits: list[_SectionHit] | None = None
        self._seed_states: list[np.ndarray] | None = None

    def compute(self, *, recompute: bool = False) -> Sequence[_SectionHit]:
        if self._hits is not None and not recompute:
            return self._hits

        logger.info(
            "ReturnMapEngine: generating %d seeds via %s", self._n_seeds, self._seeder.__class__.__name__
        )
        seeds = self._seeder.generate(
            dynsys=self._dynsys,
            surface=self._surface,
            n_seeds=self._n_seeds,
        )
        self._seed_states = list(seeds)

        hits: list[_SectionHit] = []
        for k, seed in enumerate(seeds):
            state_curr = seed.copy()
            for i in range(self._n_iter):
                try:
                    hit = find_crossing(
                        self._dynsys,
                        state_curr,
                        self._surface,
                        forward=self._forward,
                        pre_steps=self._pre_steps,
                        refine_steps=self._refine_steps,
                        bracket_dx=self._bracket_dx,
                        max_expand=self._max_expand,
                        method=self._method,
                        order=self._order,
                    )
                except Exception as exc:
                    logger.debug("Seed %d iteration %d failed: %s", k, i, exc)
                    break  # stop iterating this seed
                hits.append(hit)
                state_curr = hit.state  # next iteration starts from section point
        self._hits = hits
        logger.info("ReturnMapEngine: collected %d section hits", len(hits))
        return hits

    def points2d(self) -> np.ndarray:
        if self._hits is None:
            self.compute()
        return np.vstack([h.point2d for h in self._hits]) if self._hits else np.empty((0, 2))

    def states(self) -> np.ndarray:
        if self._hits is None:
            self.compute()
        return np.vstack([h.state for h in self._hits]) if self._hits else np.empty((0, self._dynsys.dim))

    def times(self) -> np.ndarray:
        if self._hits is None:
            self.compute()
        return np.asarray([h.time for h in self._hits]) if self._hits else np.empty(0)

    def __len__(self) -> int:
        return 0 if self._hits is None else len(self._hits)

    def __iter__(self):
        if self._hits is None:
            self.compute()
        return iter(self._hits)

    def compute_section(self, *, recompute: bool = False):
        """Return section-like object with points and states arrays."""
        if self._hits is None or recompute:
            self.compute(recompute=recompute)

        points = self.points2d()
        states = self.states()
        return _Section(points, states)

    def compute_grid(self, *, recompute: bool = False):
        raise NotImplementedError("Grid generation not supported by engine backend yet")
