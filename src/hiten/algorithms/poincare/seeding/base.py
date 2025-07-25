r"""
hiten.algorithms.poincare.seeding.base
====================================

Base class for Poincaré section seeding strategies.

The module exposes a base class :pyclass:`_CenterManifoldSeedingBase` that defines the
interface for all seeding strategies.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

if TYPE_CHECKING:
    from hiten.algorithms.poincare.config import _CenterManifoldSectionConfig


class _CenterManifoldSeedingBase(ABC):
    def __init__(self, section_config: "_CenterManifoldSectionConfig", n_seeds: int = 20) -> None:
        self._cfg = section_config
        self.n_seeds = n_seeds

    @property
    def config(self) -> "_CenterManifoldSectionConfig":
        return self._cfg

    @property
    def plane_coords(self) -> Tuple[str, str]:
        return self._cfg.plane_coords

    @abstractmethod
    def generate(self, *, h0: float, H_blocks: Any, clmo_table: Any, solve_missing_coord_fn: Any, find_turning_fn: Any) -> List[Tuple[float, float, float, float]]:
        pass

    def __call__(self, **kwargs):
        return self.generate(**kwargs)

    _cached_limits: dict[tuple[float, int], list[float]] = {}

    def _hill_boundary_limits(
        self,
        *,
        h0: float,
        H_blocks: Any,
        clmo_table: Any,
        find_turning_fn: Callable
    ) -> list[float]:
        """Return turning-point limits (max absolute) for the two plane coords.

        Results are cached per energy level to avoid recomputing when multiple
        strategies are used with identical parameters.
        """
        key = (float(h0), id(H_blocks))
        if key in self._cached_limits:
            return self._cached_limits[key]

        # The backend-specific *find_turning_fn* already carries everything it
        # needs (energy level, Hamiltonian blocks, etc.) via closure/bound
        # attributes, therefore we only pass the coordinate identifier.
        limits = [find_turning_fn(c) for c in self.plane_coords]
        self._cached_limits[key] = limits
        return limits

    def _build_seed(
        self,
        plane_vals: tuple[float, float],
        *,
        solve_missing_coord_fn,
    ) -> tuple[float, float] | None:
        """Validate *plane_vals* against the Hill boundary.

        The method **no longer** returns a full 4-tuple centre-manifold state.
        Instead it merely checks that the 2-D point lies inside the allowed
        region (by solving for the missing coordinate).  On success the same
        *plane_vals* tuple is returned; *None* indicates the point is outside
        the Hill region.
        """

        cfg = self.config

        constraints = cfg.build_constraint_dict(**{
            cfg.plane_coords[0]: plane_vals[0],
            cfg.plane_coords[1]: plane_vals[1],
        })

        # `_solve_missing_coord` (backend implementation) needs only the
        # variable name and the dict of fixed values; other parameters are
        # already bound via the backend instance.  Passing extra positional
        # arguments corrupts its `initial_guess`, `expand_factor`, etc.
        missing_val = solve_missing_coord_fn(cfg.missing_coord, constraints)

        if missing_val is None:
            # Point lies outside Hill boundary.
            return None

        return plane_vals
