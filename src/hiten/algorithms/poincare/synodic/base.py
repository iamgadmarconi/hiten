"""User-facing interface for synodic Poincare sections.

This module provides the main `SynodicMap` class that serves as the
user-facing interface for synodic Poincare section detection on
precomputed trajectories. It implements a facade pattern that mirrors
the API of other return-map modules while providing specialized
functionality for synodic sections.

The main class :class:`~hiten.algorithms.poincare.synodic.base.SynodicMap` extends the abstract base class
to provide detection capabilities on precomputed trajectory data,
including support for orbits, manifolds, and custom trajectories.

"""

from typing import Generic, Optional, Literal, Sequence, Tuple

from hiten.algorithms.poincare.synodic.config import _SynodicMapConfig
from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
from hiten.algorithms.poincare.synodic.interfaces import _SynodicInterface
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.algorithms.poincare.synodic.types import SynodicMapResults
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBaseFacade)


class SynodicMapPipeline(_HitenBaseFacade, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """User-facing interface for synodic Poincare section detection.

    This class provides a facade that mirrors the API of other return-map
    modules while specializing in synodic Poincare section detection on
    precomputed trajectories. It does not propagate trajectories; callers
    supply them explicitly through various input methods.

    Parameters
    ----------
    map_cfg : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`, optional
        Configuration object containing detection parameters, section geometry,
        and refinement settings. If None, uses default configuration.

    Attributes
    ----------
    config : :class:`~hiten.algorithms.poincare.synodic.config._SynodicMapConfig`
        The map configuration object.
    _engine : :class:`~hiten.algorithms.poincare.synodic.engine._SynodicEngine`
        The engine that coordinates detection and refinement.
    _sections : dict[str, :class:`~hiten.algorithms.poincare.core.base._Section`]
        Cache of computed sections keyed by section parameters.
    _section : :class:`~hiten.algorithms.poincare.core.base._Section` or None
        The most recently computed section.

    Notes
    -----
    This class implements a facade pattern that provides a consistent
    interface for synodic Poincare section detection while hiding the
    complexity of the underlying detection and refinement algorithms.

    The class supports multiple input methods:
    - Custom trajectories via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_trajectories`
    - Periodic orbits via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_orbit`
    - Manifold structures via :meth:`~hiten.algorithms.poincare.synodic.base.SynodicMap.from_manifold`

    All time units are in nondimensional units unless otherwise specified.
    """

    def __init__(self, config: _SynodicMapConfig, interface: _SynodicInterface, engine: _SynodicEngine | None = None) -> None:
        super().__init__(config, interface, engine)

    @classmethod
    def with_default_engine(
        cls,
        config: _SynodicMapConfig,
        interface: Optional[_SynodicInterface] = None,
    ) -> "SynodicMapPipeline":
        """Construct a map with a default-wired engine injected.

        This mirrors the DI-friendly facades (e.g., ConnectionPipeline) by creating
        a default engine using the current configuration and injecting it.
        The engine is wired for the default section coordinate in the config.
        """
        from hiten.algorithms.poincare.synodic.backend import \
            _SynodicDetectionBackend
        from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
        from hiten.algorithms.poincare.synodic.interfaces import \
            _SynodicInterface

        backend = _SynodicDetectionBackend()
        map_intf = interface or _SynodicInterface()
        strategy = _NoOpStrategy(config)
        engine = _SynodicEngine(backend=backend, seed_strategy=strategy, map_config=config, interface=map_intf)
        return cls(config, map_intf, engine)

    def generate(
        self,
        domain_obj: DomainT,
        override: bool = False,
        *,
        section_axis: str | int | None = None,
        section_offset: float = None,
        section_normal: Sequence[float] | None = None,
        plane_coords: Tuple[str, str] = None,
        interp_kind: Literal["linear", "cubic"] = None,
        newton_max_iter: int = 4,
        dedup_point_tol: float = None,
        max_hits_per_traj: int | None = None,
        segment_refine: int = None,
        tol_on_surface: float = None,
        dedup_time_tol: float = None,   
        n_workers: int | None = None,
        direction: Literal[1, -1, None] | None = None,
    ) -> SynodicMapResults:
        """Compute the section, supporting runtime overrides without mutating config.

        If no overrides are provided, this defers to the cached, default setup
        and persists the result. If any overrides are provided, a temporary
        engine is assembled for this call and the result is returned without
        polluting the persistent cache. In all cases, this method returns the
        2-D points of the section.
        """
        kwargs = {
            "section_axis": section_axis,
            "section_offset": section_offset,
            "section_normal": section_normal,
            "plane_coords": plane_coords,
            "interp_kind": interp_kind,
            "newton_max_iter": newton_max_iter,
            "dedup_point_tol": dedup_point_tol,
            "max_hits_per_traj": max_hits_per_traj,
            "segment_refine": segment_refine,
            "tol_on_surface": tol_on_surface,
            "dedup_time_tol": dedup_time_tol,
            "n_workers": n_workers,
            "direction": direction,
        }
        problem = self._create_problem(domain_obj=domain_obj, override=override, **kwargs)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results
