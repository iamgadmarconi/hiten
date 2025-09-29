"""
Center manifold Poincare map interface for the CR3BP.

This module provides the main user-facing interface for computing and
analyzing Poincare maps restricted to center manifolds of collinear
libration points in the Circular Restricted Three-Body Problem (CR3BP).

The :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap` 
class extends the base return map functionality with center manifold-specific seeding 
strategies and visualization capabilities.
"""

from __future__ import annotations

from typing import Generic, Literal, Optional

from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.engine import \
    _CenterManifoldEngine
from hiten.algorithms.poincare.centermanifold.interfaces import \
    _CenterManifoldInterface
from hiten.algorithms.poincare.centermanifold.seeding import \
    _CenterManifoldSeedingBase
from hiten.algorithms.poincare.centermanifold.strategies import _make_strategy
from hiten.algorithms.poincare.centermanifold.types import \
    CenterManifoldMapResults
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBaseFacade)


class CenterManifoldMapPipeline(_HitenBaseFacade, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """Poincare return map restricted to the center manifold of a collinear libration point.

    This class provides the main interface for computing and analyzing Poincare
    maps on center manifolds in the CR3BP. It supports various seeding strategies
    and provides visualization capabilities for understanding the local dynamics.

    Notes
    -----
    State vectors are ordered as [q1, q2, q3, p1, p2, p3] where q1=0 for
    center manifold trajectories. All coordinates are in nondimensional units
    with the primary-secondary separation as the length unit.

    Examples
    --------
    >>> from hiten.system.center import CenterManifold
    >>> from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
    >>> 
    >>> # Create center manifold for L1 point
    >>> cm = CenterManifold("L1")
    >>> 
    >>> # Create Poincare map at specific energy
    >>> energy = -1.5
    >>> poincare_map = CenterManifoldMap(cm, energy)
    >>> 
    >>> # Compute the map
    >>> poincare_map.compute()
    >>> 
    >>> # Plot the results
    >>> poincare_map.plot()
    """

    def __init__(self, config: _CenterManifoldMapConfig, interface: _CenterManifoldInterface, engine: _CenterManifoldEngine | None = None) -> None:
        super().__init__(config, interface, engine)

    @classmethod
    def with_default_engine(
        cls,
        config: _CenterManifoldMapConfig,
        interface: Optional[_CenterManifoldInterface] = None,\
    ) -> "CenterManifoldMapPipeline":
        """Construct a map with a default-wired engine injected.

        This mirrors the DI-friendly facades (e.g., ConnectionPipeline) by creating
        a default engine using the current configuration and injecting it.
        The engine is wired for the default section coordinate in the config.
        """
        from hiten.algorithms.poincare.centermanifold.backend import \
            _CenterManifoldBackend
        from hiten.algorithms.poincare.centermanifold.engine import \
            _CenterManifoldEngine
        from hiten.algorithms.poincare.centermanifold.interfaces import \
            _CenterManifoldInterface

        backend = _CenterManifoldBackend()
        map_intf = interface or _CenterManifoldInterface()
        strategy = cls._build_strategy(config)
        engine = _CenterManifoldEngine(backend=backend, seed_strategy=strategy, map_config=config, interface=map_intf)
        return cls(config, map_intf, engine)

    def generate(
        self,
        domain_obj: DomainT,
        override: bool = False,
        *,
        section_coord: str | None = None,
        dt: float | None = None,
        n_iter: int | None = None,
        n_workers: int | None = None,
        method: Literal["fixed", "adaptive", "symplectic"] | None = None,
        order: int | None = None,
        c_omega_heuristic: float | None = None,
    ) -> CenterManifoldMapResults:
        """Compute the section, supporting runtime overrides without mutating config.

        If no overrides are provided, this defers to the cached, default setup
        and persists the result. If any overrides are provided, a temporary
        engine is assembled for this call and the result is returned without
        polluting the persistent cache. In all cases, this method returns the
        2-D points of the section.
        """
        kwargs = {
            "section_coord": section_coord,
            "dt": dt,
            "n_iter": n_iter,
            "n_workers": n_workers,
            "method": method,
            "order": order,
            "c_omega_heuristic": c_omega_heuristic,
        }
        problem = self._create_problem(domain_obj=domain_obj, override=override, **kwargs)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

    @staticmethod
    def _build_strategy(config: _CenterManifoldMapConfig) -> _CenterManifoldSeedingBase:
        strategy_kwargs: dict[str, object] = {}
        if config.seed_strategy == "single":
            strategy_kwargs["seed_axis"] = config.seed_axis

        return _make_strategy(config, **strategy_kwargs)
