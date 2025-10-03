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

from typing import Generic, Optional

from hiten.algorithms.poincare.centermanifold.backend import \
    _CenterManifoldBackend
from hiten.algorithms.poincare.centermanifold.config import \
    CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.engine import \
    _CenterManifoldEngine
from hiten.algorithms.poincare.centermanifold.interfaces import \
    _CenterManifoldInterface
from hiten.algorithms.poincare.centermanifold.options import \
    CenterManifoldMapOptions
from hiten.algorithms.poincare.centermanifold.seeding import \
    _CenterManifoldSeedingBase
from hiten.algorithms.poincare.centermanifold.strategies import _make_strategy
from hiten.algorithms.poincare.centermanifold.types import \
    CenterManifoldMapResults
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)


class CenterManifoldMapPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
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

    def __init__(self, config: CenterManifoldMapConfig, engine: _CenterManifoldEngine, interface: _CenterManifoldInterface = None, backend: _CenterManifoldBackend = None) -> None:
        super().__init__(config, engine, interface, backend)

    @classmethod
    def with_default_engine(
        cls,
        config: CenterManifoldMapConfig,
        interface: Optional[_CenterManifoldInterface] = None,\
        backend: Optional[_CenterManifoldBackend] = None,
    ) -> "CenterManifoldMapPipeline":
        """Construct a map with a default-wired engine injected.

        This mirrors the DI-friendly facades (e.g., ConnectionPipeline) by creating
        a default engine using the current configuration and injecting it.
        The engine is wired for the default section coordinate in the config.

        Parameters
        ----------
        config : :class:`~hiten.algorithms.poincare.centermanifold.config.CenterManifoldMapConfig`
            Configuration object for the center manifold map.
        interface : :class:`~hiten.algorithms.poincare.centermanifold.interfaces._CenterManifoldInterface`, optional
            Interface object for the center manifold map. If None, uses the default _CenterManifoldInterface.
        backend : :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend`, optional
            Backend instance for the center manifold map. If None, uses the default _CenterManifoldBackend.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMapPipeline`
            A center manifold map facade instance with a default engine injected.
        """
        from hiten.algorithms.poincare.centermanifold.backend import \
            _CenterManifoldBackend
        from hiten.algorithms.poincare.centermanifold.engine import \
            _CenterManifoldEngine
        from hiten.algorithms.poincare.centermanifold.interfaces import \
            _CenterManifoldInterface

        backend = backend or _CenterManifoldBackend()
        map_intf = interface or _CenterManifoldInterface()
        strategy = cls._build_strategy(config)
        engine = _CenterManifoldEngine(backend=backend, seed_strategy=strategy, map_config=config, interface=map_intf)
        return cls(config, engine, map_intf, backend)

    def generate(
        self,
        domain_obj: DomainT,
        options: CenterManifoldMapOptions,
    ) -> CenterManifoldMapResults:
        """Compute the section using configured engine.

        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.core.DomainT`
            The domain object to compute the map for.
        options : :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            Runtime options for the map pipeline.

        Returns
        -------
        CenterManifoldMapResults
            The computed section results.
        """
        problem = self._create_problem(domain_obj=domain_obj, options=options)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

    @staticmethod
    def _build_strategy(config: CenterManifoldMapConfig) -> _CenterManifoldSeedingBase:
        strategy_kwargs: dict[str, object] = {}
        if config.seed_strategy == "single":
            strategy_kwargs["seed_axis"] = config.seed_axis

        return _make_strategy(config, **strategy_kwargs)
