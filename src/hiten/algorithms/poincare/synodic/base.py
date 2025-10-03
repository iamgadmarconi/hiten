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

from typing import Generic, Optional

from hiten.algorithms.poincare.synodic.backend import _SynodicDetectionBackend
from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
from hiten.algorithms.poincare.synodic.interfaces import _SynodicInterface
from hiten.algorithms.poincare.synodic.options import SynodicMapOptions
from hiten.algorithms.poincare.synodic.strategies import _NoOpStrategy
from hiten.algorithms.poincare.synodic.types import SynodicMapResults
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)


class SynodicMapPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """User-facing interface for synodic Poincare section detection.

    This class provides a facade that mirrors the API of other return-map
    modules while specializing in synodic Poincare section detection on
    precomputed trajectories. It does not propagate trajectories; callers
    supply them explicitly through various input methods.

    Parameters
    ----------
    map_cfg : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`, optional
        Configuration object containing detection parameters, section geometry,
        and refinement settings. If None, uses default configuration.

    Attributes
    ----------
    config : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
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

    def __init__(self, config: SynodicMapConfig, engine: _SynodicEngine, interface: _SynodicInterface = None, backend: _SynodicDetectionBackend = None) -> None:
        super().__init__(config, engine, interface, backend)

    @classmethod
    def with_default_engine(
        cls,
        config: SynodicMapConfig,
        interface: Optional[_SynodicInterface] = None,
        backend: Optional[_SynodicDetectionBackend] = None,
    ) -> "SynodicMapPipeline":
        """Construct a map with a default-wired engine injected.

        This mirrors the DI-friendly facades (e.g., ConnectionPipeline) by creating
        a default engine using the current configuration and injecting it.
        The engine is wired for the default section coordinate in the config.

        Parameters
        ----------
        config : :class:`~hiten.algorithms.poincare.synodic.config.SynodicMapConfig`
            Configuration object for the synodic map.
        interface : :class:`~hiten.algorithms.poincare.synodic.interfaces._SynodicInterface`, optional
            Interface object for the synodic map. If None, uses the default _SynodicInterface.
        backend : :class:`~hiten.algorithms.poincare.synodic.backend._SynodicDetectionBackend`, optional
            Backend instance for the synodic map. If None, uses the default _SynodicDetectionBackend.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.base.SynodicMapPipeline`
            A synodic map facade instance with a default engine injected.
        """
        from hiten.algorithms.poincare.synodic.backend import \
            _SynodicDetectionBackend
        from hiten.algorithms.poincare.synodic.engine import _SynodicEngine
        from hiten.algorithms.poincare.synodic.interfaces import \
            _SynodicInterface

        backend = backend or _SynodicDetectionBackend()
        map_intf = interface or _SynodicInterface()
        strategy = _NoOpStrategy(config)
        engine = _SynodicEngine(backend=backend, seed_strategy=strategy, map_config=config, interface=map_intf)
        return cls(config, engine, map_intf, backend)

    def generate(
        self,
        domain_obj: DomainT,
        options: SynodicMapOptions,
    ) -> SynodicMapResults:
        """Compute the section using configured engine.

        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.core.DomainT`
            The domain object to compute the map for.
        options : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            Runtime options for the map pipeline.

        Returns
        -------
        SynodicMapResults
            The computed section results.
        """
        problem = self._create_problem(domain_obj=domain_obj, options=options)
        engine = self._get_engine()
        engine_result = engine.solve(problem)
        self._results = engine_result
        return engine_result
