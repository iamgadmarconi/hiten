"""Poincare section algorithms for dynamical systems.

This module provides algorithms for computing Poincare sections in
dynamical systems, particularly the circular restricted three-body
problem. It includes both center manifold and synodic Poincare section
implementations with various seeding strategies.
"""

from hiten.algorithms.poincare.centermanifold.config import \
    CenterManifoldMapConfig as CenterManifoldMapConfig
from hiten.algorithms.poincare.synodic.config import \
    SynodicMapConfig as SynodicMapConfig


def _build_seeding_strategy(section_cfg, config):
    """Select and instantiate the requested seed-generation strategy.

    This factory function creates seeding strategy instances based on
    the configuration. It uses a mapping approach to avoid long if/elif
    chains and makes it easy to add new strategies.

    Parameters
    ----------
    section_cfg : :class:`~hiten.algorithms.poincare.core.interfaces._SectionInterface`
        The section configuration object.
    config : :class:`~hiten.algorithms.poincare.core.config._ReturnMapConfig`
        The map configuration containing the seed strategy specification.

    Returns
    -------
    :class:`~hiten.algorithms.poincare.core.strategies._SeedingStrategyBase`
        The instantiated seeding strategy.

    Raises
    ------
    ValueError
        If the specified seed strategy is unknown.

    Notes
    -----
    This function supports the following seeding strategies:
    - "single": Single axis seeding
    - "axis_aligned": Axis-aligned seeding
    - "level_sets": Level sets seeding
    - "radial": Radial seeding
    - "random": Random seeding

    The function uses a factory mapping pattern to avoid long conditional
    chains and makes it easy to add new strategies by simply registering
    them in the factories dictionary.

    All time units are in nondimensional units unless otherwise specified.
    """

    # Import here to avoid circular imports
    from hiten.algorithms.poincare.centermanifold.strategies import (
        _AxisAlignedSeeding, _LevelSetsSeeding, _RadialSeeding, _RandomSeeding,
        _SingleAxisSeeding)

    strat = config.seed_strategy.lower()

    factories = {
        "single": lambda: _SingleAxisSeeding(section_cfg, config, seed_axis=config.seed_axis),
        "axis_aligned": lambda: _AxisAlignedSeeding(section_cfg, config),
        "level_sets": lambda: _LevelSetsSeeding(section_cfg, config),
        "radial": lambda: _RadialSeeding(section_cfg, config),
        "random": lambda: _RandomSeeding(section_cfg, config),
    }

    try:
        return factories[strat]()
    except KeyError as exc:
        raise ValueError(f"Unknown seed strategy '{strat}'") from exc


__all__ = [
    "_build_seeding_strategy",
    "CenterManifoldMapConfig",
    "SynodicMapConfig",
]
