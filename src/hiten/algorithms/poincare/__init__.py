from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig as CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.strategies import (
    _AxisAlignedSeeding, _LevelSetsSeeding, _RadialSeeding, _RandomSeeding,
    _SingleAxisSeeding)
from hiten.algorithms.poincare.synodic.base import SynodicMap
from hiten.algorithms.poincare.synodic.config import \
    _SynodicMapConfig as SynodicMapConfig


def _build_seeding_strategy(section_cfg, config):
    """Select and instantiate the requested seed-generation strategy.

    Uses a small factory-mapping so adding a new strategy only requires
    registering one extra line instead of editing a long if/elif chain.
    """

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
    "_SingleAxisSeeding", 
    "_AxisAlignedSeeding", 
    "_LevelSetsSeeding", 
    "_RadialSeeding", 
    "_RandomSeeding",
    "CenterManifoldMap",   
    "CenterManifoldMapConfig",
    "SynodicMap",
    "SynodicMapConfig",
]
