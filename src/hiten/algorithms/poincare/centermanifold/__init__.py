"""Center manifold seeding strategies for Poincare maps.

This module provides various strategies for seeding initial conditions
on center manifolds of periodic orbits in the Circular Restricted 
Three-Body Problem (CR3BP). The strategies are used to generate 
initial conditions for computing center manifold trajectories.

The module exports a factory function :func:`~hiten.algorithms.poincare.centermanifold._make_strategy` 
that creates concrete seeding strategy instances based on a string identifier.
"""

from .config import _CenterManifoldSectionConfig, _CenterManifoldMapConfig
from .seeding import _CenterManifoldSeedingBase
from .strategies import (_AxisAlignedSeeding, _LevelSetsSeeding,
                         _RadialSeeding, _RandomSeeding, _SingleAxisSeeding)

_STRATEGY_MAP = {
    "single": _SingleAxisSeeding,
    "axis_aligned": _AxisAlignedSeeding,
    "level_sets": _LevelSetsSeeding,
    "radial": _RadialSeeding,
    "random": _RandomSeeding,
}


def _make_strategy(kind: str, section_config: _CenterManifoldSectionConfig, 
                   map_config: _CenterManifoldMapConfig, **kwargs) -> _CenterManifoldSeedingBase:
    """Factory returning a concrete seeding strategy.

    Parameters
    ----------
    kind : str
        Strategy identifier. Must be one of: 'single', 'axis_aligned', 
        'level_sets', 'radial', or 'random'.
    section_config : :class:`~hiten.algorithms.poincare.centermanifold.config._CenterManifoldSectionConfig`
        Configuration describing the Poincare section parameters.
    map_config : :class:`~hiten.algorithms.poincare.centermanifold.config._CenterManifoldMapConfig`
        Map-level configuration containing global parameters such as
        ``n_seeds`` and ``seed_axis``.
    **kwargs
        Additional keyword arguments forwarded to the concrete strategy
        constructor.

    Returns
    -------
    :class:`~hiten.algorithms.poincare.centermanifold.seeding._CenterManifoldSeedingBase`
        Concrete seeding strategy instance.

    Raises
    ------
    ValueError
        If ``kind`` is not a valid strategy identifier.

    Notes
    -----
    The available strategies are:
    - 'single': Single axis seeding along one coordinate direction
    - 'axis_aligned': Seeding aligned with coordinate axes
    - 'level_sets': Seeding based on level sets of the Hamiltonian
    - 'radial': Radial seeding pattern from the periodic orbit
    - 'random': Random seeding within specified bounds
    """
    try:
        cls = _STRATEGY_MAP[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown seed_strategy '{kind}'") from exc
    return cls(section_config, map_config, **kwargs)

__all__ = [
    "_CenterManifoldSeedingBase",
    "_make_strategy",
    "_CenterManifoldSectionConfig",
    "_AxisAlignedSeeding",
    "_LevelSetsSeeding",
    "_RadialSeeding",
    "_RandomSeeding",
    "_SingleAxisSeeding",
]
