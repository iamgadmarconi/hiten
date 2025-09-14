"""Center manifold seeding strategies for Poincare maps.

This module provides various strategies for seeding initial conditions
on center manifolds of periodic orbits in the Circular Restricted 
Three-Body Problem (CR3BP). The strategies are used to generate 
initial conditions for computing center manifold trajectories.

The module exports a factory function :func:`~hiten.algorithms.poincare.centermanifold._make_strategy` 
that creates concrete seeding strategy instances based on a string identifier.
"""

from .backend import _CenterManifoldBackend
from .base import CenterManifoldMap
from .config import (_CenterManifoldMapConfig, _CenterManifoldSectionConfig,
                     _get_section_config)
from .engine import _CenterManifoldEngine
from .seeding import _CenterManifoldSeedingBase
from .strategies import (_AxisAlignedSeeding, _LevelSetsSeeding,
                         _RadialSeeding, _RandomSeeding, _SingleAxisSeeding,
                         _make_strategy)

__all__ = [
    "CenterManifoldMap",
    "_CenterManifoldMapConfig",
    "_CenterManifoldSectionConfig",
    "_CenterManifoldBackend",
    "_CenterManifoldEngine",
    "_CenterManifoldSeedingBase",
    "_SingleAxisSeeding",
    "_AxisAlignedSeeding",
    "_LevelSetsSeeding",
    "_RadialSeeding",
    "_RandomSeeding",
    "_make_strategy",
    "_get_section_config",
]
