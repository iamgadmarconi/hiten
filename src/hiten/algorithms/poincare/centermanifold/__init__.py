"""Center manifold seeding strategies for Poincare maps.

This module provides various strategies for seeding initial conditions
on center manifolds of periodic orbits in the Circular Restricted 
Three-Body Problem (CR3BP). The strategies are used to generate 
initial conditions for computing center manifold trajectories.

The module exports a factory function :func:`~hiten.algorithms.poincare.centermanifold._make_strategy` 
that creates concrete seeding strategy instances based on a string identifier.
"""

from .base import CenterManifoldMap
from .config import _CenterManifoldMapConfig
from .types import CenterManifoldMapResults

__all__ = [
    "CenterManifoldMap",
    "_CenterManifoldMapConfig",
    "CenterManifoldMapResults",
]
