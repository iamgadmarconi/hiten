"""Public API for the :mod:`~hiten.system` package.

This module re-exports the most frequently used classes so that users can
simply write::

>>> from hiten.system import System, Body, L1Point, HaloOrbit
"""

from ..algorithms.utils.types import SynodicState
from ..algorithms.poincare.centermanifold.base import CenterManifoldMap
from ..algorithms.poincare.centermanifold.config import _CenterManifoldMapConfig
from .base import System
# Core containers
from .body import Body
# Center manifold
from .center import CenterManifold
from .family import OrbitFamily
# Libration points
from .libration.base import LibrationPoint, LinearData
from .libration.collinear import CollinearPoint, L1Point, L2Point, L3Point
from .libration.triangular import L4Point, L5Point, TriangularPoint
from .manifold import Manifold, ManifoldResult
# Orbits
from .orbits.base import GenericOrbit, PeriodicOrbit
from .orbits.halo import HaloOrbit
from .orbits.lyapunov import LyapunovOrbit
from .orbits.vertical import VerticalOrbit

__all__ = [
    # Base system
    "Body",
    "System",
    "ManifoldResult",
    "Manifold",
    # Libration points
    "LinearData",
    "LibrationPoint",
    "CollinearPoint",
    "TriangularPoint",
    "L1Point",
    "L2Point",
    "L3Point",
    "L4Point",
    "L5Point",
    # Center manifold
    "CenterManifold",
    # Poincare map
    "_CenterManifoldMapConfig",
    "CenterManifoldMap",
    # Orbits / configs
    "PeriodicOrbit",
    "GenericOrbit",
    "HaloOrbit",
    "LyapunovOrbit",
    "VerticalOrbit",
    "SynodicState",
    # Family
    "OrbitFamily",
]
