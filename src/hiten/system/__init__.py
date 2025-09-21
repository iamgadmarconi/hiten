"""Public API for the :mod:`~hiten.system` package.

This module re-exports the most frequently used classes so that users can
simply write::

>>> from hiten.system import System, Body, L1Point, HaloOrbit
"""

from ..algorithms.poincare.centermanifold.base import CenterManifoldMap
from ..algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from ..algorithms.utils.types import SynodicState
from .base import System
from .body import Body
from .center import CenterManifold
from .family import OrbitFamily
from .libration.base import LibrationPoint, LinearData
from .libration.collinear import CollinearPoint, L1Point, L2Point, L3Point
from .libration.triangular import L4Point, L5Point, TriangularPoint
from .manifold import Manifold, ManifoldResult
from .orbits.base import GenericOrbit, PeriodicOrbit
from .orbits.halo import HaloOrbit
from .orbits.lyapunov import LyapunovOrbit
from .orbits.vertical import VerticalOrbit
from .torus import InvariantTori, Torus

__all__ = [
    "Body",
    "System",
    "ManifoldResult",
    "Manifold",
    "LinearData",
    "LibrationPoint",
    "CollinearPoint",
    "TriangularPoint",
    "L1Point",
    "L2Point",
    "L3Point",
    "L4Point",
    "L5Point",
    "CenterManifold",
    "_CenterManifoldMapConfig",
    "CenterManifoldMap",
    "PeriodicOrbit",
    "GenericOrbit",
    "HaloOrbit",
    "LyapunovOrbit",
    "VerticalOrbit",
    "SynodicState",
    # Family
    "OrbitFamily",
    # Tori
    "InvariantTori",
    "Torus",
]
