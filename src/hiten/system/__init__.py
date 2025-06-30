"""
Public API for the ``system`` package.

This module re-exports the most frequently used classes so that users can
simply write::

    from system import System, Body, L1Point, HaloOrbit

instead of navigating the full internal hierarchy (``hiten.system.base``,
``hiten.system.libration.collinear`` ...).
"""

# Poincare map
from ..algorithms.poincare.base import _PoincareMap, _PoincareMapConfig
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
from .orbits.base import S  # state-vector helper enum
from .orbits.base import GenericOrbit, PeriodicOrbit, _CorrectionConfig
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
    "_PoincareMapConfig",
    "_PoincareMap",
    # Orbits / configs
    "_CorrectionConfig",
    "PeriodicOrbit",
    "GenericOrbit",
    "HaloOrbit",
    "LyapunovOrbit",
    "VerticalOrbit",
    "S",
    # Family
    "OrbitFamily",
]
