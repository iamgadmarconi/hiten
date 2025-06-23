"""
Public API for the ``system`` package.

This module re-exports the most frequently used classes so that users can
simply write::

    from system import System, Body, L1Point, HaloOrbit

instead of navigating the full internal hierarchy (``hiten.system.base``,
``hiten.system.libration.collinear`` …).
"""

# Core containers
from .body import Body
from .base import System
from .manifold import ManifoldResult, Manifold

# Libration points
from .libration.base import LinearData, LibrationPoint
from .libration.collinear import CollinearPoint, L1Point, L2Point, L3Point
from .libration.triangular import TriangularPoint, L4Point, L5Point

# Center manifold
from .center import CenterManifold

# Poincare map
from ..algorithms.poincare.base import _PoincareMapConfig, _PoincareMap

# Orbits
from .orbits.base import (
    _CorrectionConfig,
    PeriodicOrbit,
    GenericOrbit,
    S,  # state-vector helper enum
)
from .orbits.halo import HaloOrbit
from .orbits.lyapunov import LyapunovOrbit, VerticalLyapunovOrbit

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
    "VerticalLyapunovOrbit",
    "S",
]
