"""Public API for the :mod:`~hiten.system` package.

This module re-exports the most frequently used classes so that users can
simply write::

>>> from hiten.system import System, Body, L1Point, HaloOrbit
"""

from ..algorithms.poincare.centermanifold.config import CenterManifoldMapConfig
from ..algorithms.poincare.synodic.config import SynodicMapConfig
from ..algorithms.types.states import SynodicState
from .base import System
from .body import Body
from .center import CenterManifold
from .family import OrbitFamily
from .hamiltonian import Hamiltonian, LieGeneratingFunction
from .libration.base import LibrationPoint
from .libration.collinear import CollinearPoint, L1Point, L2Point, L3Point
from .libration.triangular import L4Point, L5Point, TriangularPoint
from .manifold import Manifold
from .maps import CenterManifoldMap, SynodicMap
from .orbits.base import GenericOrbit, PeriodicOrbit
from .orbits.halo import HaloOrbit
from .orbits.lissajous import LissajousOrbit
from .orbits.lyapunov import LyapunovOrbit
from .orbits.vertical import VerticalOrbit
from .torus import InvariantTori, Torus

__all__ = [
    "Body",
    "System",
    "Manifold",
    "LibrationPoint",
    "CollinearPoint",
    "LissajousOrbit",
    "TriangularPoint",
    "L1Point",
    "L2Point",
    "L3Point",
    "L4Point",
    "L5Point",
    "CenterManifold",
    "CenterManifoldMapConfig",
    "SynodicMapConfig",
    "CenterManifoldMap",
    "SynodicMap",
    "PeriodicOrbit",
    "GenericOrbit",
    "HaloOrbit",
    "LyapunovOrbit",
    "VerticalOrbit",
    "SynodicState",
    "OrbitFamily",
    "InvariantTori",
    "Torus",
    "Hamiltonian",
    "LieGeneratingFunction",
]
