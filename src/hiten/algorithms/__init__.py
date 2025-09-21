""" Public API for the :mod:`~hiten.algorithms` package.
"""

from .continuation.config import \
    _OrbitContinuationConfig as OrbitContinuationConfig
from .continuation.facades import StateParameter
from .corrector.config import _LineSearchConfig as LineSearchConfig
from .corrector.config import _OrbitCorrectionConfig as OrbitCorrectionConfig
from .poincare.centermanifold.base import CenterManifoldMap
from .poincare.centermanifold.config import \
    _CenterManifoldMapConfig as CenterManifoldMapConfig
from .poincare.synodic.base import SynodicMap
from .poincare.synodic.config import _SynodicMapConfig as SynodicMapConfig
from .poincare.synodic.config import \
    _SynodicSectionConfig as SynodicSectionConfig

__all__ = [
    "StateParameter",
    "CenterManifoldMap",
    "CenterManifoldMapConfig",
    "SynodicMap",
    "SynodicMapConfig",
    "SynodicSectionConfig",
    "LineSearchConfig",
    "OrbitCorrectionConfig",
    "OrbitContinuationConfig",
    "_CONVERSION_REGISTRY",
]
