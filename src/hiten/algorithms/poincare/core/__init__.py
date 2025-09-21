"""Core Poincare map infrastructure.

This module provides the fundamental infrastructure for Poincare return map
computation, including base classes, configuration management, and common utilities.
"""

from .backend import _ReturnMapBackend
from .base import _ReturnMapBase, _Section
from .config import (_EngineConfigLike, _IntegrationConfig, _IterationConfig,
                     _ReturnMapBaseConfig, _ReturnMapConfig, _SectionConfig,
                     _SeedingConfig, _SeedingConfigLike)
from .engine import _ReturnMapEngine
from .events import _PlaneEvent, _SurfaceEvent
from .seeding import _SeedingProtocol
from .strategies import _SeedingStrategyBase
from .types import _SectionHit

__all__ = [
    "_ReturnMapBase",
    "_Section",
    "_ReturnMapBackend",
    "_ReturnMapEngine",
    "_SeedingStrategyBase",
    "_ReturnMapBaseConfig",
    "_IntegrationConfig",
    "_IterationConfig",
    "_SeedingConfig",
    "_ReturnMapConfig",
    "_SectionConfig",
    "_EngineConfigLike",
    "_SeedingConfigLike",
    "_SurfaceEvent",
    "_SectionHit",
    "_PlaneEvent",
    "_SeedingProtocol",
]
