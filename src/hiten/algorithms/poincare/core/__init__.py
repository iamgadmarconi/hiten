"""Core Poincare map infrastructure.

This module provides the fundamental infrastructure for Poincare return map
computation, including base classes, configuration management, and common utilities.
"""

from .backend import _ReturnMapBackend
from .config import (_IntegrationConfig, _IterationConfig,
                     _ReturnMapBaseConfig, _ReturnMapConfig, _SeedingConfig)
from .engine import _ReturnMapEngine
from .events import _PlaneEvent, _SurfaceEvent
from .interfaces import _SectionInterface
from .seeding import _SeedingProtocol
from .strategies import _SeedingStrategyBase
from .types import _SectionHit

__all__ = [
    "_ReturnMapBackend",
    "_ReturnMapEngine",
    "_SeedingStrategyBase",
    "_ReturnMapBaseConfig",
    "_IntegrationConfig",
    "_IterationConfig",
    "_SeedingConfig",
    "_ReturnMapConfig",
    "_SectionInterface",
    "_SurfaceEvent",
    "_SectionHit",
    "_PlaneEvent",
    "_SeedingProtocol",
]
