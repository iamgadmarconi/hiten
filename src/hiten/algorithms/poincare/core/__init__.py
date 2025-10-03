"""Core Poincare map infrastructure.

This module provides the fundamental infrastructure for Poincare return map
computation, including base classes, configuration management, and common utilities.
"""

from .backend import _ReturnMapBackend
from .config import _ReturnMapConfig, _SeedingConfig
from .engine import _ReturnMapEngine
from .events import _PlaneEvent, _SurfaceEvent
from .interfaces import _SectionInterface
from .options import IterationOptions, SeedingOptions
from .seeding import _SeedingProtocol
from .strategies import _SeedingStrategyBase
from .types import _SectionHit

__all__ = [
    # Backends and Engines
    "_ReturnMapBackend",
    "_ReturnMapEngine",
    "_SeedingStrategyBase",
    # Configuration (compile-time structure)
    "_ReturnMapConfig",
    "_SeedingConfig",
    # Options (runtime tuning)
    "IterationOptions",
    "SeedingOptions",
    # Interfaces and Events
    "_SectionInterface",
    "_SurfaceEvent",
    "_PlaneEvent",
    "_SeedingProtocol",
    # Types
    "_SectionHit",
]
