"""
Initialize the system module.

This module provides classes for defining the Circular Restricted Three-Body Problem (CR3BP)
system, celestial bodies, and libration points.
"""

# Import core components to be available at the package level
from .base import System, systemConfig
from .body import Body
from .libration import (
    LibrationPoint,
    CollinearPoint,  # Often useful for type checking
    TriangularPoint, # Often useful for type checking
    L1Point,
    L2Point,
    L3Point,
    L4Point,
    L5Point
)

# Define what gets imported with `from system import *`
__all__ = [
    "System",
    "systemConfig",
    "Body",
    "LibrationPoint",
    "CollinearPoint",
    "TriangularPoint",
    "L1Point",
    "L2Point",
    "L3Point",
    "L4Point",
    "L5Point",
]
