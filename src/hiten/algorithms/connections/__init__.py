"""
hiten.algorithms.connections
===========================

Scaffolding for the connection-finding framework (heteroclinic/pseudo-heteroclinic)
in the CR3BP. This package will orchestrate endpoints (orbits/manifolds/LPs),
section adapters, configuration, a unified engine, and result containers.

This module currently exposes light-weight stubs that will be filled incrementally.
"""

from .config import (_BallisticConfig, _ConnectionEngineConfig,
                     _ImpulsiveConfig, _SearchConfig, _SectionUseConfig)
from .endpoints import LPRef, ManifoldRef, OrbitRef
from .engine import ConnectionEngine, ConnectionProblem
from .section.base import _SectionAdapter
from .results import ConnectionResult
from .base import Connections

__all__ = [
    # Configs
    "_ConnectionEngineConfig",
    "_SectionUseConfig",
    "_BallisticConfig",
    "_ImpulsiveConfig",
    "_SearchConfig",
    # Endpoints
    "OrbitRef",
    "ManifoldRef",
    "LPRef",
    # Section adapter
    "_SectionAdapter",
    # Engine / problem / results
    "ConnectionProblem",
    "ConnectionEngine",
    "ConnectionResult",
    "Connections",
]


