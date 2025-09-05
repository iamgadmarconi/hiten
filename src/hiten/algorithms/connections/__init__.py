"""
hiten.algorithms.connections
===========================

Scaffolding for the connection-finding framework (heteroclinic/pseudo-heteroclinic)
in the CR3BP. This package will orchestrate endpoints (orbits/manifolds/LPs),
section adapters, configuration, a unified engine, and result containers.

This module currently exposes light-weight stubs that will be filled incrementally.
"""

from .base import Connections
from .config import _SearchConfig
from .engine import _ConnectionEngine, _ConnectionProblem
from .interfaces import _ManifoldInterface
from .results import ConnectionResult

__all__ = [
    # Configs
    "_ConnectionConfig",
    "_SearchConfig",
    # Endpoints
    "_ManifoldInterface",
    # Engine / problem / results
    "_ConnectionProblem",
    "_ConnectionEngine",
    "ConnectionResult",
    "Connections",
]


