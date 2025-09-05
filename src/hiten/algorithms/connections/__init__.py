"""
hiten.algorithms.connections
===========================

Scaffolding for the connection-finding framework (heteroclinic/pseudo-heteroclinic)
in the CR3BP. This package will orchestrate endpoints (orbits/manifolds/LPs),
section adapters, configuration, a unified engine, and result containers.

This module currently exposes light-weight stubs that will be filled incrementally.
"""

from .base import Connection
from .config import _SearchConfig as SearchConfig

__all__ = [
    # Configs
    "SearchConfig",
    "Connection",
]


