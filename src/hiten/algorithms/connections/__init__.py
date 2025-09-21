"""Provide a connection discovery framework for manifold transfers in the CR3BP.

This package provides a comprehensive framework for discovering ballistic and
impulsive transfers between manifolds in the Circular Restricted Three-Body
Problem (CR3BP). It orchestrates the complete pipeline from manifold intersection
with synodic sections to geometric analysis and Delta-V computation.

Examples
--------
>>> from hiten.algorithms.connections import Connection, SearchConfig
>>> from hiten.algorithms.poincare import SynodicMapConfig
>>> 
>>> # Configure synodic section
>>> section = SynodicMapConfig(section_axis="x", section_offset=0.8)
>>> 
>>> # Configure search parameters
>>> search = SearchConfig(delta_v_tol=1e-3, eps2d=1e-4)
>>> 
>>> # Create facade with a default engine
>>> conn = Connection.with_default_engine(section=section, search_cfg=search)
>>> 
>>> # Discover connections
>>> results = conn.solve(unstable_manifold, stable_manifold)
>>> print(f"Found {len(results)} connections")
>>> 
>>> # Visualize results
>>> conn.plot()

See Also
--------
:mod:`~hiten.system.manifold`
    Manifold classes for CR3BP invariant structures.
:mod:`~hiten.algorithms.poincare`
    Poincare map functionality for section intersections.
:mod:`~hiten.system`
    CR3BP system definition and libration points.
"""

from .base import Connection
from .config import _SearchConfig as SearchConfig

__all__ = [
    # Main interface
    "Connection",
    # Configuration
    "SearchConfig",
]


