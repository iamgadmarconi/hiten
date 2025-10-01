"""Provide a connection discovery framework for manifold transfers in the CR3BP.

This package provides a comprehensive framework for discovering ballistic and
impulsive transfers between manifolds in the Circular Restricted Three-Body
Problem (CR3BP). It orchestrates the complete pipeline from manifold intersection
with synodic sections to geometric analysis and Delta-V computation.

Examples
--------
>>> from hiten.algorithms.connections import ConnectionPipeline
>>> from hiten.algorithms.connections.config import ConnectionConfig
>>> from hiten.algorithms.connections.options import ConnectionOptions
>>> from hiten.algorithms.poincare.synodic.config import SynodicMapConfig
>>> 
>>> # Configure synodic section (compile-time)
>>> section_cfg = SynodicMapConfig(section_axis="x", section_offset=0.8)
>>> config = ConnectionConfig(section=section_cfg, direction=1)
>>> 
>>> # Configure search parameters (runtime)
>>> options = ConnectionOptions(delta_v_tol=1e-3, eps2d=1e-4)
>>> 
>>> # Create facade with a default engine
>>> conn = ConnectionPipeline.with_default_engine(config=config)
>>> 
>>> # Discover connections (can override options at runtime)
>>> results = conn.solve(unstable_manifold, stable_manifold, options=options)
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

from .base import ConnectionPipeline
from .config import ConnectionConfig
from .options import ConnectionOptions

__all__ = [
    # Main interface
    "ConnectionPipeline",
    # Configuration (compile-time structure)
    "ConnectionConfig",
    # Options (runtime tuning)
    "ConnectionOptions",
]


