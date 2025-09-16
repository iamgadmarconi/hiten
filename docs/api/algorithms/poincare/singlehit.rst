Single-Hit Poincare Section Detection
======================================

The singlehit module provides single-hit Poincare section detection for individual trajectories, useful for finding specific section crossings. This module is designed for cases where you need to find a single intersection point rather than generating complete return maps.

.. currentmodule:: hiten.algorithms.poincare.singlehit

.. autoclass:: _SingleHitBackend()
   :members:
   :exclude-members: __init__

Backend for single-hit Poincare section detection. Implements efficient algorithms for finding the next section crossing from a given initial state using numerical integration and root finding.

.. autofunction:: find_crossing()

High-level function for finding a single Poincare section crossing. Provides a convenient interface for detecting section crossings from initial conditions.

.. autofunction:: _plane_crossing_factory()

Factory function for creating plane crossing detection functions. Generates specialized crossing detection functions for specific coordinate planes.
