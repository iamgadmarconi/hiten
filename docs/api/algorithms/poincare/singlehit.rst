Single-Hit Poincare Section Backend
===================================

The backend module provides a concrete implementation of single-hit Poincare section crossing search using numerical integration and root finding.

.. currentmodule:: hiten.algorithms.poincare.singlehit.backend

.. autoclass:: _SingleHitBackend()
   :members:
   :exclude-members: __init__

Concrete backend for single-hit Poincare section crossing search. Implements a two-stage approach: coarse integration followed by fine root finding to locate exact crossing points.

.. autofunction:: find_crossing()

High-level function for finding a single Poincare section crossing. Creates a backend instance and finds the crossing for a single state vector.

.. autofunction:: _plane_crossing_factory()

Factory function for creating plane crossing functions. Creates specialized crossing functions for specific coordinate planes with optional direction filtering.
