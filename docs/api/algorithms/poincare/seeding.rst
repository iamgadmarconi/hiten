Seeding Strategies
==================

The seeding module provides protocols and implementations for generating initial conditions in Poincare return map computation. Seeding strategies determine how initial conditions are distributed on section planes or in phase space.

Core Seeding Protocol
---------------------

.. currentmodule:: hiten.algorithms.poincare.core.seeding

.. autoclass:: _SeedingProtocol()
   :members:

Protocol interface for seeding strategies. Defines the contract that all seeding implementations must follow, providing a flexible interface for various distribution strategies while maintaining consistency for the return map engine.

Center Manifold Seeding
-----------------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.seeding

.. autofunction:: _generate_seeds()

Generate seed points for center manifold Poincare maps. Creates initial conditions distributed according to the specified seeding strategy on the center manifold surface.

.. autofunction:: _axis_aligned_seeds()

Generate axis-aligned seed distribution for center manifold coordinates. Creates a regular grid of initial conditions aligned with the coordinate axes.

.. autofunction:: _radial_seeds()

Generate radially distributed seed points for center manifold maps. Creates initial conditions distributed in concentric circles or spheres around the origin.
