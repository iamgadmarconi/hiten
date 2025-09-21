Domain Interfaces
=================

The interfaces module provides domain translation interfaces that adapt between high-level domain objects and low-level numerical kernels. These interfaces centralize the logic for coordinate transformations and constraint handling.

Center Manifold Interfaces
--------------------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.interfaces

.. autofunction:: _solve_bracketed()

Solve a scalar equation using bracketed root finding. Provides robust root finding for constraint equations with guaranteed convergence within specified brackets.

.. autofunction:: _solve_for_missing_coordinate()

Solve for missing coordinate on a Poincare section using energy constraint. Uses root finding to determine the missing coordinate value that satisfies the energy equation H(q,p) = h0.

.. autofunction:: _lift_to_4d()

Lift 2D plane points to 4D center manifold states. Converts points from the 2D section plane to full 4D state vectors (q2, p2, q3, p3) on the center manifold.

.. autofunction:: _build_constraint_dict()

Build constraint dictionary for energy equation solving. Creates the necessary constraint information for root finding algorithms to solve the energy constraint equation.

.. autoclass:: _CenterManifoldInterface()
   :members:
   :exclude-members: __init__

Stateless interface for center manifold domain translations. Provides pure functions for coordinate transformations, constraint solving, and state vector manipulation specific to center manifold computations.
