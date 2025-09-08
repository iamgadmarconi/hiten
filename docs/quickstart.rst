Quick Start Guide
==================

This guide will help you get started with HITEN by walking through some basic examples.

Basic Setup
-----------

First, import the necessary modules:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hiten import Constants, CenterManifold, HaloOrbit

Setting Up a System
-------------------

HITEN works with the Circular Restricted Three-Body Problem (CR3BP). Let's set up the Earth-Moon system:

.. code-block:: python

   # Create constants for the Earth-Moon system
   # mu = m2 / (m1 + m2) where m1 is Earth and m2 is Moon
   constants = Constants(mu=0.012150585609624)
   
   print(f"Mass parameter: {constants.mu}")
   print(f"L1 position: {constants.l1_position}")
   print(f"L2 position: {constants.l2_position}")

Working with Periodic Orbits
----------------------------

Let's compute a simple periodic orbit near the L1 Lagrange point:

.. code-block:: python

   # Create a halo orbit object
   orbit = HaloOrbit(constants)
   
   # Set initial conditions near L1
   x0 = np.array([constants.l1_position - 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
   
   # Compute the orbit
   result = orbit.compute(x0, period_guess=2.0)
   
   print(f"Period: {result.period}")
   print(f"Initial state: {result.initial_state}")

Center Manifold Analysis
------------------------

For more advanced analysis, we can work with center manifolds:

.. code-block:: python

   # Create a center manifold
   manifold = CenterManifold(constants)
   
   # Set up the manifold near L1
   manifold.setup_libration_point(constants.l1_position)
   
   # Compute the center manifold
   center_manifold = manifold.compute()
   
   print(f"Center manifold dimension: {center_manifold.dimension}")

Visualization
-------------

Let's create a simple plot of our results:

.. code-block:: python

   # Plot the periodic orbit
   fig, ax = plt.subplots(1, 1, figsize=(10, 8))
   
   # Plot the primary bodies
   ax.plot(-constants.mu, 0, 'bo', markersize=10, label='Earth')
   ax.plot(1 - constants.mu, 0, 'go', markersize=6, label='Moon')
   
   # Plot the Lagrange points
   ax.plot(constants.l1_position, 0, 'ro', markersize=8, label='L1')
   ax.plot(constants.l2_position, 0, 'ro', markersize=8, label='L2')
   
   # Plot the periodic orbit
   if hasattr(result, 'states'):
       ax.plot(result.states[:, 0], result.states[:, 1], 'b-', 
               linewidth=2, label='Periodic Orbit')
   
   ax.set_xlabel('x (nondimensional)')
   ax.set_ylabel('y (nondimensional)')
   ax.set_title('Earth-Moon System with Periodic Orbit')
   ax.legend()
   ax.grid(True)
   ax.set_aspect('equal')
   
   plt.tight_layout()
   plt.show()

Advanced Features
-----------------

HITEN also provides tools for:

- **Invariant Manifolds**: Stable and unstable manifolds of periodic orbits
- **Bifurcation Analysis**: Detection and analysis of bifurcations
- **Poincare Maps**: Various mapping techniques
- **Fourier Analysis**: Spectral methods for periodic solutions
- **Hamiltonian Methods**: Normal form theory and center manifold reduction

Example: Computing Invariant Manifolds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import Manifold
   
   # Create an invariant manifold object
   manifold = Manifold(constants)
   
   # Set up the manifold for a periodic orbit
   manifold.setup_periodic_orbit(result)
   
   # Compute stable manifold
   stable_manifold = manifold.compute_stable()
   
   # Compute unstable manifold
   unstable_manifold = manifold.compute_unstable()
   
   print(f"Stable manifold points: {len(stable_manifold.points)}")
   print(f"Unstable manifold points: {len(unstable_manifold.points)}")

Next Steps
----------

Now that you have a basic understanding of HITEN, you can:

1. Explore the :doc:`examples/index` for more detailed examples
2. Read the :doc:`user_guide/system` for system configuration
3. Check the :doc:`api/system` for detailed API documentation
4. Learn about :doc:`user_guide/algorithms` for advanced algorithms

For more examples, see the `examples/` directory in the source code repository.

Troubleshooting
---------------

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify your Python version (3.9+ required)
3. Check the `GitHub Issues <https://github.com/iamgadmarconi/hiten/issues>`_ for known problems
4. Create a new issue if you find a bug

For more help, see the :doc:`installation` guide or the full :doc:`api/system` documentation.
