Basic Orbit Propagation
========================

This guide covers the fundamental concepts of propagating orbits in the Circular Restricted Three-Body Problem using HITEN's integration methods.

System-Level Propagation
------------------------

The simplest way to propagate an orbit is using the system's built-in propagation method:

.. code-block:: python

   from hiten import System
   import numpy as np
   
   # Create Earth-Moon system
   system = System.from_bodies("earth", "moon")
   
   # Initial conditions in rotating frame [x, y, z, vx, vy, vz]
   initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])
   
   # Propagate for one orbital period
   trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,  # One orbital period
       steps=1000
   )
   
   print(f"Propagated {trajectory.n_samples} time steps")
   print(f"Final time: {trajectory.tf}")
   print(f"Final state: {trajectory.states[-1]}")

State Vector Format
-------------------

HITEN uses the standard CR3BP state vector format:

.. code-block:: python

   # State vector: [x, y, z, vx, vy, vz]
   # x, y, z: position in rotating frame (nondimensional)
   # vx, vy, vz: velocity in rotating frame (nondimensional)
   
   # Example: Position near L1
   l1 = system.get_libration_point(1)
   x_l1 = l1.position[0]
   
   # Initial state near L1
   initial_state = np.array([
       x_l1 - 0.01,  # x: slightly left of L1
       0.0,          # y: on x-axis
       0.0,          # z: in orbital plane
       0.0,          # vx: no radial velocity
       0.1,          # vy: small tangential velocity
       0.0           # vz: no out-of-plane velocity
   ])

Time Units
----------

All times are in nondimensional units:

.. code-block:: python

   # Time units: T = 2*pi / n
   # where n is the mean motion of the primaries
   
   # One orbital period = 2*pi
   period = 2 * np.pi
   
   # Half period
   half_period = np.pi
   
   # Multiple periods
   multiple_periods = 4 * np.pi

Practical Examples
------------------

.. code-block:: python

   # Custom initial conditions
   custom_state = np.array([0.7, 0.1, 0.05, 0.0, 0.15, 0.02])
   
   trajectory = system.propagate(
       initial_conditions=custom_state,
       tf=10 * np.pi,  # 5 orbital periods
       steps=2000
   )

Visualization
-------------

Plot propagated trajectories:

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Propagate orbit
   trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       steps=1000
   )
   
   # Extract position components
   x = trajectory.states[:, 0]
   y = trajectory.states[:, 1]
   z = trajectory.states[:, 2]
   
   # 3D plot
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   ax.plot(x, y, z, 'b-', linewidth=2)
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Orbit Trajectory')
   plt.show()
   
   # 2D projection
   plt.figure(figsize=(8, 8))
   plt.plot(x, y, 'b-', linewidth=2)
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title('Orbit Projection (X-Y plane)')
   plt.axis('equal')
   plt.show()

Next Steps
----------

Once you understand basic propagation, you can:

- Create periodic orbits (see :doc:`guide_04_orbits`)
- Compute manifolds (see :doc:`guide_05_manifolds`)
- Analyze Poincare sections (see :doc:`guide_06_poincare`)

For advanced propagation techniques, see :doc:`guide_10_integrators`.
