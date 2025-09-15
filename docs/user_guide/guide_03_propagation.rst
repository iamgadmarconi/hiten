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
   times, trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,  # One orbital period
       steps=1000
   )
   
   print(f"Propagated {len(times)} time steps")
   print(f"Final time: {times[-1]}")
   print(f"Final state: {trajectory[-1]}")

Integration Methods
-------------------

HITEN supports multiple integration methods:

.. code-block:: python

   # Runge-Kutta method
   times_rk, traj_rk = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       method="rk",
       order=8
   )
   
   # Scipy integration (default)
   times_scipy, traj_scipy = system.propagate(
       initial_conditions=initial_state,
       method="scipy"
   )
   
   # Symplectic integration
   times_symp, traj_symp = system.propagate(
       initial_conditions=initial_state,
       method="symplectic",
       order=4
   )
   
   # Adaptive integration
   times_adapt, traj_adapt = system.propagate(
       initial_conditions=initial_state,
       method="adaptive"
   )

Integration Parameters
----------------------

Control integration accuracy and performance:

.. code-block:: python

   # High accuracy integration
   times, trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       steps=5000,  # More steps for higher accuracy
       method="scipy",
       order=8
   )
   
   # Fast integration
   times, trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       steps=100,   # Fewer steps for speed
       method="rk",
       order=4
   )

Energy Conservation
-------------------

Monitor energy conservation during integration:

.. code-block:: python

   from hiten.algorithms.dynamics.utils.energy import crtbp_energy
   
   # Initial energy
   initial_energy = crtbp_energy(initial_state, system.mu)
   print(f"Initial energy: {initial_energy}")
   
   # Propagate
   times, trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       steps=1000
   )
   
   # Check energy conservation
   final_energy = crtbp_energy(trajectory[-1], system.mu)
   energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
   print(f"Energy error: {energy_error:.2e}")

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

L1 Halo Orbit Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   import numpy as np
   
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct()
   halo.propagate()
   
   # Get the trajectory
   times = halo.times
   trajectory = halo.trajectory
   
   print(f"Halo period: {halo.period}")
   print(f"Trajectory shape: {trajectory.shape}")

Lyapunov Orbit Propagation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create Lyapunov orbit
   lyapunov = l1.create_orbit("lyapunov", amplitude_x=0.05)
   lyapunov.correct()
   lyapunov.propagate()
   
   # Access trajectory
   times = lyapunov.times
   trajectory = lyapunov.trajectory

Custom Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom initial conditions
   custom_state = np.array([0.7, 0.1, 0.05, 0.0, 0.15, 0.02])
   
   times, trajectory = system.propagate(
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
   times, trajectory = system.propagate(
       initial_conditions=initial_state,
       tf=2 * np.pi,
       steps=1000
   )
   
   # Extract position components
   x = trajectory[:, 0]
   y = trajectory[:, 1]
   z = trajectory[:, 2]
   
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

Error Analysis
--------------

Monitor integration errors:

.. code-block:: python

   # Compare different integration methods
   methods = ["scipy", "rk", "symplectic", "adaptive"]
   
   for method in methods:
       times, trajectory = system.propagate(
           initial_conditions=initial_state,
           tf=2 * np.pi,
           method=method,
           steps=1000
       )
       
       # Check energy conservation
       initial_energy = crtbp_energy(initial_state, system.mu)
       final_energy = crtbp_energy(trajectory[-1], system.mu)
       energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
       
       print(f"{method}: Energy error = {energy_error:.2e}")

Next Steps
----------

Once you understand basic propagation, you can:

- Create periodic orbits (see :doc:`guide_04_orbits`)
- Compute manifolds (see :doc:`guide_05_manifolds`)
- Analyze Poincare sections (see :doc:`guide_06_poincare`)

For advanced propagation techniques, see :doc:`guide_08_advanced`.
