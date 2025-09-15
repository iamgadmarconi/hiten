Periodic Orbit Creation and Analysis
====================================

This guide covers the creation, correction, and analysis of periodic orbits in the Circular Restricted Three-Body Problem, including halo orbits, Lyapunov orbits, and vertical orbits.

Creating Periodic Orbits
------------------------

HITEN provides several types of periodic orbits that can be created from libration points:

Halo Orbits
~~~~~~~~~~~

Halo orbits are three-dimensional periodic orbits that appear as halos around libration points:

.. code-block:: python

   from hiten import System
   
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   
   # Correct the orbit to make it periodic
   halo.correct(max_attempts=25)
   
   # Propagate to get the trajectory
   halo.propagate(steps=1000)
   
   print(f"Halo period: {halo.period}")
   print(f"Halo family: {halo.family}")

Halo Orbit Parameters
~~~~~~~~~~~~~~~~~~~~~

Control halo orbit characteristics:

.. code-block:: python

   # Northern halo (above orbital plane)
   halo_north = l1.create_orbit("halo", amplitude_z=0.3, zenith="northern")
   
   # Southern halo (below orbital plane)
   halo_south = l1.create_orbit("halo", amplitude_z=0.3, zenith="southern")
   
   # Small amplitude halo
   halo_small = l1.create_orbit("halo", amplitude_z=0.1, zenith="southern")
   
   # Large amplitude halo
   halo_large = l1.create_orbit("halo", amplitude_z=0.5, zenith="southern")

Lyapunov Orbits
~~~~~~~~~~~~~~~

Lyapunov orbits are planar periodic orbits in the orbital plane:

.. code-block:: python

   # Create Lyapunov orbit
   lyapunov = l1.create_orbit("lyapunov", amplitude_x=0.05)
   
   # Correct and propagate
   lyapunov.correct(max_attempts=25)
   lyapunov.propagate(steps=1000)
   
   print(f"Lyapunov period: {lyapunov.period}")

Vertical Orbits
~~~~~~~~~~~~~~~

Vertical orbits are periodic orbits perpendicular to the orbital plane:

.. code-block:: python

   # Create vertical orbit
   vertical = l1.create_orbit("vertical", initial_state=[...])
   
   # Correct and propagate
   vertical.correct(max_attempts=100, finite_difference=True)
   vertical.propagate(steps=1000)
   
   print(f"Vertical period: {vertical.period}")

Generic Orbits
~~~~~~~~~~~~~~

Create orbits with custom initial conditions:

.. code-block:: python

   import numpy as np
   
   # Custom initial state
   custom_state = np.array([0.8, 0.0, 0.1, 0.0, 0.15, 0.0])
   
   # Create generic orbit
   generic = l1.create_orbit("generic", initial_state=custom_state)
   
   # Correct and propagate
   generic.correct(max_attempts=50)
   generic.propagate(steps=1000)

Orbit Correction
----------------

Differential correction is essential for making orbits truly periodic:

Basic Correction
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Correct with default parameters
   halo.correct()
   
   # Check if correction was successful
   if halo.period is not None:
       print(f"Correction successful, period: {halo.period}")
   else:
       print("Correction failed")

Advanced Correction
~~~~~~~~~~~~~~~~~~~

Control correction parameters:

.. code-block:: python

   # High accuracy correction
   halo.correct(
       max_attempts=50,
       tol=1e-12,
       max_delta=1e-6
   )
   
   # Fast correction
   halo.correct(
       max_attempts=10,
       tol=1e-6,
       max_delta=1e-3
   )

Finite Difference Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For some orbits, finite difference methods work better:

.. code-block:: python

   # Use finite difference for vertical orbits
   vertical.correct(
       max_attempts=100,
       finite_difference=True,
       tol=1e-10
   )

Orbit Analysis
--------------

Once corrected, orbits provide various analysis capabilities:

Period and Stability
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic properties
   print(f"Period: {halo.period}")
   print(f"Family: {halo.family}")
   print(f"Jacobi constant: {halo.jacobi_constant}")
   
   # Stability analysis
   stability_info = halo.compute_stability()
   print(f"Stability info: {stability_info}")

Trajectory Access
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get trajectory data
   times = halo.times
   trajectory = halo.trajectory
   
   print(f"Trajectory shape: {trajectory.shape}")
   print(f"Time range: {times[0]} to {times[-1]}")
   
   # Extract position components
   x = trajectory[:, 0]
   y = trajectory[:, 1]
   z = trajectory[:, 2]

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics.utils.energy import crtbp_energy
   
   # Compute energy along trajectory
   energies = [crtbp_energy(state, system.mu) for state in trajectory]
   
   # Check energy conservation
   initial_energy = energies[0]
   final_energy = energies[-1]
   energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
   
   print(f"Energy error: {energy_error:.2e}")

Orbit Families
--------------

Create and manage families of periodic orbits:

Creating Families
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import OrbitFamily
   
   # Create empty family
   family = OrbitFamily(system, orbit_type="halo")
   
   # Add orbits to family
   family.add_orbit(halo1)
   family.add_orbit(halo2)
   family.add_orbit(halo3)
   
   print(f"Family size: {len(family)}")

Family Properties
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access family properties
   orbits = family.orbits
   periods = family.periods
   jacobi_constants = family.jacobi_constants
   
   print(f"Periods: {periods}")
   print(f"Jacobi constants: {jacobi_constants}")

Family Propagation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Propagate all orbits in family
   family.propagate(steps=1000)
   
   # Access individual orbit trajectories
   for i, orbit in enumerate(family.orbits):
       print(f"Orbit {i}: {orbit.trajectory.shape}")

Continuation Methods
--------------------

Generate families of orbits using continuation:

State Parameter Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms import StateParameter
   from hiten.algorithms.utils.types import SynodicState
   
   # Create initial orbit
   initial_orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   initial_orbit.correct()
   
   # Set up continuation
   state_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.X, SynodicState.Z),
       amplitude=False,
       target=([0.8, 0.0], [0.9, 0.3]),
       step=(0.01, 0.03),
       max_orbits=10
   )
   
   # Run continuation
   state_engine.run()
   
   # Create family from continuation
   family = OrbitFamily.from_engine(state_engine)
   family.propagate()

Visualization
-------------

Plot periodic orbits and their families:

Single Orbit
~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # 3D plot
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   x = halo.trajectory[:, 0]
   y = halo.trajectory[:, 1]
   z = halo.trajectory[:, 2]
   
   ax.plot(x, y, z, 'b-', linewidth=2)
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Halo Orbit')
   plt.show()

Orbit Family
~~~~~~~~~~~~

.. code-block:: python

   # Plot family in 3D
   fig = plt.figure(figsize=(12, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   colors = plt.cm.viridis(np.linspace(0, 1, len(family)))
   
   for i, orbit in enumerate(family.orbits):
       x = orbit.trajectory[:, 0]
       y = orbit.trajectory[:, 1]
       z = orbit.trajectory[:, 2]
       
       ax.plot(x, y, z, color=colors[i], linewidth=1.5)
   
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Halo Family')
   plt.show()

Practical Examples
------------------

Earth-Moon L1 Halo Family
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms import StateParameter
   from hiten.algorithms.utils.types import SynodicState
   
   # Create system
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create initial halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct(max_attempts=25)
   
   # Generate family
   state_engine = StateParameter(
       initial_orbit=halo,
       state=(SynodicState.Z,),
       amplitude=True,
       target=(0.1, 0.5),
       step=0.05,
       max_orbits=10
   )
   
   state_engine.run()
   family = OrbitFamily.from_engine(state_engine)
   family.propagate()
   
   # Plot family
   family.plot()

Sun-Earth L2 Halo Family
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sun-Earth system
   system = System.from_bodies("sun", "earth")
   l2 = system.get_libration_point(2)
   
   # Create L2 halo
   halo_l2 = l2.create_orbit("halo", amplitude_z=0.1, zenith="northern")
   halo_l2.correct()
   halo_l2.propagate()
   
   # Generate family
   state_engine = StateParameter(
       initial_orbit=halo_l2,
       state=(SynodicState.Z,),
       amplitude=True,
       target=(0.05, 0.3),
       step=0.025,
       max_orbits=15
   )
   
   state_engine.run()
   family = OrbitFamily.from_engine(state_engine)
   family.propagate()

Common Issues
~~~~~~~~~~~~~

**Correction fails**
   - Check initial conditions are reasonable
   - Increase max_attempts
   - Try different correction method
   - Adjust tolerance parameters

**Orbit not periodic**
   - Verify correction was successful
   - Check period is not None
   - Increase correction accuracy

**Family generation fails**
   - Ensure initial orbit is well-corrected
   - Check continuation parameters
   - Verify target states are reachable

Next Steps
----------

Once you understand periodic orbits, you can:

- Compute their manifolds (see :doc:`guide_05_manifolds`)
- Analyze Poincare sections (see :doc:`guide_06_poincare`)
- Use center manifold methods (see :doc:`guide_07_center_manifold`)

For advanced orbit analysis, see :doc:`guide_08_advanced`.
