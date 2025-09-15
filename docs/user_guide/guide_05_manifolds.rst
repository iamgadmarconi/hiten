Manifold and Tori Creation
===========================

This guide covers the creation and analysis of invariant manifolds and tori in the Circular Restricted Three-Body Problem, including stable/unstable manifolds and invariant tori.

Invariant Manifolds
-------------------

Invariant manifolds are geometric structures that organize the dynamics around periodic orbits. They provide natural transport channels in the CR3BP.

Creating Manifolds
~~~~~~~~~~~~~~~~~~

Manifolds are created from periodic orbits:

.. code-block:: python

   from hiten import System
   
   # Create system and orbit
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create and correct halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct(max_attempts=25)
   halo.propagate(steps=1000)
   
   # Create stable manifold
   stable_manifold = halo.manifold(stable=True, direction="positive")
   
   # Create unstable manifold
   unstable_manifold = halo.manifold(stable=False, direction="negative")

Manifold Types
~~~~~~~~~~~~~~

HITEN supports different manifold types:

.. code-block:: python

   # Stable manifolds
   stable_pos = halo.manifold(stable=True, direction="positive")
   stable_neg = halo.manifold(stable=True, direction="negative")
   
   # Unstable manifolds
   unstable_pos = halo.manifold(stable=False, direction="positive")
   unstable_neg = halo.manifold(stable=False, direction="negative")

Computing Manifolds
-------------------

Once created, manifolds must be computed to generate trajectories:

Basic Computation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute stable manifold
   result = stable_manifold.compute()
   
   print(f"Computation successful: {result is not None}")
   print(f"Success rate: {result.success_rate:.2%}")
   print(f"Number of trajectories: {len(result.trajectories)}")

Advanced Computation
~~~~~~~~~~~~~~~~~~~~

Control computation parameters:

.. code-block:: python

   # High accuracy computation
   result = stable_manifold.compute(
       step=0.01,                    # Smaller step for higher resolution
       integration_fraction=0.9,     # Integrate for 90% of period
       displacement=1e-6,            # Small displacement along eigenvector
       method="scipy",               # Integration method
       order=8,                      # Integration order
       energy_tol=1e-6,              # Energy conservation tolerance
       safe_distance=1e-2            # Safety distance from primaries
   )

Computation Parameters
~~~~~~~~~~~~~~~~~~~~~~

Key parameters for manifold computation:

.. code-block:: python

   # Step size: controls resolution along generating orbit
   result = stable_manifold.compute(step=0.02)  # 50 samples per period
   
   # Integration fraction: how long to integrate each trajectory
   result = stable_manifold.compute(integration_fraction=0.75)  # 75% of period
   
   # Displacement: initial displacement along eigenvector
   result = stable_manifold.compute(displacement=1e-6)  # Very small displacement
   
   # Integration method
   result = stable_manifold.compute(method="scipy", order=8)  # High order

Manifold Results
----------------

Computed manifolds provide access to trajectory data:

Accessing Trajectories
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get trajectory data
   trajectories = result.trajectories
   times = result.times
   poincare_section = result.poincare_section
   
   print(f"Number of trajectories: {len(trajectories)}")
   print(f"Trajectory lengths: {[len(traj) for traj in trajectories]}")
   
   # Access individual trajectory
   traj = trajectories[0]
   print(f"First trajectory shape: {traj.shape}")

Poincare Section
~~~~~~~~~~~~~~~~

Manifolds provide Poincare section data:

.. code-block:: python

   # Access Poincare section
   section_points = poincare_section.points
   section_labels = poincare_section.labels
   
   print(f"Poincare section points: {len(section_points)}")
   print(f"Section labels: {section_labels}")

Success Rate
~~~~~~~~~~~~

Monitor computation success:

.. code-block:: python

   # Check success rate
   success_rate = result.success_rate
   print(f"Success rate: {success_rate:.2%}")
   
   # Access failed trajectories
   failed_trajectories = result.failed_trajectories
   print(f"Failed trajectories: {len(failed_trajectories)}")

Invariant Tori
--------------

Invariant tori can be computed for certain orbits:

Creating Tori
~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms import InvariantTori
   
   # Create invariant torus
   torus = InvariantTori(halo)
   
   # Compute the torus
   torus.compute(
       scheme='linear',      # Linear scheme
       epsilon=1e-2,         # Perturbation parameter
       n_theta1=512,         # Grid resolution in theta1
       n_theta2=512          # Grid resolution in theta2
   )

Torus Parameters
~~~~~~~~~~~~~~~~

Control torus computation:

.. code-block:: python

   # Different schemes
   torus.compute(scheme='linear', epsilon=1e-2)
   
   # Grid resolution
   torus.compute(n_theta1=256, n_theta2=256)  # Lower resolution
   torus.compute(n_theta1=1024, n_theta2=1024)  # Higher resolution
   
   # Perturbation parameter
   torus.compute(epsilon=1e-1)   # Large perturbation
   torus.compute(epsilon=1e-4)   # Small perturbation

Torus Analysis
~~~~~~~~~~~~~~

Analyze computed tori:

.. code-block:: python

   # Check if torus was computed successfully
   if torus.is_computed:
       print("Torus computed successfully")
       
       # Access torus data
       torus_data = torus.torus_data
       print(f"Torus data shape: {torus_data.shape}")
       
       # Get torus properties
       print(f"Torus dimension: {torus.dimension}")
       print(f"Torus frequency: {torus.frequency}")

Visualization
-------------

Plot manifolds and tori:

Manifold Visualization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot manifold
   stable_manifold.plot()
   
   # Custom plotting
   fig = plt.figure(figsize=(12, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   # Plot trajectories
   for traj in result.trajectories:
       x = traj[:, 0]
       y = traj[:, 1]
       z = traj[:, 2]
       ax.plot(x, y, z, 'b-', alpha=0.6, linewidth=0.8)
   
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Stable Manifold')
   plt.show()

Torus Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot torus
   torus.plot()
   
   # Custom torus plotting
   if torus.is_computed:
       fig = plt.figure(figsize=(10, 8))
       ax = fig.add_subplot(111, projection='3d')
       
       # Plot torus surface
       torus_data = torus.torus_data
       ax.plot_surface(torus_data[:, :, 0], 
                      torus_data[:, :, 1], 
                      torus_data[:, :, 2], 
                      alpha=0.7, cmap='viridis')
       
       ax.set_xlabel('X')
       ax.set_ylabel('Y')
       ax.set_zlabel('Z')
       ax.set_title('Invariant Torus')
       plt.show()

Poincare Section Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot Poincare section
   fig, ax = plt.subplots(figsize=(8, 8))
   
   # Plot section points
   points = poincare_section.points
   ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
   
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_title('Poincare Section')
   ax.set_aspect('equal')
   plt.show()

Practical Examples
------------------

Earth-Moon L1 Halo Manifolds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   
   # Create system and orbit
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.3, zenith="southern")
   halo.correct(max_attempts=25)
   halo.propagate(steps=1000)
   
   # Create and compute manifolds
   stable_manifold = halo.manifold(stable=True, direction="positive")
   unstable_manifold = halo.manifold(stable=False, direction="negative")
   
   # Compute manifolds
   stable_result = stable_manifold.compute(
       step=0.02,
       integration_fraction=0.8,
       displacement=1e-6
   )
   
   unstable_result = unstable_manifold.compute(
       step=0.02,
       integration_fraction=0.8,
       displacement=1e-6
   )
   
   # Plot both manifolds
   stable_manifold.plot()
   unstable_manifold.plot()

Sun-Earth L2 Halo Tori
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sun-Earth system
   system = System.from_bodies("sun", "earth")
   l2 = system.get_libration_point(2)
   
   # Create L2 halo
   halo_l2 = l2.create_orbit("halo", amplitude_z=0.1, zenith="northern")
   halo_l2.correct()
   halo_l2.propagate()
   
   # Create and compute torus
   torus = InvariantTori(halo_l2)
   torus.compute(
       scheme='linear',
       epsilon=1e-2,
       n_theta1=512,
       n_theta2=512
   )
   
   # Plot torus
   torus.plot()

Multiple Manifold Types
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create all manifold types
   manifolds = {
       'stable_pos': halo.manifold(stable=True, direction="positive"),
       'stable_neg': halo.manifold(stable=True, direction="negative"),
       'unstable_pos': halo.manifold(stable=False, direction="positive"),
       'unstable_neg': halo.manifold(stable=False, direction="negative")
   }
   
   # Compute all manifolds
   results = {}
   for name, manifold in manifolds.items():
       results[name] = manifold.compute(
           step=0.05,
           integration_fraction=0.75
       )
       print(f"{name}: {results[name].success_rate:.2%} success rate")

Next Steps
----------

Once you understand manifolds and tori, you can:

- Analyze Poincare sections (see :doc:`guide_06_poincare`)
- Find heteroclinic connections (see :doc:`guide_06_poincare`)
- Use center manifold methods (see :doc:`guide_07_center_manifold`)

For advanced manifold analysis, see :doc:`guide_08_advanced`.
