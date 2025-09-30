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

Computing Manifolds
-------------------

Once created, manifolds must be computed to generate trajectories:

Basic Computation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute stable manifold
   stable_manifold.compute()
   
   # Access computed trajectories
   trajectories = stable_manifold.trajectories
   print(f"Number of trajectories: {len(trajectories)}")

Advanced Computation
~~~~~~~~~~~~~~~~~~~~

Control computation parameters:

.. code-block:: python

   # High accuracy computation
   stable_manifold.compute(
       step=0.01,                    # Smaller step for higher resolution
       integration_fraction=0.9,     # Integrate for 90% of period
       displacement=1e-6,            # Small displacement along eigenvector
       dt=1e-3,                      # Integration time step
       method="adaptive",            # Integration method
       order=8                       # Integration order
   )

Manifold Results
----------------

Computed manifolds provide access to trajectory data:

Accessing Trajectories
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get trajectory data after computing
   trajectories = stable_manifold.trajectories

   print(f"Number of trajectories: {len(trajectories)}")
   print(f"Trajectory shapes: {[traj.states.shape for traj in trajectories]}")
   
   # Access individual trajectory
   traj = trajectories[0]
   print(f"First trajectory time range: {traj.t0} to {traj.tf}")
   print(f"First trajectory states shape: {traj.states.shape}")

Invariant Tori
--------------

Invariant tori can be computed for certain orbits:

Creating Tori
~~~~~~~~~~~~~

.. code-block:: python

   from hiten import InvariantTori
   
   # Create invariant torus
   torus = InvariantTori(halo)
   
   # Compute the torus
   torus.compute(
       epsilon=1e-2,         # Perturbation parameter
       n_theta1=512,         # Grid resolution in theta1
       n_theta2=512          # Grid resolution in theta2
   )

Torus Parameters
~~~~~~~~~~~~~~~~

Control torus computation:

.. code-block:: python

   # Different schemes
   torus.compute(epsilon=1e-2)
   
   # Grid resolution
   torus.compute(n_theta1=256, n_theta2=256)  # Lower resolution
   torus.compute(n_theta1=1024, n_theta2=1024)  # Higher resolution
   
   # Perturbation parameter
   torus.compute(epsilon=1e-1)   # Large perturbation
   torus.compute(epsilon=1e-4)   # Small perturbation

Examples
--------

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
   stable_manifold.compute(
       step=0.02,
       integration_fraction=0.8,
       displacement=1e-6
   )
   
   unstable_manifold.compute(
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
   for name, manifold in manifolds.items():
       manifold.compute(
           step=0.05,
           integration_fraction=0.75
       )
       trajectories = manifold.trajectories
       print(f"{name}: {len(trajectories)} trajectories computed")

Next Steps
----------

Once you understand manifolds and tori, you can:

- Analyze Poincare sections (see :doc:`guide_06_poincare`)
- Find heteroclinic connections (see :doc:`guide_06_poincare`)
- Use center manifold methods (see :doc:`guide_07_center_manifold`)

For advanced manifold analysis, see :doc:`guide_16_connections`.
