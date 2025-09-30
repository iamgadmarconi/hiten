Poincare Sections and Connections
==================================

This guide covers the creation and analysis of Poincare sections, manifold intersections, and heteroclinic connections in the Circular Restricted Three-Body Problem.

Poincare Sections
-----------------

Poincare sections are cross-sections of phase space that reveal the underlying dynamics of the system.

Creating Poincare Sections
~~~~~~~~~~~~~~~~~~~~~~~~~~

HITEN provides several types of Poincare sections:

Synodic Sections
~~~~~~~~~~~~~~~~

Synodic sections are defined in the rotating frame:

.. code-block:: python

   from hiten.system import SynodicMap, System
   
   # Create system and orbit
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct()
   halo.propagate()
   
   # Generate section from manifold
   stable_manifold = halo.manifold(stable=True, direction="positive")
   stable_manifold.compute()
   
   # Create synodic map from manifold
   synodic_map = SynodicMap(stable_manifold)
   
   # Compute section with configuration
   overrides = {
       "interp_kind": "cubic",
       "segment_refine": 30,
   }
   synodic_map.compute(
       section_axis="x",
       section_offset=0.8,
       plane_coords=("y", "z"),
       overrides=overrides
   )

Center Manifold Sections
~~~~~~~~~~~~~~~~~~~~~~~~

Center manifold sections are defined in the center manifold coordinates:

.. code-block:: python

   from hiten.algorithms.poincare import CenterManifoldMapConfig
   
   # Get center manifold
   center_manifold = l1.get_center_manifold(degree=6)
   center_manifold.compute()
   
   # Create Poincare map
   poincare_map = center_manifold.poincare_map(
       energy=0.7,                  # Energy level
       section_coord="p3",          # Section coordinate
       n_seeds=50,                  # Number of seed points
       n_iter=100,                  # Number of iterations
       seed_strategy="axis_aligned" # Seed strategy
   )

Section Configuration
---------------------

Control section parameters for different applications:

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple x-section
   synodic_map.compute(
       section_axis="x",
       section_offset=0.8,
       plane_coords=("y", "z")
   )
   
   # Simple y-section  
   synodic_map.compute(
       section_axis="y",
       section_offset=0.0,
       plane_coords=("x", "z")
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High resolution section
   overrides = {
       "interp_kind": "cubic",
       "segment_refine": 50,
   }
   synodic_map.compute(
       section_axis="x",
       section_offset=0.8,
       plane_coords=("y", "z"),
       overrides=overrides
   )

Section Analysis
----------------

Analyze computed sections:

Accessing Section Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get section points
   points = synodic_map.points
   labels = synodic_map.labels
   
   print(f"Number of section points: {len(points)}")
   print(f"Section labels: {labels}")
   
   # Access individual points
   for i, point in enumerate(points):
       print(f"Point {i}: {point}")

Section Properties
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Section properties
   print(f"Section axis: {synodic_map.section_axis}")
   print(f"Section offset: {synodic_map.section_offset}")
   print(f"Plane coordinates: {synodic_map.plane_coords}")
   
   # Bounds
   bounds = synodic_map.bounds
   print(f"Section bounds: {bounds}")

Heteroclinic Connections
------------------------

A heteroclinic connection is a path in phase space which joins two different equilibrium points.

Creating Connections
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.connections import ConnectionPipeline
   from hiten.algorithms.connections.config import _ConnectionConfig
   from hiten.algorithms.poincare import SynodicMapConfig
   
   # Create two orbits
   l1 = system.get_libration_point(1)
   l2 = system.get_libration_point(2)
   
   # L1 halo orbit
   halo_l1 = l1.create_orbit("halo", amplitude_z=0.5, zenith="southern")
   halo_l1.correct()
   halo_l1.propagate()
   
   # L2 halo orbit
   halo_l2 = l2.create_orbit("halo", amplitude_z=0.3663368, zenith="northern")
   halo_l2.correct()
   halo_l2.propagate()
   
   # Create manifolds
   manifold_l1 = halo_l1.manifold(stable=True, direction="positive")
   manifold_l1.compute(integration_fraction=0.9, step=0.005)
   
   manifold_l2 = halo_l2.manifold(stable=False, direction="negative")
   manifold_l2.compute(integration_fraction=1.0, step=0.005)

Once you created the manifolds, you can create a connection between them by configuring the section and search parameters.

.. code-block:: python

   # Create unified configuration
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=1 - system.mu,
       plane_coords=("y", "z"),
       interp_kind="cubic",
       segment_refine=30,
       tol_on_surface=1e-9,
       dedup_time_tol=1e-9,
       dedup_point_tol=1e-9,
       max_hits_per_traj=None,
       n_workers=None,
   )
   
   config = _ConnectionConfig(
       section=section_cfg,
       direction=None,
       delta_v_tol=1.0,
       ballistic_tol=1e-8,
       eps2d=1e-3,
   )
   
   # Create connection pipeline
   conn = ConnectionPipeline.with_default_engine(config=config)

Then you can solve for connections between the manifolds.

.. code-block:: python

   # Solve for connections
   conn.solve(manifold_l1, manifold_l2)
   
   # Check results
   print(f"Connections found: {conn}")
   
   # Plot results
   conn.plot(dark_mode=True)

Connection Analysis
-------------------

Analyze found connections:

Connection Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot and analyze results
   conn.plot(dark_mode=True)

Connection Classification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Connection results are displayed via plot
   conn.plot(dark_mode=True)

Connection Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot connections
   conn.plot(dark_mode=True)

Practical Examples
------------------

Earth-Moon L1-L2 Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms.connections import ConnectionPipeline
   from hiten.algorithms.connections.config import _ConnectionConfig
   from hiten.algorithms.poincare import SynodicMapConfig
   
   # Create system
   system = System.from_bodies("earth", "moon")
   mu = system.mu
   
   # Get libration points
   l1 = system.get_libration_point(1)
   l2 = system.get_libration_point(2)
   
   # Create orbits
   halo_l1 = l1.create_orbit("halo", amplitude_z=0.5, zenith="southern")
   halo_l1.correct()
   halo_l1.propagate()
   
   halo_l2 = l2.create_orbit("halo", amplitude_z=0.3663368, zenith="northern")
   halo_l2.correct()
   halo_l2.propagate()
   
   # Create manifolds
   manifold_l1 = halo_l1.manifold(stable=True, direction="positive")
   manifold_l1.compute(integration_fraction=0.9, step=0.005)
   
   manifold_l2 = halo_l2.manifold(stable=False, direction="negative")
   manifold_l2.compute(integration_fraction=1.0, step=0.005)
   
   # Configure connection
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=1 - mu,
       plane_coords=("y", "z"),
       interp_kind="cubic",
       segment_refine=30,
       tol_on_surface=1e-9,
       dedup_time_tol=1e-9,
       dedup_point_tol=1e-9
   )
   
   config = _ConnectionConfig(
       section=section_cfg,
       direction=None,
       delta_v_tol=1.0,
       ballistic_tol=1e-8,
       eps2d=1e-3,
   )
   
   # Find connections
   conn = ConnectionPipeline.with_default_engine(config=config)
   conn.solve(manifold_l1, manifold_l2)
   
   # Display results
   print(conn)
   conn.plot(dark_mode=True)

Sun-Earth L1-L2 Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sun-Earth system
   system = System.from_bodies("sun", "earth")
   mu = system.mu
   
   # Get libration points
   l1 = system.get_libration_point(1)
   l2 = system.get_libration_point(2)
   
   # Create orbits
   halo_l1 = l1.create_orbit("halo", amplitude_z=0.1, zenith="southern")
   halo_l1.correct()
   halo_l1.propagate()
   
   halo_l2 = l2.create_orbit("halo", amplitude_z=0.1, zenith="northern")
   halo_l2.correct()
   halo_l2.propagate()
   
   # Create manifolds
   manifold_l1 = halo_l1.manifold(stable=True, direction="positive")
   manifold_l1.compute(integration_fraction=0.8, step=0.01)
   
   manifold_l2 = halo_l2.manifold(stable=False, direction="negative")
   manifold_l2.compute(integration_fraction=0.8, step=0.01)
   
   # Configure connection
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=1 - mu,
       plane_coords=("y", "z"),
       interp_kind="cubic",
       segment_refine=20,
       tol_on_surface=1e-10,
       dedup_time_tol=1e-10,
       dedup_point_tol=1e-10
   )
   
   config = _ConnectionConfig(
       section=section_cfg,
       direction=None,
       delta_v_tol=0.1,
       ballistic_tol=1e-10,
       eps2d=1e-4,
   )
   
   # Find connections
   conn = ConnectionPipeline.with_default_engine(config=config)
   conn.solve(manifold_l1, manifold_l2)
   
   # Display results
   print(conn)
   conn.plot()

Next Steps
----------

Once you understand Poincare sections and connections, you can:

- Use center manifold methods (see :doc:`guide_07_center_manifold`)
- Perform advanced dynamical analysis (see :doc:`guide_16_connections`)
- Create custom systems (see :doc:`guide_17_dynamical_systems`)

For more advanced connection analysis, see :doc:`guide_16_connections`.
