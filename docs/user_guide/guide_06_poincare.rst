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

   from hiten.algorithms.poincare import SynodicMap, SynodicMapConfig
   from hiten import System
   
   # Create system and orbit
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct()
   halo.propagate()
   
   # Create synodic section configuration
   section_cfg = SynodicMapConfig(
       section_axis="x",           # Section perpendicular to x-axis
       section_offset=0.8,         # At x = 0.8
       plane_coords=("y", "z"),    # Plot y vs z
       interp_kind="cubic",        # Cubic interpolation
       segment_refine=30,          # Refinement factor
       tol_on_surface=1e-9,        # Surface tolerance
       dedup_time_tol=1e-9,        # Time deduplication tolerance
       dedup_point_tol=1e-9,       # Point deduplication tolerance
       max_hits_per_traj=None,     # No limit on hits per trajectory
       n_workers=None              # Use all available workers
   )
   
   # Create synodic map
   synodic_map = SynodicMap(section_cfg)
   
   # Generate section from manifold
   stable_manifold = halo.manifold(stable=True, direction="positive")
   stable_manifold.compute()
   
   synodic_map.from_manifold(stable_manifold)

Center Manifold Sections
~~~~~~~~~~~~~~~~~~~~~~~~

Center manifold sections are defined in the center manifold coordinates:

.. code-block:: python

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
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=0.8,
       plane_coords=("y", "z")
   )
   
   # Simple y-section
   section_cfg = SynodicMapConfig(
       section_axis="y",
       section_offset=0.0,
       plane_coords=("x", "z")
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High resolution section
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=0.8,
       plane_coords=("y", "z"),
       interp_kind="cubic",
       segment_refine=50,          # Higher refinement
       tol_on_surface=1e-12,       # Higher accuracy
       dedup_time_tol=1e-12,
       dedup_point_tol=1e-12
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

Heteroclinic connections are trajectories that connect different invariant manifolds.

Creating Connections
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.connections import Connection, SearchConfig
   
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

Connection Configuration
~~~~~~~~~~~~~~~~~~~~~~~

Configure connection search parameters:

.. code-block:: python

   # Section configuration
   section_cfg = SynodicMapConfig(
       section_axis="x",
       section_offset=1 - system.mu,  # Near secondary body
       plane_coords=("y", "z"),
       interp_kind="cubic",
       segment_refine=30,
       tol_on_surface=1e-9,
       dedup_time_tol=1e-9,
       dedup_point_tol=1e-9
   )
   
   # Search configuration
   search_cfg = SearchConfig(
       delta_v_tol=1,           # Delta-V tolerance
       ballistic_tol=1e-8,      # Ballistic tolerance
       eps2d=1e-3               # 2D distance tolerance
   )
   
   # Create connection
   connection = Connection(
       section=section_cfg,
       direction=None,           # Both directions
       search_cfg=search_cfg
   )

Finding Connections
~~~~~~~~~~~~~~~~~~~

Search for connections between manifolds:

.. code-block:: python

   # Solve for connections
   connection.solve(manifold_l1, manifold_l2)
   
   # Check results
   print(f"Connections found: {len(connection.results)}")
   
   # Access results
   for i, result in enumerate(connection.results):
       print(f"Connection {i}:")
       print(f"  Delta-V: {result.delta_v}")
       print(f"  Point: {result.point2d}")
       print(f"  Type: {result.transfer_type}")

Connection Analysis
-------------------

Analyze found connections:

Connection Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access connection results
   results = connection.results
   
   # Delta-V requirements
   delta_vs = [r.delta_v for r in results]
   print(f"Delta-V range: {min(delta_vs):.2e} to {max(delta_vs):.2e}")
   
   # Transfer types
   transfer_types = [r.transfer_type for r in results]
   print(f"Transfer types: {set(transfer_types)}")

Connection Classification
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Classify connections
   ballistic_connections = [r for r in results if r.transfer_type == "ballistic"]
   impulsive_connections = [r for r in results if r.transfer_type == "impulsive"]
   
   print(f"Ballistic connections: {len(ballistic_connections)}")
   print(f"Impulsive connections: {len(impulsive_connections)}")

Visualization
-------------

Plot Poincare sections and connections:

Section Visualization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot synodic section
   synodic_map.plot()
   
   # Custom section plotting
   fig, ax = plt.subplots(figsize=(10, 8))
   
   points = synodic_map.points
   ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.6)
   
   ax.set_xlabel('Y')
   ax.set_ylabel('Z')
   ax.set_title('Poincare Section')
   ax.set_aspect('equal')
   plt.show()

Connection Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot connections
   connection.plot(dark_mode=True)
   
   # Custom connection plotting
   fig, ax = plt.subplots(figsize=(12, 8))
   
   # Plot section points
   points_src = connection._last_source_section.points
   points_tgt = connection._last_target_section.points
   
   ax.scatter(points_src[:, 0], points_src[:, 1], s=1, alpha=0.6, label='Source')
   ax.scatter(points_tgt[:, 0], points_tgt[:, 1], s=1, alpha=0.6, label='Target')
   
   # Plot connections
   for result in connection.results:
       ax.scatter(result.point2d[0], result.point2d[1], 
                 s=50, c='red', marker='x')
   
   ax.set_xlabel('Y')
   ax.set_ylabel('Z')
   ax.set_title('Heteroclinic Connections')
   ax.legend()
   plt.show()

Practical Examples
------------------

Earth-Moon L1-L2 Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms.connections import Connection, SearchConfig
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
   
   search_cfg = SearchConfig(
       delta_v_tol=1,
       ballistic_tol=1e-8,
       eps2d=1e-3
   )
   
   # Find connections
   connection = Connection(section=section_cfg, search_cfg=search_cfg)
   connection.solve(manifold_l1, manifold_l2)
   
   # Display results
   print(connection)
   connection.plot(dark_mode=True)

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
   
   search_cfg = SearchConfig(
       delta_v_tol=0.1,
       ballistic_tol=1e-10,
       eps2d=1e-4
   )
   
   # Find connections
   connection = Connection(section=section_cfg, search_cfg=search_cfg)
   connection.solve(manifold_l1, manifold_l2)
   
   # Display results
   print(connection)
   connection.plot()

Next Steps
----------

Once you understand Poincare sections and connections, you can:

- Use center manifold methods (see :doc:`guide_07_center_manifold`)
- Perform advanced dynamical analysis (see :doc:`guide_16_connections`)
- Create custom systems (see :doc:`guide_17_dynamical_systems`)

For more advanced connection analysis, see :doc:`guide_16_connections`.
