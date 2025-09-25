Center Manifold Analysis
=========================

This guide covers the creation and analysis of center manifolds in the Circular Restricted Three-Body Problem, including normal form theory, center manifold reduction, and Poincare maps.

Center Manifold Theory
----------------------

Center manifolds are invariant manifolds associated with the center directions of libration points. They provide a powerful framework for understanding the local dynamics and finding periodic orbits.

Creating Center Manifolds
-------------------------

Center manifolds are created from libration points:

.. code-block:: python

   from hiten import System
   
   # Create system and libration point
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create center manifold
   center_manifold = l1.get_center_manifold(degree=6)
   
   # Compute the center manifold
   center_manifold.compute()
   
   print(f"Center manifold degree: {center_manifold.degree}")
   print(f"Center manifold computed: {center_manifold.is_computed}")

Center Manifold Degrees
~~~~~~~~~~~~~~~~~~~~~~~

Control the degree of the center manifold expansion:

.. code-block:: python

   # Low degree (faster, less accurate)
   cm_low = l1.get_center_manifold(degree=3)
   cm_low.compute()
   
   # Medium degree (balanced)
   cm_med = l1.get_center_manifold(degree=6)
   cm_med.compute()
   
   # High degree (slower, more accurate)
   cm_high = l1.get_center_manifold(degree=10)
   cm_high.compute()

Center Manifold Properties
--------------------------

Access center manifold properties and data:

Basic Properties
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic properties
   print(f"Degree: {center_manifold.degree}")
   print(f"Libration point: {center_manifold.libration_point}")
   print(f"Mass parameter: {center_manifold.mu}")
   print(f"Position: {center_manifold.position}")

Coefficients
~~~~~~~~~~~~

Access the computed coefficients:

.. code-block:: python

   # Get coefficients
   coefficients = center_manifold.coefficients()
   print(f"Coefficients shape: {coefficients.shape}")
   
   # Access specific coefficients
   print(f"First few coefficients: {coefficients[:10]}")

Normal Form Data
~~~~~~~~~~~~~~~~

Access normal form transformation data:

.. code-block:: python

   # Get normal form data
   normal_form = center_manifold.normal_form
   print(f"Normal form type: {type(normal_form)}")
   
   # Access transformation matrices
   if hasattr(normal_form, 'C'):
       print(f"Transformation matrix C shape: {normal_form.C.shape}")
   if hasattr(normal_form, 'Cinv'):
       print(f"Inverse transformation matrix shape: {normal_form.Cinv.shape}")

Center Manifold Coordinates
---------------------------

Work with center manifold coordinates:

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Transform from center manifold to physical coordinates
   cm_coords = np.array([0.1, 0.05, 0.0, 0.0])  # [q1, q2, p1, p2]
   physical_coords = center_manifold.to_synodic(cm_coords)
   print(f"Physical coordinates: {physical_coords}")
   
   # Transform from physical to center manifold coordinates
   physical_state = np.array([0.8, 0.0, 0.1, 0.0, 0.15, 0.0])  # [x, y, z, vx, vy, vz]
   cm_state = center_manifold.cm(physical_state)
   print(f"Center manifold coordinates: {cm_state}")

Initial Conditions
~~~~~~~~~~~~~~~~~~

Generate initial conditions for periodic orbits:

.. code-block:: python

   # Generate initial conditions
   ic = center_manifold.to_synodic([0.0, 0.0], 0.6, "q3")
   print(f"Initial conditions: {ic}")
   
   # Different coordinate choices
   ic_q1 = center_manifold.to_synodic([0.1, 0.0], 0.6, "q1")
   ic_q2 = center_manifold.to_synodic([0.0, 0.1], 0.6, "q2")
   ic_p1 = center_manifold.to_synodic([0.0, 0.0], 0.6, "p1")
   ic_p2 = center_manifold.to_synodic([0.0, 0.0], 0.6, "p2")

Poincare Maps
-------------

Create Poincare maps from center manifolds:

Basic Poincare Maps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create Poincare map
   poincare_map = center_manifold.poincare_map(
       energy=0.7,                 # Energy level
       section_coord="p3",         # Section coordinate
       n_seeds=50,                 # Number of seed points
       n_iter=100,                 # Number of iterations
       seed_strategy="axis_aligned" # Seed strategy
   )
   
   print(f"Poincare map created: {poincare_map is not None}")

Poincare Map Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control Poincare map parameters:

.. code-block:: python

   # High resolution map
   poincare_map = center_manifold.poincare_map(
       energy=0.7,
       section_coord="p3",
       n_seeds=100,                # More seeds
       n_iter=200,                 # More iterations
       seed_strategy="axis_aligned"
   )
   
   # Different section coordinates
   poincare_map_q1 = center_manifold.poincare_map(
       energy=0.7,
       section_coord="q1",
       n_seeds=50,
       n_iter=100
   )
   
   poincare_map_q2 = center_manifold.poincare_map(
       energy=0.7,
       section_coord="q2",
       n_seeds=50,
       n_iter=100
   )

Poincare Map Analysis
---------------------

Analyze computed Poincare maps:

Map Properties
~~~~~~~~~~~~~~

.. code-block:: python

   # Map properties
   print(f"Energy level: {poincare_map.energy}")
   print(f"Section coordinate: {poincare_map.section_coord}")
   print(f"Number of seeds: {poincare_map.n_seeds}")
   print(f"Number of iterations: {poincare_map.n_iter}")

Map Data
~~~~~~~~

.. code-block:: python

   # Access map data
   map_data = poincare_map.map_data
   print(f"Map data shape: {map_data.shape}")
   
   # Access seed points
   seeds = poincare_map.seeds
   print(f"Seed points shape: {seeds.shape}")
   
   # Access iteration data
   iterations = poincare_map.iterations
   print(f"Iterations shape: {iterations.shape}")

Map Visualization
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot Poincare map
   poincare_map.plot(axes=("p2", "q3"))
   
   # Custom plotting
   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(figsize=(10, 8))
   
   # Plot map data
   map_data = poincare_map.map_data
   ax.scatter(map_data[:, 0], map_data[:, 1], s=1, alpha=0.6)
   
   ax.set_xlabel('p2')
   ax.set_ylabel('q3')
   ax.set_title('Poincare Map')
   ax.set_aspect('equal')
   plt.show()

Finding Orbits in Center Manifold
---------------------------------

Use center manifolds to find periodic orbits:

Analytical Initial Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate initial conditions for different orbit types
   
   # Halo orbit
   halo_ic = center_manifold.to_synodic([0.0, 0.0], 0.6, "q3")
   print(f"Halo initial conditions: {halo_ic}")
   
   # Lyapunov orbit
   lyapunov_ic = center_manifold.to_synodic([0.1, 0.0], 0.6, "q1")
   print(f"Lyapunov initial conditions: {lyapunov_ic}")
   
   # Vertical orbit
   vertical_ic = center_manifold.to_synodic([0.0, 0.0], 0.6, "p2")
   print(f"Vertical initial conditions: {vertical_ic}")

Orbit Creation from Center Manifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create orbits using center manifold initial conditions
   
   # Halo orbit
   halo = l1.create_orbit("halo", initial_state=halo_ic)
   halo.correct()
   halo.propagate()
   
   # Lyapunov orbit
   lyapunov = l1.create_orbit("lyapunov", initial_state=lyapunov_ic)
   lyapunov.correct()
   lyapunov.propagate()
   
   # Vertical orbit
   vertical = l1.create_orbit("vertical", initial_state=vertical_ic)
   vertical.correct()
   vertical.propagate()

Practical Examples
------------------

Earth-Moon L1 Center Manifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   
   # Create system
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create center manifold
   center_manifold = l1.get_center_manifold(degree=6)
   center_manifold.compute()
   
   # Create Poincare map
   poincare_map = center_manifold.poincare_map(
       energy=0.7,
       section_coord="p3",
       n_seeds=50,
       n_iter=100
   )
   
   # Plot map
   poincare_map.plot(axes=("p2", "q3"))
   
   # Generate orbits
   halo_ic = center_manifold.to_synodic([0.0, 0.0], 0.6, "q3")
   halo = l1.create_orbit("halo", initial_state=halo_ic)
   halo.correct()
   halo.propagate()
   halo.plot()

Sun-Earth L2 Center Manifold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sun-Earth system
   system = System.from_bodies("sun", "earth")
   l2 = system.get_libration_point(2)
   
   # Create center manifold
   center_manifold = l2.get_center_manifold(degree=8)
   center_manifold.compute()
   
   # Create Poincare map
   poincare_map = center_manifold.poincare_map(
       energy=0.5,
       section_coord="q1",
       n_seeds=100,
       n_iter=200
   )
   
   # Plot map
   poincare_map.plot(axes=("q2", "p1"))
   
   # Generate orbits
   lyapunov_ic = center_manifold.to_synodic([0.05, 0.0], 0.5, "q1")
   lyapunov = l2.create_orbit("lyapunov", initial_state=lyapunov_ic)
   lyapunov.correct()
   lyapunov.propagate()
   lyapunov.plot()

Custom Center Manifold Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom mass parameter
   system = System.from_mu(0.1)  # 10% mass ratio
   l1 = system.get_libration_point(1)
   
   # Create center manifold
   center_manifold = l1.get_center_manifold(degree=5)
   center_manifold.compute()
   
   # Generate multiple orbits
   energies = np.linspace(0.3, 0.7, 5)
   
   for energy in energies:
       ic = center_manifold.to_synodic([0.0, 0.0], energy, "q3")
       orbit = l1.create_orbit("halo", initial_state=ic)
       orbit.correct()
       orbit.propagate()
       
       print(f"Energy {energy:.2f}: Period {orbit.period:.3f}")

Next Steps
----------

Once you understand center manifolds, you can:

- Perform advanced dynamical analysis (see :doc:`guide_16_connections`)
- Create custom systems (see :doc:`guide_17_dynamical_systems`)
- Use advanced continuation methods (see :doc:`guide_12_continuation`)

For more advanced center manifold analysis, see :doc:`guide_14_polynomial`.
