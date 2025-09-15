Libration Point Analysis
========================

This guide covers the analysis of libration points (Lagrange points) in the Circular Restricted Three-Body Problem, including their computation, stability analysis, and properties.

Accessing Libration Points
--------------------------

Once you have a system, you can access its libration points:

.. code-block:: python

   from hiten import System
   
   system = System.from_bodies("earth", "moon")
   
   # Get individual libration points
   l1 = system.get_libration_point(1)
   l2 = system.get_libration_point(2)
   l3 = system.get_libration_point(3)
   l4 = system.get_libration_point(4)
   l5 = system.get_libration_point(5)
   
   # Or access all at once
   libration_points = system.libration_points
   print(f"Available points: {list(libration_points.keys())}")

Libration Point Properties
--------------------------

Each libration point provides access to key properties:

.. code-block:: python

   # Position in rotating frame
   print(f"L1 position: {l1.position}")
   print(f"L2 position: {l2.position}")
   
   # Energy and Jacobi constant
   print(f"L1 energy: {l1.energy}")
   print(f"L1 Jacobi constant: {l1.jacobi_constant}")
   
   # Stability analysis
   print(f"L1 is stable: {l1.is_stable}")
   print(f"L2 is stable: {l2.is_stable}")

Position Analysis
-----------------

Libration point positions are computed automatically and cached:

.. code-block:: python

   # Collinear points (L1, L2, L3)
   print(f"L1: {l1.position}")  # Between primaries
   print(f"L2: {l2.position}")  # Beyond smaller primary
   print(f"L3: {l3.position}")  # Beyond larger primary
   
   # Triangular points (L4, L5)
   print(f"L4: {l4.position}")  # Leading triangular point
   print(f"L5: {l5.position}")  # Trailing triangular point

Energy Analysis
---------------

Each libration point has associated energy and Jacobi constant:

.. code-block:: python

   # Energy (dimensionless)
   print(f"L1 energy: {l1.energy}")
   print(f"L2 energy: {l2.energy}")
   
   # Jacobi constant (CJ = -2*E)
   print(f"L1 Jacobi constant: {l1.jacobi_constant}")
   print(f"L2 Jacobi constant: {l2.jacobi_constant}")

Stability Analysis
------------------

HITEN provides comprehensive stability analysis for libration points:

.. code-block:: python

   # Basic stability check
   print(f"L1 stable: {l1.is_stable}")
   print(f"L2 stable: {l2.is_stable}")
   print(f"L3 stable: {l3.is_stable}")
   print(f"L4 stable: {l4.is_stable}")
   print(f"L5 stable: {l5.is_stable}")

Detailed Stability Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more detailed analysis, access eigenvalues and eigenvectors:

.. code-block:: python

   # Get stability eigenvalues
   stable_vals, unstable_vals, center_vals = l1.eigenvalues
   print(f"L1 stable eigenvalues: {stable_vals}")
   print(f"L1 unstable eigenvalues: {unstable_vals}")
   print(f"L1 center eigenvalues: {center_vals}")
   
   # Get corresponding eigenvectors
   stable_vecs, unstable_vecs, center_vecs = l1.eigenvectors
   print(f"L1 stable eigenvectors shape: {stable_vecs.shape}")
   print(f"L1 unstable eigenvectors shape: {unstable_vecs.shape}")
   print(f"L1 center eigenvectors shape: {center_vecs.shape}")

Linear Data
-----------

For advanced analysis, access the linearized system data:

.. code-block:: python

   # Get linear data for normal form analysis
   linear_data = l1.linear_data
   print(f"Linear data type: {type(linear_data)}")
   
   # Access specific linear invariants
   if hasattr(linear_data, 'lambda1'):
       print(f"Lambda1: {linear_data.lambda1}")
   if hasattr(linear_data, 'omega1'):
       print(f"Omega1: {linear_data.omega1}")
   if hasattr(linear_data, 'omega2'):
       print(f"Omega2: {linear_data.omega2}")

Collinear Points (L1, L2, L3)
------------------------------

Collinear points have special properties and methods:

.. code-block:: python

   # L1 point properties
   print(f"L1 gamma: {l1.gamma}")  # Distance ratio
   print(f"L1 sign: {l1.sign}")    # Sign convention
   print(f"L1 a: {l1.a}")          # Offset parameter
   
   # Linear modes
   lambda1, omega1, omega2 = l1.linear_modes
   print(f"L1 linear modes: lambda_1={lambda1}, omega_1={omega1}, omega_2={omega2}")

Triangular Points (L4, L5)
---------------------------

Triangular points have different stability characteristics:

.. code-block:: python

   # L4/L5 are typically stable for small mass ratios
   print(f"L4 stable: {l4.is_stable}")
   print(f"L5 stable: {l5.is_stable}")
   
   # They form equilateral triangles with the primaries
   print(f"L4 position: {l4.position}")
   print(f"L5 position: {l5.position}")

Center Manifold Access
----------------------

Libration points provide access to their center manifolds:

.. code-block:: python

   # Get center manifold for normal form analysis
   center_manifold = l1.get_center_manifold(degree=6)
   print(f"Center manifold degree: {center_manifold.degree}")
   
   # Compute the center manifold
   center_manifold.compute()
   
   # Access computed coefficients
   coefficients = center_manifold.coefficients()
   print(f"Center manifold coefficients shape: {coefficients.shape}")

Orbit Creation
--------------

Libration points can create periodic orbits:

.. code-block:: python

   # Create different types of orbits
   halo_orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   lyapunov_orbit = l1.create_orbit("lyapunov", amplitude_x=0.05)
   vertical_orbit = l1.create_orbit("vertical", initial_state=[...])
   
   print(f"Created halo orbit: {halo_orbit}")
   print(f"Created Lyapunov orbit: {lyapunov_orbit}")

System Integration
------------------

Libration points are integrated with the system's dynamical system:

.. code-block:: python

   # Access the underlying dynamical system
   dynsys = l1.dynsys
   print(f"Dynamical system: {dynsys}")
   
   # Access variational equations
   var_system = l1.var_eq_system
   print(f"Variational system: {var_system}")

Practical Examples
------------------

Earth-Moon L1 Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   
   # Earth-Moon system
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   print(f"Earth-Moon L1 position: {l1.position}")
   print(f"Mass parameter: {system.mu}")
   print(f"L1 stable: {l1.is_stable}")
   print(f"L1 Jacobi constant: {l1.jacobi_constant}")

Sun-Earth L2 Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sun-Earth system
   system = System.from_bodies("sun", "earth")
   l2 = system.get_libration_point(2)
   
   print(f"Sun-Earth L2 position: {l2.position}")
   print(f"Mass parameter: {system.mu}")
   print(f"L2 stable: {l2.is_stable}")

Custom System Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom mass parameter
   system = System.from_mu(0.1)  # 10% mass ratio
   l1 = system.get_libration_point(1)
   
   print(f"Custom L1 position: {l1.position}")
   print(f"Custom L1 stable: {l1.is_stable}")

Next Steps
----------

Once you understand libration points, you can:

- Propagate orbits around them (see :doc:`guide_03_propagation`)
- Create periodic orbits (see :doc:`guide_04_orbits`)
- Analyze center manifolds (see :doc:`guide_07_center_manifold`)

For more advanced libration point analysis, see :doc:`guide_08_advanced`.
