Advanced Connection Analysis and Custom Detection
=================================================

This guide covers HITEN's connection analysis capabilities for finding heteroclinic and homoclinic connections between invariant manifolds, including custom connection detection algorithms.

Understanding Connections
-------------------------------

Connections are trajectories that link different dynamical structures, representing natural pathways for low-energy transfers in the CR3BP.

Basic Connection Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms.connections import Connection, SearchConfig
   from hiten.algorithms.poincare import SynodicMapConfig
   import numpy as np

   system = System.from_bodies("earth", "moon")
   mu = system.mu

   # Create two different halo orbits
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

   # Compute manifolds
   manifold_l1 = halo_l1.manifold(stable=True, direction="positive")
   manifold_l1.compute(integration_fraction=0.9, step=0.005)

   manifold_l2 = halo_l2.manifold(stable=False, direction="negative")
   manifold_l2.compute(integration_fraction=1.0, step=0.005)

Connection Search Configuration
------------------------------------

Configure the connection search parameters:

Poincare Section Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Define the Poincare section for connection search
   section_cfg = SynodicMapConfig(
       section_axis="x",                    # Intersect when x = section_offset
       section_offset=1 - mu,              # Near the secondary body
       plane_coords=("y", "z"),            # Plot y vs z on the section
       interp_kind="cubic",                # Interpolation method
       segment_refine=30,                  # Refinement for intersection detection
       newton_max_iter=10,                 # Newton iterations for refinement
       tol_on_surface=1e-9,                # Tolerance for surface intersection
       dedup_time_tol=1e-9,                # Time tolerance for deduplication
       dedup_point_tol=1e-9                # Point tolerance for deduplication
   )

Search Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configure the connection search
   search_cfg = SearchConfig(
       delta_v_tol=1.0,                    # Maximum delta-V for valid connections
       ballistic_tol=1e-8,                 # Tolerance for ballistic connections
       eps2d=1e-3,                         # 2D distance tolerance
       min_connection_time=0.1,            # Minimum connection time
       max_connection_time=10.0            # Maximum connection time
   )

Finding Connections
-------------------------

Search for connections between manifolds:

.. code-block:: python

   # Create connection object
   conn = Connection(
       section=section_cfg,
       direction=None,                     # Search in both directions
       search_cfg=search_cfg
   )

   # Solve for connections
   conn.solve(manifold_l1, manifold_l2)

   # Display results
   print(f"Found {len(conn.connections)} connections")
   print(f"Search completed: {conn.solved}")

   # Access connection details
   if conn.connections:
       for i, connection in enumerate(conn.connections):
           print(f"Connection {i+1}:")
           print(f"  Delta-V: {connection.delta_v}")
           print(f"  Transfer time: {connection.transfer_time}")
           print(f"  Initial state: {connection.initial_state}")
           print(f"  Final state: {connection.final_state}")

Connection Analysis
-------------------------

Analyze found connections:

Connection Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze connection properties
   if conn.connections:
       connection = conn.connections[0]  # Take first connection
       
       print(f"Connection analysis:")
       print(f"  Delta-V required: {connection.delta_v:.6f}")
       print(f"  Transfer time: {connection.transfer_time:.6f}")
       print(f"  Energy change: {connection.energy_change:.6f}")
       
       # Check if connection is ballistic
       if connection.is_ballistic:
           print("  This is a ballistic connection (no delta-V required)")
       else:
           print(f"  Delta-V required: {connection.delta_v:.6f}")

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics.utils.energy import crtbp_energy

   # Analyze energy along connection
   if conn.connections:
       connection = conn.connections[0]
       
       # Get trajectory points
       trajectory = connection.trajectory
       times = connection.times
       
       # Compute energy along trajectory
       energies = [crtbp_energy(state, mu) for state in trajectory]
       
       # Check energy conservation
       initial_energy = energies[0]
       final_energy = energies[-1]
       energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
       
       print(f"Energy conservation error: {energy_error:.2e}")
       
       # Plot energy evolution
       import matplotlib.pyplot as plt
       plt.figure(figsize=(10, 6))
       plt.plot(times, energies, 'b-', linewidth=2)
       plt.xlabel('Time')
       plt.ylabel('Energy')
       plt.title('Energy Evolution Along Connection')
       plt.grid(True)
       plt.show()

Custom Connection Detection
---------------------------------

Create custom connection detection algorithms:

Basic Custom Detector
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.connections.base import ConnectionBase
   import numpy as np

   class CustomConnectionDetector(ConnectionBase):
       """Custom connection detection algorithm."""
       
       def __init__(self, tolerance=1e-6, max_iterations=100):
           super().__init__()
           self.tolerance = tolerance
           self.max_iterations = max_iterations
       
       def find_connections(self, manifold1, manifold2):
           """Find connections between two manifolds."""
           
           connections = []
           
           # Get manifold trajectories
           traj1 = manifold1.manifold_result.trajectories
           traj2 = manifold2.manifold_result.trajectories
           
           # Simple distance-based detection
           for i, t1 in enumerate(traj1):
               for j, t2 in enumerate(traj2):
                   # Check if trajectories are close
                   min_distance = self._compute_minimum_distance(t1, t2)
                   
                   if min_distance < self.tolerance:
                       # Found potential connection
                       connection = self._create_connection(t1, t2, min_distance)
                       connections.append(connection)
           
           return connections
       
       def _compute_minimum_distance(self, traj1, traj2):
           """Compute minimum distance between two trajectories."""
           
           min_dist = float('inf')
           
           for state1 in traj1:
               for state2 in traj2:
                   dist = np.linalg.norm(state1[:3] - state2[:3])  # Position only
                   min_dist = min(min_dist, dist)
           
           return min_dist
       
       def _create_connection(self, traj1, traj2, distance):
           """Create connection object from trajectories."""
           
           # Simple connection creation
           connection = type('Connection', (), {
               'trajectory1': traj1,
               'trajectory2': traj2,
               'distance': distance,
               'delta_v': 0.0,  # Simplified
               'transfer_time': 0.0  # Simplified
           })()
           
           return connection

   # Use custom detector
   custom_detector = CustomConnectionDetector(tolerance=1e-4)
   custom_connections = custom_detector.find_connections(manifold_l1, manifold_l2)
   
   print(f"Custom detector found {len(custom_connections)} connections")

Advanced Custom Detector
~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated detection, implement optimization-based methods:

.. code-block:: python

   from scipy.optimize import minimize

   class AdvancedConnectionDetector(ConnectionBase):
       """Optimization-based connection detection."""
       
       def __init__(self, tolerance=1e-6, max_iterations=100):
           super().__init__()
           self.tolerance = tolerance
           self.max_iterations = max_iterations
       
       def find_connections(self, manifold1, manifold2):
           """Find connections using optimization."""
           
           connections = []
           
           # Get manifold data
           traj1 = manifold1.manifold_result.trajectories
           traj2 = manifold2.manifold_result.trajectories
           
           # Try multiple starting points
           for i in range(0, len(traj1), 10):  # Sample every 10th trajectory
               for j in range(0, len(traj2), 10):
                   
                   # Initial guess
                   x0 = [i, j]  # Indices into trajectories
                   
                   # Objective function: minimize distance
                   def objective(x):
                       idx1, idx2 = int(x[0]), int(x[1])
                       idx1 = max(0, min(idx1, len(traj1)-1))
                       idx2 = max(0, min(idx2, len(traj2)-1))
                       
                       state1 = traj1[idx1]
                       state2 = traj2[idx2]
                       return np.linalg.norm(state1[:3] - state2[:3])
                   
                   # Bounds
                   bounds = [(0, len(traj1)-1), (0, len(traj2)-1)]
                   
                   # Optimize
                   result = minimize(
                       objective, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': self.max_iterations}
                   )
                   
                   if result.success and result.fun < self.tolerance:
                       connection = self._create_optimized_connection(
                           traj1, traj2, result.x, result.fun
                       )
                       connections.append(connection)
           
           return connections
       
       def _create_optimized_connection(self, traj1, traj2, indices, distance):
           """Create connection from optimization result."""
           
           idx1, idx2 = int(indices[0]), int(indices[1])
           state1 = traj1[idx1]
           state2 = traj2[idx2]
           
           connection = type('Connection', (), {
               'trajectory1': traj1,
               'trajectory2': traj2,
               'indices': (idx1, idx2),
               'distance': distance,
               'delta_v': self._compute_delta_v(state1, state2),
               'transfer_time': self._compute_transfer_time(state1, state2)
           })()
           
           return connection
       
       def _compute_delta_v(self, state1, state2):
           """Compute delta-V required for connection."""
           # Simplified delta-V computation
           return np.linalg.norm(state1[3:6] - state2[3:6])
       
       def _compute_transfer_time(self, state1, state2):
           """Compute transfer time for connection."""
           # Simplified transfer time computation
           return 1.0  # Placeholder

Connection Visualization
------------------------------

Visualize connections and their properties:

.. code-block:: python

   def plot_connections(conn, manifold1, manifold2):
       """Plot connections between manifolds."""
       
       import matplotlib.pyplot as plt
       from mpl_toolkits.mplot3d import Axes3D
       
       fig = plt.figure(figsize=(15, 5))
       
       # 3D plot
       ax1 = fig.add_subplot(131, projection='3d')
       
       # Plot manifold trajectories
       for traj in manifold1.manifold_result.trajectories[:10]:  # Sample
           ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', alpha=0.3)
       
       for traj in manifold2.manifold_result.trajectories[:10]:  # Sample
           ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', alpha=0.3)
       
       # Plot connections
       if conn.connections:
           for connection in conn.connections:
               traj = connection.trajectory
               ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g-', linewidth=3)
       
       ax1.set_xlabel('X')
       ax1.set_ylabel('Y')
       ax1.set_zlabel('Z')
       ax1.set_title('3D Connection Visualization')
       
       # Poincare section plot
       ax2 = fig.add_subplot(132)
       
       # Plot section points
       if hasattr(conn, 'section_points1'):
           ax2.scatter(conn.section_points1[:, 0], conn.section_points1[:, 1], 
                      c='b', alpha=0.5, s=1, label='Manifold 1')
       
       if hasattr(conn, 'section_points2'):
           ax2.scatter(conn.section_points2[:, 0], conn.section_points2[:, 1], 
                      c='r', alpha=0.5, s=1, label='Manifold 2')
       
       ax2.set_xlabel('Y')
       ax2.set_ylabel('Z')
       ax2.set_title('Poincare Section')
       ax2.legend()
       ax2.grid(True)
       
       # Connection properties
       ax3 = fig.add_subplot(133)
       
       if conn.connections:
           delta_vs = [c.delta_v for c in conn.connections]
           transfer_times = [c.transfer_time for c in conn.connections]
           
           ax3.scatter(delta_vs, transfer_times, c='g', s=50)
           ax3.set_xlabel('Delta-V')
           ax3.set_ylabel('Transfer Time')
           ax3.set_title('Connection Properties')
           ax3.grid(True)
       
       plt.tight_layout()
       plt.show()

   # Plot connections
   plot_connections(conn, manifold_l1, manifold_l2)

Next Steps
----------

Once you understand connection analysis, you can:

- Learn about advanced integration techniques (see :doc:`guide_10_integrators`)
- Explore correction methods (see :doc:`guide_11_correction`)
- Study continuation algorithms (see :doc:`guide_12_continuation`)

For more advanced connection techniques, see the HITEN source code in :mod:`hiten.algorithms.connections`.
