Connection Analysis and Custom Detection
=================================================

This guide covers HITEN's connection analysis capabilities for finding heteroclinic and homoclinic connections between invariant manifolds, including custom connection detection algorithms.

Understanding Connections
-------------------------------

Connections are trajectories that link different dynamical structures, representing natural pathways for low-energy transfers in the CR3BP.

Basic Connection Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms.connections import ConnectionPipeline
   from hiten.algorithms.connections.config import _ConnectionConfig
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
       tol_on_surface=1e-9,                # Tolerance for surface intersection
       dedup_time_tol=1e-9,                # Time tolerance for deduplication
       dedup_point_tol=1e-9,               # Point tolerance for deduplication
       max_hits_per_traj=None,             # Maximum hits per trajectory
       n_workers=None                      # Number of workers for parallel processing
   )

Search Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create unified configuration with all parameters
   config = _ConnectionConfig(
       section=section_cfg,                # Synodic section configuration
       direction=None,                     # Crossing direction (None = both)
       delta_v_tol=1.0,                    # Maximum delta-V for valid connections
       ballistic_tol=1e-8,                 # Tolerance for ballistic connections
       eps2d=1e-3                          # 2D distance tolerance
   )

Finding Connections
-------------------------

Search for connections between manifolds:

.. code-block:: python

   # Create connection pipeline using the factory method
   conn = ConnectionPipeline.with_default_engine(config=config)

   # Solve for connections
   results = conn.solve(manifold_l1, manifold_l2)

   # Display results
   print(f"Found {len(results)} connections")
   print(f"Search completed: {len(results) > 0}")

   # Access connection details
   if results:
       for i, connection in enumerate(results):
           print(f"Connection {i+1}:")
           print(f"  Delta-V: {connection.delta_v}")
           print(f"  Type: {connection.kind}")
           print(f"  Section point: {connection.point2d}")
           print(f"  Source state: {connection.state_u}")
           print(f"  Target state: {connection.state_s}")

Connection Analysis
-------------------------

Analyze found connections:

Connection Properties
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze connection properties
   results_list = list(results)  # Convert to list for indexing
   if results_list:
       connection = results_list[0]  # Take first connection
       
       print(f"Connection analysis:")
       print(f"  Delta-V required: {connection.delta_v:.6f}")
       print(f"  Transfer type: {connection.kind}")
       print(f"  Section point: {connection.point2d}")
       
       # Check if connection is ballistic
       if connection.kind == "ballistic":
           print("  This is a ballistic connection (no delta-V required)")
       else:
           print(f"  Delta-V required: {connection.delta_v:.6f}")

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics.utils.energy import crtbp_energy

   # Analyze energy at connection points
   results_list = list(results)
   if results_list:
       connection = results_list[0]
       
       # Get states at connection point
       state_u = connection.state_u
       state_s = connection.state_s
       
       # Compute energy at both states
       energy_u = crtbp_energy(state_u, mu)
       energy_s = crtbp_energy(state_s, mu)
       energy_difference = abs(energy_s - energy_u)
       
       print(f"Energy at source state: {energy_u:.6f}")
       print(f"Energy at target state: {energy_s:.6f}")
       print(f"Energy difference: {energy_difference:.2e}")
       
       # Plot energy comparison
       import matplotlib.pyplot as plt
       plt.figure(figsize=(10, 6))
       plt.bar(['Source', 'Target'], [energy_u, energy_s], color=['blue', 'red'])
       plt.ylabel('Energy')
       plt.title('Energy Comparison at Connection Point')
       plt.grid(True)
       plt.show()

Custom Connection Detection
---------------------------

HITEN's connection architecture supports custom detection algorithms through several extension points:

Custom Backend Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~

The most powerful way to create custom connection detection is by extending the `_ConnectionsBackend` class:

.. code-block:: python

   from hiten.algorithms.connections.backends.base import _ConnectionsBackend
   from hiten.algorithms.connections.types import _ConnectionResult
   import numpy as np

   class CustomConnectionsBackend(_ConnectionsBackend):
       """Custom connection detection with enhanced filtering."""
       
       def __init__(self, custom_tolerance=1e-5, energy_threshold=1e-6):
           super().__init__()
           self.custom_tolerance = custom_tolerance
           self.energy_threshold = energy_threshold
       
       def solve(self, problem):
           """Custom connection discovery with energy-based filtering."""
           
           # Get section intersections (reuse parent logic)
           sec_u = problem.source.to_section(problem.section, direction=problem.direction)
           sec_s = problem.target.to_section(problem.section, direction=problem.direction)

           pu = np.asarray(sec_u.points, dtype=float)
           ps = np.asarray(sec_s.points, dtype=float)
           Xu = np.asarray(sec_u.states, dtype=float)
           Xs = np.asarray(sec_s.states, dtype=float)

           if pu.size == 0 or ps.size == 0:
               return []

           # Use custom tolerance
           eps = self.custom_tolerance
           dv_tol = float(getattr(problem.search, "delta_v_tol", 1e-3)) if problem.search else 1e-3
           bal_tol = float(getattr(problem.search, "ballistic_tol", 1e-8)) if problem.search else 1e-8

           # Find pairs using standard algorithm from backend
           from hiten.algorithms.connections.backends.utils import _radius_pairs_2d, _nearest_neighbor_2d, _refine_pairs_on_section
           
           pairs_arr = _radius_pairs_2d(pu, ps, eps)
           if pairs_arr.size == 0:
               return []

           # Apply custom energy-based filtering
           filtered_pairs = []
           for k in range(pairs_arr.shape[0]):
               i, j = int(pairs_arr[k, 0]), int(pairs_arr[k, 1])
               
               # Check energy compatibility
               if self._energy_compatible(Xu[i], Xs[j]):
                   filtered_pairs.append((i, j))
           
           if not filtered_pairs:
               return []
           
           # Convert to numpy array for processing
           pairs_np = np.asarray(filtered_pairs, dtype=np.int64)
           
           # Apply standard refinement
           nn_u = _nearest_neighbor_2d(pu) if pu.shape[0] >= 2 else np.full(pu.shape[0], -1, dtype=int)
           nn_s = _nearest_neighbor_2d(ps) if ps.shape[0] >= 2 else np.full(ps.shape[0], -1, dtype=int)
           
           rstar, u0, u1, s0, s1, sval, tval, valid = _refine_pairs_on_section(pu, ps, pairs_np, nn_u, nn_s)

           # Create results with custom processing
           results = []
           for k in range(pairs_np.shape[0]):
               i, j = int(pairs_np[k, 0]), int(pairs_np[k, 1])
               
               if valid[k] and (u0[k] != u1[k]) and (s0[k] != s1[k]):
                   # Interpolated states
                   Xu_seg = (1.0 - sval[k]) * Xu[u0[k]] + sval[k] * Xu[u1[k]]
                   Xs_seg = (1.0 - tval[k]) * Xs[s0[k]] + tval[k] * Xs[s1[k]]
               else:
                   # Direct states
                   Xu_seg = Xu[i]
                   Xs_seg = Xs[j]
               
               # Compute delta-V
               vu = Xu_seg[3:6]
               vs = Xs_seg[3:6]
               dv = float(np.linalg.norm(vu - vs))
               
               if dv <= dv_tol:
                   kind = "ballistic" if dv <= bal_tol else "impulsive"
                   pt = (float(rstar[k, 0]), float(rstar[k, 1])) if valid[k] else (float(pu[i, 0]), float(pu[i, 1]))
                   
                   # Apply custom result processing
                   result = self._process_connection_result(
                       kind, dv, pt, Xu_seg, Xs_seg, i, j
                   )
                   if result is not None:
                       results.append(result)
           
           results.sort(key=lambda r: r.delta_v)
           return results
       
       def _energy_compatible(self, state1, state2):
           """Check if two states are energy-compatible for connection."""
           # Custom energy compatibility check
           energy1 = self._compute_energy(state1)
           energy2 = self._compute_energy(state2)
           return abs(energy1 - energy2) < self.energy_threshold
       
       def _compute_energy(self, state):
           """Compute energy of a state."""
           # Simplified energy computation
           x, y, z, vx, vy, vz = state
           return 0.5 * (vx*vx + vy*vy + vz*vz) - (x*x + y*y + z*z)
       
       def _process_connection_result(self, kind, delta_v, point2d, state_u, state_s, index_u, index_s):
           """Process connection result with custom logic."""
           # Add custom processing here
           return _ConnectionResult(
               kind=kind,
               delta_v=delta_v,
               point2d=point2d,
               state_u=state_u.copy(),
               state_s=state_s.copy(),
               index_u=index_u,
               index_s=index_s
           )

Custom Connection Engine
~~~~~~~~~~~~~~~~~~~~~~~~

Create custom engines that use different backends:

.. code-block:: python

   from hiten.algorithms.connections.engine import _ConnectionEngine
   from hiten.algorithms.connections.types import _ConnectionProblem
   from hiten.algorithms.connections.interfaces import _ManifoldInterface

   class CustomConnectionEngine(_ConnectionEngine):
       """Custom connection engine with specialized backend."""
       
       def __init__(self, backend=None):
           self.backend = backend or CustomConnectionsBackend()
       
       def solve(self, problem: _ConnectionProblem):
           """Solve using custom backend."""
           return self.backend.solve(problem)

   # Use custom engine with proper configuration
   from hiten.algorithms.connections.config import _ConnectionConfig
   from hiten.algorithms.connections.interfaces import _ManifoldInterface
   
   custom_engine = CustomConnectionEngine(CustomConnectionsBackend())
   interface = _ManifoldInterface()
   
   config = _ConnectionConfig(
       section=section_cfg,
       direction=None,
       delta_v_tol=1.0,
       ballistic_tol=1e-8,
       eps2d=1e-3
   )
   
   problem = interface.create_problem(domain_obj=(manifold_l1, manifold_l2), config=config)
   custom_results = custom_engine.solve(problem)

Custom Connection Class
~~~~~~~~~~~~~~~~~~~~~~~

Extend the high-level Connection class to use custom engines:

.. code-block:: python

   from hiten.algorithms.connections.base import ConnectionPipeline
   from hiten.algorithms.connections.config import _ConnectionConfig
   from hiten.system.manifold import Manifold

   class CustomConnectionPipeline(ConnectionPipeline):
       """Custom connection pipeline with specialized engine."""
       
       def __init__(self, config, interface, engine=None):
           custom_engine = engine or CustomConnectionEngine(CustomConnectionsBackend())
           super().__init__(config, interface, custom_engine)

   # Use custom connection pipeline
   config = _ConnectionConfig(
       section=section_cfg,
       direction=None,
       delta_v_tol=1.0,
       ballistic_tol=1e-8,
       eps2d=1e-3
   )
   
   from hiten.algorithms.connections.interfaces import _ManifoldInterface
   custom_conn = CustomConnectionPipeline(
       config=config,
       interface=_ManifoldInterface(),
       engine=CustomConnectionEngine(CustomConnectionsBackend())
   )
   custom_results = custom_conn.solve(manifold_l1, manifold_l2)

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
       for traj in manifold1.trajectories[:10]:  # Sample
           ax1.plot(traj.states[:, 0], traj.states[:, 1], traj.states[:, 2], 'b-', alpha=0.3)
       
       for traj in manifold2.trajectories[:10]:  # Sample
           ax1.plot(traj.states[:, 0], traj.states[:, 1], traj.states[:, 2], 'r-', alpha=0.3)
       
       # Plot connection points
       results_list = list(conn.results)
       if results_list:
           for connection in results_list:
               state_u = connection.state_u
               state_s = connection.state_s
               ax1.scatter(state_u[0], state_u[1], state_u[2], c='g', s=50, marker='o')
               ax1.scatter(state_s[0], state_s[1], state_s[2], c='g', s=50, marker='s')
       
       ax1.set_xlabel('X')
       ax1.set_ylabel('Y')
       ax1.set_zlabel('Z')
       ax1.set_title('3D Connection Visualization')
       
       # Poincare section plot
       ax2 = fig.add_subplot(132)
       
       # Plot section points using the built-in plot method
       conn.plot(ax=ax2)
       
       # Connection properties
       ax3 = fig.add_subplot(133)
       
       results_list = list(conn.results)
       if results_list:
           delta_vs = [c.delta_v for c in results_list]
           kinds = [c.kind for c in results_list]
           
           # Color by connection type
           colors = ['blue' if k == 'ballistic' else 'red' for k in kinds]
           ax3.scatter(range(len(delta_vs)), delta_vs, c=colors, s=50)
           ax3.set_xlabel('Connection Index')
           ax3.set_ylabel('Delta-V')
           ax3.set_title('Connection Properties')
           ax3.grid(True)
       
       plt.tight_layout()
       plt.show()

   # Plot connections
   plot_connections(conn, manifold_l1, manifold_l2)

Advanced Connection Architecture
--------------------------------

HITEN's connection discovery framework is built on a modular architecture that separates algorithmic components from domain-specific logic.

Connection Framework Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The connection framework consists of several key components:

**Base Connection Class** 

    - `ConnectionPipeline`: High-level user-facing facade that provides convenient methods for connection discovery, result visualization, and problem specification.

**Connection Engine** 

    - `_ConnectionEngine`: Main orchestration engine that coordinates the connection discovery process between manifolds and delegates computational work to backend algorithms.

**Backend Algorithms** 

    - `_ConnectionsBackend`: Computational backend that implements the core algorithms for geometric matching, refinement, and Delta-V computation between synodic sections.

**Manifold Interfaces** 

    - `_ManifoldInterface`: Interface adapter that provides clean access to manifold data and handles conversion to synodic section intersections for connection analysis.

**Result Classes** 

    - `_ConnectionResult`: Individual connection result data structure that stores transfer information including Delta-V, connection points, and full state vectors.
    - `ConnectionResults`: Collection class providing convenient access and formatting for multiple connection results.
    - `_ConnectionProblem`: Problem specification that encapsulates all parameters needed for connection discovery between manifolds.

Next Steps
----------

Once you understand connection analysis, you can:

- Learn about advanced integration techniques (see :doc:`guide_10_integrators`)
- Explore correction methods (see :doc:`guide_11_correction`)
- Study continuation algorithms (see :doc:`guide_12_continuation`)

For more advanced connection techniques, see the HITEN source code in :mod:`hiten.algorithms.connections`.
