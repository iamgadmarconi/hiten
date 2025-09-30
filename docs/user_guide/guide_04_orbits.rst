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

To create a halo orbit, you need to specify the amplitude and zenith of the orbit. The amplitude is the maximum distance from the libration point in the z-direction, and the zenith is the family of the orbit.

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

Internally, HITEN uses Richardson's third-order analytical approximation to generate the initial guess for the halo orbit. This is typically accurate for small amplitudes, but may not be accurate for large amplitudes (above 0.8).

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

To create a Lyapunov orbit, you need to specify the amplitude of the orbit. The amplitude is the maximum distance from the libration point in the x-direction.

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

An initial state is required to create a vertical orbit. This can be computed from the center manifold of the libration point (see :doc:`guide_07_center_manifold`).

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

Custom Correction
~~~~~~~~~~~~~~~~~

You can create a custom corrector by implementing the :class:`~hiten.algorithms.corrector.interfaces._OrbitCorrectionConfig`:

.. code-block:: python

   @dataclass(frozen=True, slots=True)
   class _OrbitCorrectionConfig(_BaseCorrectionConfig):
      """Define a configuration for periodic orbit correction.

      Extends the base correction configuration with orbit-specific parameters
      for constraint selection, integration settings, and event detection.

      Parameters
      ----------
      residual_indices : tuple of int, default=()
         State components used to build the residual vector.
      control_indices : tuple of int, default=()
         State components allowed to change during correction.
      extra_jacobian : callable or None, default=None
         Additional Jacobian contribution function.
      target : tuple of float, default=(0.0,)
         Target values for the residual components.
      event_func : callable, default=:class:`~hiten.algorithms.poincare.singlehit.backend._y_plane_crossing`
         Function to detect Poincare section crossings.
      method : str, default="adaptive"
         Integration method for trajectory computation.
      order : int, default=8
         Integration order for numerical methods.
      steps : int, default=500
         Number of integration steps.
      forward : int, default=1
         Integration direction (1 for forward, -1 for backward).
      """

      residual_indices: tuple[int, ...] = ()  # Components used to build R(x)
      control_indices: tuple[int, ...] = ()   # Components allowed to change
      extra_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
      target: tuple[float, ...] = (0.0,)  # Desired residual values

      event_func: Callable[..., tuple[float, np.ndarray]] = _y_plane_crossing

      method: Literal["fixed", "symplectic", "adaptive"] = "adaptive"
      order: int = 8
      steps: int = 500

      forward: int = 1

This requires you to define the residual indices, control indices, extra Jacobian, target, and the event function.
Then, pass it to a :class:`~hiten.system.orbits.base.GenericOrbit` instance by setting the :attr:`~hiten.system.orbits.base.GenericOrbit.correction_config` property.

Orbit Analysis
--------------

Once corrected, orbits provide various analysis capabilities:

Period and Stability
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic properties
   print(f"Period: {halo.period}")
   print(f"Jacobi constant: {halo.jacobi}")
   
   # Stability analysis
   eigenvalues = halo.eigenvalues
   stability_indices = halo.stability_indices
   print(f"Eigenvalues: {eigenvalues}")
   print(f"Stability indices: {stability_indices}")

Trajectory Access
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get trajectory data
   trajectory = halo.trajectory
   
   print(f"Trajectory shape: {trajectory.states.shape}")
   print(f"Time range: {trajectory.t0} to {trajectory.tf}")
   
   # Extract position components
   x = trajectory.states[:, 0]
   y = trajectory.states[:, 1]
   z = trajectory.states[:, 2]

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics.utils.energy import crtbp_energy
   
   # Compute energy along trajectory
   energies = [crtbp_energy(state, system.mu) for state in trajectory.states]
   
   # Check energy conservation
   initial_energy = energies[0]
   final_energy = energies[-1]
   energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
   
   print(f"Energy error: {energy_error:.2e}")

Continuation Methods
--------------------

Generate families of orbits using continuation:

State Parameter Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms import ContinuationPipeline
   from hiten.algorithms.types.states import SynodicState
   from hiten.algorithms.continuation.config import _OrbitContinuationConfig
   from hiten.system.family import OrbitFamily
   
   # Create initial orbit
   initial_orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   initial_orbit.correct()
   
   # Set up continuation config
   config = _OrbitContinuationConfig(
       target=([0.2], [0.5]),
       step=((0.03,),),
       state=(SynodicState.Z,),
       max_members=10,
       extra_params=dict(max_attempts=50, tol=1e-12),
       stepper="secant",
   )
   
   # Create continuation pipeline
   continuation = ContinuationPipeline.with_default_engine(config=config)
   
   # Run continuation
   result = continuation.generate(initial_orbit)
   
   # Create family from result
   family = OrbitFamily.from_result(result)
   family.propagate()


Examples
--------

Earth-Moon L1 Halo Family
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms import ContinuationPipeline
   from hiten.algorithms.types.states import SynodicState
   from hiten.algorithms.continuation.config import _OrbitContinuationConfig
   from hiten.system.family import OrbitFamily
   
   # Create system
   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)
   
   # Create initial halo orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   halo.correct(max_attempts=25)
   
   # Set up continuation config
   config = _OrbitContinuationConfig(
       target=([0.2], [0.5]),
       step=((0.03,),),
       state=(SynodicState.Z,),
       max_members=10,
       extra_params=dict(max_attempts=50, tol=1e-12),
       stepper="secant",
   )
   
   # Generate family
   continuation = ContinuationPipeline.with_default_engine(config=config)
   result = continuation.generate(halo)
   
   family = OrbitFamily.from_result(result)
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
   
   # Set up continuation config
   config = _OrbitContinuationConfig(
       target=([0.1], [0.3]),
       step=((0.025,),),
       state=(SynodicState.Z,),
       max_members=15,
       extra_params=dict(max_attempts=50, tol=1e-12),
       stepper="secant",
   )
   
   # Generate family
   continuation = ContinuationPipeline.with_default_engine(config=config)
   result = continuation.generate(halo_l2)
   
   family = OrbitFamily.from_result(result)
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

For advanced orbit analysis, see :doc:`guide_11_correction`.
