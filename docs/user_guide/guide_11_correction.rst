Orbit Correction and Custom Correctors
======================================

This guide covers HITEN's orbit correction methods, including Newton-based correctors, finite difference methods, and how to create custom correctors for specialized applications.

Understanding Orbit Correction
------------------------------------

Orbit correction is the process of refining initial guesses for periodic orbits to make them truly periodic. This is essential because analytical approximations are rarely exact.

Why Correction is Needed
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   import numpy as np

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   # Create a halo orbit (initial guess)
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   
   # Check if it's already periodic
   print(f"Initial period: {halo.period}")  # Will be None
   print(f"Initial state: {halo.initial_state}")

   # The orbit needs correction to be truly periodic
   halo.correct()
   print(f"Corrected period: {halo.period}")  # Now has a value
   print(f"Corrected state: {halo.initial_state}")

Available Correction Methods
----------------------------------

HITEN provides several correction algorithms optimized for different orbit types.

Newton-Based Correction
~~~~~~~~~~~~~~~~~~~~~~~

The most common correction method uses Newton's method with analytical or numerical Jacobians:

.. code-block:: python

   from hiten.algorithms.corrector import _NewtonOrbitCorrector

   # Create a Newton corrector
   newton_corrector = _NewtonOrbitCorrector(
       max_attempts=25,
       tol=1e-10,
       max_delta=1e-6
   )

   # Correct an orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   corrected_state, half_period = newton_corrector.correct(halo)
   
   print(f"Correction successful: {halo.period is not None}")
   print(f"Final period: {halo.period}")
   print(f"Half period: {half_period}")

Finite Difference Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For orbits where analytical Jacobians are difficult to compute, finite difference methods can be used:

.. code-block:: python

   # Use finite difference for vertical orbits
   vertical = l1.create_orbit("vertical", initial_state=[0.8, 0, 0, 0, 0.1, 0])
   
   # Correct with finite difference using the orbit's correct method
   corrected_state, half_period = vertical.correct(
       max_attempts=100,
       finite_difference=True,
       tol=1e-10
   )

Correction Parameters
---------------------------

Control correction behavior through various parameters:

Convergence Criteria
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy correction
   corrected_state, half_period = halo.correct(
       max_attempts=50,
       tol=1e-12,        # Very tight tolerance
       max_delta=1e-8    # Small maximum step size
   )

   # Fast correction
   corrected_state, half_period = halo.correct(
       max_attempts=10,
       tol=1e-6,         # Looser tolerance
       max_delta=1e-3    # Larger step size
   )

Step Size Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Conservative correction (smaller steps)
   corrected_state, half_period = halo.correct(
       max_attempts=30,
       max_delta=1e-8,
       line_search_config=True  # Use line search for better convergence
   )

   # Aggressive correction (larger steps)
   corrected_state, half_period = halo.correct(
       max_attempts=20,
       max_delta=1e-4,
       line_search_config=False
   )

Line Search Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced control over the line search behavior, you can use the `_LineSearchConfig` class:

.. code-block:: python

   from hiten.algorithms.corrector.line import _LineSearchConfig

   # Custom line search configuration
   line_search_config = _LineSearchConfig(
       armijo_c=1e-4,        # Armijo parameter for sufficient decrease
       alpha_reduction=0.5,  # Step size reduction factor
       min_alpha=1e-4,       # Minimum step size
       max_delta=1e-3        # Maximum step size
   )

   # Use custom line search configuration
   corrected_state, half_period = halo.correct(
       max_attempts=30,
       line_search_config=line_search_config
   )

Advanced Correction
-------------------

HITEN's correction system is built on a modular architecture that separates algorithmic components from domain-specific logic. This design enables flexible combinations of different correction strategies with various problem types.

Correction Interfaces
~~~~~~~~~~~~~~~~~~~~~

The correction framework uses several key interfaces:

**Base Corrector Interface** (`_Corrector`)
    The abstract base class that defines the core correction algorithm interface. All correctors must implement the `correct` method.

**Domain-Specific Interfaces**
    - `_PeriodicOrbitCorrectorInterface`: Handles orbit-specific correction logic
    - `_InvariantToriCorrectorInterface`: Reserved for future tori correction

**Step Control Interfaces**
    - `_StepInterface`: Abstract base for step-size control strategies
    - `_PlainStepInterface`: Simple Newton steps with safeguards
    - `_ArmijoStepInterface`: Armijo line search with backtracking

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector._step_interface import _ArmijoStepInterface
   from hiten.algorithms.corrector.newton import _NewtonCore

   # Create a custom corrector by combining interfaces
   class CustomOrbitCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
       """Custom corrector combining orbit interface with Newton core."""
       
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Add custom initialization logic here
           pass

   # Use the custom corrector
   custom_corrector = CustomOrbitCorrector()
   corrected_state, half_period = custom_corrector.correct(orbit)

Custom Line Search Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized applications, you can implement custom line search strategies by extending the step interface:

.. code-block:: python

   from hiten.algorithms.corrector._step_interface import _StepInterface, _Stepper
   from hiten.algorithms.corrector.line import _LineSearchConfig
   import numpy as np

   class CustomStepInterface(_StepInterface):
       """Custom step interface with specialized line search."""
       
       def __init__(self, custom_param=0.1, **kwargs):
           super().__init__(**kwargs)
           self.custom_param = custom_param
       
       def _build_line_searcher(self, residual_fn, norm_fn, max_delta):
           """Build custom line search stepper."""
           
           def custom_stepper(x, delta, current_norm):
               """Custom line search implementation."""
               
               # Custom step size selection logic
               alpha = self._compute_step_size(x, delta, current_norm)
               
               # Apply step with custom scaling
               x_new = x + alpha * delta
               r_norm_new = norm_fn(residual_fn(x_new))
               
               return x_new, r_norm_new, alpha
           
           return custom_stepper
       
       def _compute_step_size(self, x, delta, current_norm):
           """Custom step size computation."""
           # Implement your custom step size logic here
           base_alpha = 1.0
           
           # Example: Adaptive step size based on residual norm
           if current_norm > 1e-6:
               base_alpha *= 0.5
           
           # Apply custom parameter
           alpha = base_alpha * self.custom_param
           
           return max(alpha, 1e-6)  # Minimum step size

   # Use custom step interface
   class CustomCorrector(_PeriodicOrbitCorrectorInterface, CustomStepInterface):
       pass

   custom_corrector = CustomCorrector(custom_param=0.2)
   corrected_state, half_period = custom_corrector.correct(orbit)

Advanced Line Search Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `_LineSearchConfig` class provides fine-grained control over line search behavior:

.. code-block:: python

   from hiten.algorithms.corrector.line import _LineSearchConfig

   # High-precision line search
   precise_config = _LineSearchConfig(
       armijo_c=1e-4,        # Stricter sufficient decrease condition
       alpha_reduction=0.5,  # Step size reduction factor
       min_alpha=1e-6,       # Very small minimum step size
       max_delta=1e-4        # Conservative maximum step size
   )

   # Fast line search for well-behaved problems
   fast_config = _LineSearchConfig(
       armijo_c=1e-3,        # Looser sufficient decrease condition
       alpha_reduction=0.8,  # Less aggressive step size reduction
       min_alpha=1e-4,       # Larger minimum step size
       max_delta=1e-2        # Larger maximum step size
   )

   # Robust line search for challenging problems
   robust_config = _LineSearchConfig(
       armijo_c=1e-5,        # Very strict sufficient decrease condition
       alpha_reduction=0.3,  # Aggressive step size reduction
       min_alpha=1e-8,       # Very small minimum step size
       max_delta=1e-5        # Very conservative maximum step size
   )

   # Use different configurations for different problems
   corrected_state, half_period = orbit.correct(
       line_search_config=precise_config,
       max_attempts=50
   )

Custom Jacobian Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized problems, you can implement custom Jacobian computation strategies:

.. code-block:: python

   from hiten.algorithms.corrector.base import JacobianFn
   import numpy as np

   def custom_jacobian_fn(x):
       """Custom Jacobian computation with problem-specific optimizations."""
       
       # Example: Sparse Jacobian for structured problems
       n = len(x)
       J = np.zeros((n, n))
       
       # Fill only the non-zero elements based on problem structure
       for i in range(n):
           for j in range(n):
               if abs(i - j) <= 1:  # Tridiagonal structure
                   J[i, j] = compute_jacobian_element(x, i, j)
       
       return J

   def compute_jacobian_element(x, i, j):
       """Compute specific Jacobian element."""
       # Implement your custom Jacobian element computation
       h = 1e-8
       x_plus = x.copy()
       x_minus = x.copy()
       x_plus[j] += h
       x_minus[j] -= h
       
       # Use your custom residual function
       r_plus = your_residual_function(x_plus)
       r_minus = your_residual_function(x_minus)
       
       return (r_plus[i] - r_minus[i]) / (2 * h)

   # Use custom Jacobian in correction
   corrected_state, half_period = orbit.correct(
       jacobian_fn=custom_jacobian_fn
   )

Creating Custom Correctors
--------------------------------

HITEN's modular design allows you to create custom correctors by implementing the correction interface:

Basic Custom Corrector
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   import numpy as np

   class SimpleFixedPointCorrector(_PeriodicOrbitCorrectorInterface):
       """Simple fixed-point iteration corrector."""
       
       def __init__(self, max_attempts=50, tol=1e-8, relaxation=0.5):
           super().__init__()
           self.max_attempts = max_attempts
           self.tol = tol
           self.relaxation = relaxation
       
       def correct(self, orbit, **kwargs):
           """Correct orbit using fixed-point iteration."""
           
           for attempt in range(self.max_attempts):
               # Get current state
               current_state = orbit.initial_state.copy()
               
               # Propagate for one period
               orbit.propagate(steps=1000)
               
               # Compute correction
               final_state = orbit.trajectory[-1]
               error = final_state - current_state
               
               # Check convergence
               if np.linalg.norm(error) < self.tol:
                   orbit.period = orbit.times[-1] - orbit.times[0]
                   return current_state, orbit.times[-1] - orbit.times[0]
               
               # Apply correction with relaxation
               new_state = current_state + self.relaxation * error
               orbit.initial_state = new_state
           
           return current_state, None

   # Use the custom corrector
   custom_corrector = SimpleFixedPointCorrector(relaxation=0.3)
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   
   corrected_state, half_period = custom_corrector.correct(halo)
   print(f"Custom correction successful: {half_period is not None}")
   print(f"Half period: {half_period}")

Advanced Custom Corrector
~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, you can implement quasi-Newton or other advanced algorithms:

.. code-block:: python

   class QuasiNewtonCorrector(_PeriodicOrbitCorrectorInterface):
       """Quasi-Newton corrector using Broyden's method."""
       
       def __init__(self, max_attempts=30, tol=1e-10):
           super().__init__()
           self.max_attempts = max_attempts
           self.tol = tol
           self.jacobian = None
       
       def correct(self, orbit, **kwargs):
           """Correct orbit using quasi-Newton method."""
           
           state_dim = len(orbit.initial_state)
           self.jacobian = np.eye(state_dim)  # Initialize with identity
           
           for attempt in range(self.max_attempts):
               # Store previous state
               prev_state = orbit.initial_state.copy()
               
               # Propagate orbit
               orbit.propagate(steps=1000)
               
               # Compute residual
               final_state = orbit.trajectory[-1]
               residual = final_state - prev_state
               
               # Check convergence
               if np.linalg.norm(residual) < self.tol:
                   orbit.period = orbit.times[-1] - orbit.times[0]
                   return prev_state, orbit.times[-1] - orbit.times[0]
               
               # Update Jacobian using Broyden's method
               if attempt > 0:
                   delta_state = orbit.initial_state - prev_state
                   delta_residual = residual - prev_residual
                   
                   # Broyden update
                   u = delta_residual - self.jacobian @ delta_state
                   v = delta_state
                   
                   if np.dot(v, v) > 1e-12:  # Avoid division by zero
                       self.jacobian += np.outer(u, v) / np.dot(v, v)
               
               # Solve for correction
               try:
                   correction = np.linalg.solve(self.jacobian, -residual)
                   orbit.initial_state = prev_state + correction
               except np.linalg.LinAlgError:
                   # Fall back to simple correction
                   orbit.initial_state = prev_state - 0.1 * residual
               
               prev_residual = residual.copy()
           
           return prev_state, None

Next Steps
----------

Once you understand correction methods, you can:

- Learn about continuation algorithms (see :doc:`guide_12_continuation`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study connection analysis (see :doc:`guide_16_connections`)

For more advanced correction techniques, see the HITEN source code in :mod:`hiten.algorithms.corrector`.
