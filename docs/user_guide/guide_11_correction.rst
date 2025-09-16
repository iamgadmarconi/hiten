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

   from hiten.algorithms.corrector.newton import _NewtonCore
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface

   # Create a Newton corrector by combining interfaces
   class NewtonOrbitCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
       pass
   
   newton_corrector = NewtonOrbitCorrector()

   # Correct an orbit using the orbit's correct method
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   corrected_state, half_period = halo.correct()
   
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

Creating Custom Correctors
--------------------------------

HITEN's modular design allows you to create custom correctors by implementing the correction interface:

Basic Custom Corrector
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.corrector.newton import _NewtonCore
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   import numpy as np

   class SimpleFixedPointCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
       """Simple fixed-point iteration corrector using Newton framework."""
       
       def __init__(self, max_attempts=50, tol=1e-8, relaxation=0.5):
           super().__init__()
           self.max_attempts = max_attempts
           self.tol = tol
           self.relaxation = relaxation
       
       def correct(self, orbit, **kwargs):
           """Correct orbit using fixed-point iteration with Newton framework."""
           
           # Use the orbit's built-in correct method with custom parameters
           return orbit.correct(
               max_attempts=self.max_attempts,
               tol=self.tol,
               **kwargs
           )

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

Advanced Correction
-------------------

HITEN's correction system is built on a modular architecture that separates algorithmic components from domain-specific logic. This design enables flexible combinations of different correction strategies with various problem types.

Correction Interfaces
~~~~~~~~~~~~~~~~~~~~~

The correction framework uses several key interfaces:

**Base Corrector Interface** 
    - `_Corrector`: The abstract base class that defines the core correction algorithm interface. All correctors must implement the `correct` method.

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

Creating Custom Interfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized correction problems, you can create custom interfaces that extend HITEN's correction framework. This allows you to implement domain-specific logic while leveraging the robust numerical algorithms.

**Important**: When extending `_Corrector`, you must implement the abstract `correct` method with the exact signature: `correct(self, x0, residual_fn, *, jacobian_fn=None, norm_fn=None, **kwargs)`. This ensures compatibility with the correction framework.

Custom Domain Interface
~~~~~~~~~~~~~~~~~~~~~~~

Create a custom interface for a specific problem domain:

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector, _BaseCorrectionConfig
   from hiten.algorithms.corrector._step_interface import _Stepper
   from hiten.algorithms.corrector.newton import _NewtonCore
   from dataclasses import dataclass
   from typing import Optional, Tuple
   import numpy as np

   @dataclass(frozen=True, slots=True)
   class _CustomProblemConfig(_BaseCorrectionConfig):
       """Configuration for custom problem correction."""
       
       # Problem-specific parameters
       constraint_type: str = "equality"
       penalty_weight: float = 1.0
       custom_tolerance: float = 1e-8
       
       # Additional constraints
       bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
       linear_constraints: Optional[np.ndarray] = None

   class _CustomProblemInterface(_Corrector):
       """Custom interface for specialized correction problems."""
       
       def __init__(self, problem_config: _CustomProblemConfig, **kwargs):
           super().__init__(**kwargs)
           self.config = problem_config
           self._problem_state = None
           self._constraint_cache = {}
       
       def _setup_problem(self, problem_data):
           """Initialize problem-specific data structures."""
           self._problem_state = {
               'initial_guess': problem_data.get('initial_guess'),
               'constraints': problem_data.get('constraints', []),
               'objective': problem_data.get('objective'),
               'bounds': self.config.bounds
           }
           return self._problem_state
       
       def _compute_residual(self, x: np.ndarray) -> np.ndarray:
           """Compute residual vector for custom problem."""
           # Extract problem components
           state = self._problem_state
           constraints = state['constraints']
           
           residuals = []
           
           # Primary constraint residuals
           for constraint in constraints:
               if constraint['type'] == 'equality':
                   residual = constraint['function'](x) - constraint['target']
               elif constraint['type'] == 'inequality':
                   residual = np.maximum(0, constraint['function'](x) - constraint['upper_bound'])
               else:
                   raise ValueError(f"Unknown constraint type: {constraint['type']}")
               
               residuals.append(residual)
           
           # Add penalty terms for bounds violations
           if self.config.bounds is not None:
               lower, upper = self.config.bounds
               penalty = self._compute_bounds_penalty(x, lower, upper)
               residuals.append(penalty)
           
           return np.concatenate(residuals)
       
       def _compute_bounds_penalty(self, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
           """Compute penalty for bounds violations."""
           violations = np.maximum(0, x - upper) + np.maximum(0, lower - x)
           return self.config.penalty_weight * violations
       
       def _compute_jacobian(self, x: np.ndarray) -> np.ndarray:
           """Compute Jacobian matrix for custom problem."""
           # Use finite differences for complex constraints
           h = 1e-8
           n = len(x)
           m = len(self._compute_residual(x))
           
           J = np.zeros((m, n))
           
           for j in range(n):
               x_plus = x.copy()
               x_minus = x.copy()
               x_plus[j] += h
               x_minus[j] -= h
               
               r_plus = self._compute_residual(x_plus)
               r_minus = self._compute_residual(x_minus)
               
               J[:, j] = (r_plus - r_minus) / (2 * h)
           
           return J
       
       def correct(self, x0: np.ndarray, residual_fn, *, jacobian_fn=None, norm_fn=None, **kwargs):
           """Implement the abstract _Corrector.correct method."""
           # This method must match the _Corrector interface signature
           # For custom problems, we would typically delegate to a Newton core
           # or implement the correction logic here
           
           # Example: Use the provided residual_fn and jacobian_fn
           if norm_fn is None:
               norm_fn = lambda r: np.linalg.norm(r)
           
           # Simple Newton iteration (in practice, use _NewtonCore)
           x = x0.copy()
           max_attempts = kwargs.get('max_attempts', 25)
           tol = kwargs.get('tol', 1e-10)
           
           for attempt in range(max_attempts):
               r = residual_fn(x)
               r_norm = norm_fn(r)
               
               if r_norm < tol:
                   return x, {'iterations': attempt, 'residual_norm': r_norm}
               
               if jacobian_fn is not None:
                   J = jacobian_fn(x)
                   try:
                       delta = np.linalg.solve(J, -r)
                       x = x + delta
                   except np.linalg.LinAlgError:
                       # Fallback to gradient descent
                       delta = -J.T @ r
                       alpha = 0.1
                       x = x + alpha * delta
               else:
                   # Simple gradient descent fallback
                   x = x - 0.1 * r
           
           return x, {'iterations': max_attempts, 'residual_norm': norm_fn(residual_fn(x))}
       
       def solve_custom_problem(self, problem_data, **kwargs):
           """Custom method for solving specialized problems."""
           # Setup problem
           self._setup_problem(problem_data)
           
           # Extract parameters
           x0 = self._problem_state['initial_guess']
           tol = kwargs.get('tol', self.config.custom_tolerance)
           max_attempts = kwargs.get('max_attempts', 25)
           
           # Build residual and Jacobian functions
           def residual_fn(x):
               return self._compute_residual(x)
           
           def jacobian_fn(x):
               return self._compute_jacobian(x)
           
           def norm_fn(r):
               return np.linalg.norm(r)
           
           # Use the correct method
           return self.correct(
               x0=x0,
               residual_fn=residual_fn,
               jacobian_fn=jacobian_fn,
               norm_fn=norm_fn,
               tol=tol,
               max_attempts=max_attempts
           )

   # Create custom corrector by combining interfaces
   class CustomProblemCorrector(_CustomProblemInterface, _NewtonCore):
       """Custom corrector combining domain interface with Newton core."""
       
       def __init__(self, problem_config: _CustomProblemConfig, **kwargs):
           super().__init__(problem_config=problem_config, **kwargs)

   # Usage example
   config = _CustomProblemConfig(
       constraint_type="equality",
       penalty_weight=10.0,
       custom_tolerance=1e-10,
       bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
   )
   
   corrector = CustomProblemCorrector(config)
   
   # Define problem data
   problem_data = {
       'initial_guess': np.array([0.5, 0.5]),
       'constraints': [
           {
               'type': 'equality',
               'function': lambda x: x[0]**2 + x[1]**2,
               'target': 1.0
           }
       ],
       'objective': lambda x: x[0] + x[1]
   }
   
   # Run correction using the custom method
   solution, info = corrector.solve_custom_problem(problem_data)
   print(f"Solution: {solution}")
   print(f"Final residual: {info['residual_norm']}")
   print(f"Iterations: {info['iterations']}")

Custom Step Interface
~~~~~~~~~~~~~~~~~~~~~

Create a custom step control strategy:

.. code-block:: python

   from hiten.algorithms.corrector._step_interface import _StepInterface, _Stepper
   import numpy as np

   class _TrustRegionStepInterface(_StepInterface):
       """Custom step interface implementing trust region method."""
       
       def __init__(self, initial_radius=1.0, max_radius=10.0, **kwargs):
           super().__init__(**kwargs)
           self.initial_radius = initial_radius
           self.max_radius = max_radius
           self.current_radius = initial_radius
           self.radius_history = []
       
       def _build_line_searcher(self, residual_fn, norm_fn, max_delta):
           """Build trust region stepper."""
           
           def trust_region_step(x, delta, current_norm):
               """Trust region step with radius adaptation."""
               
               # Compute full Newton step
               x_full = x + delta
               r_full = residual_fn(x_full)
               r_norm_full = norm_fn(r_full)
               
               # Check if full step is within trust region
               if np.linalg.norm(delta) <= self.current_radius:
                   # Full step is acceptable
                   actual_reduction = current_norm - r_norm_full
                   predicted_reduction = self._predict_reduction(x, delta, residual_fn)
                   
                   # Compute ratio of actual to predicted reduction
                   if predicted_reduction > 0:
                       ratio = actual_reduction / predicted_reduction
                       
                       if ratio > 0.75:  # Good agreement
                           self.current_radius = min(2.0 * self.current_radius, self.max_radius)
                       elif ratio < 0.25:  # Poor agreement
                           self.current_radius = max(0.5 * self.current_radius, 0.1)
                       
                       self.radius_history.append(self.current_radius)
                       return x_full, r_norm_full, 1.0
               
               # Full step outside trust region - compute constrained step
               x_constrained, alpha = self._constrained_step(x, delta, residual_fn, norm_fn)
               r_constrained = residual_fn(x_constrained)
               r_norm_constrained = norm_fn(r_constrained)
               
               return x_constrained, r_norm_constrained, alpha
           
           return trust_region_step
       
       def _predict_reduction(self, x, delta, residual_fn):
           """Predict reduction using quadratic model."""
           # Simplified prediction - in practice, you'd use the quadratic model
           return 0.5 * np.linalg.norm(delta)**2
       
       def _constrained_step(self, x, delta, residual_fn, norm_fn):
           """Compute step constrained by trust region."""
           delta_norm = np.linalg.norm(delta)
           
           if delta_norm <= self.current_radius:
               return x + delta, 1.0
           
           # Scale step to fit within trust region
           alpha = self.current_radius / delta_norm
           return x + alpha * delta, alpha

   # Use custom step interface
   class TrustRegionCorrector(_PeriodicOrbitCorrectorInterface, _TrustRegionStepInterface):
       """Corrector using trust region method."""
       
       def __init__(self, initial_radius=1.0, **kwargs):
           super().__init__(initial_radius=initial_radius, **kwargs)

   # Usage
   trust_corrector = TrustRegionCorrector(initial_radius=0.5)
   corrected_state, half_period = trust_corrector.correct(orbit)

Next Steps
----------

Once you understand correction methods, you can:

- Learn about continuation algorithms (see :doc:`guide_12_continuation`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study connection analysis (see :doc:`guide_16_connections`)

For more advanced correction techniques, see the HITEN source code in :mod:`hiten.algorithms.corrector`.
