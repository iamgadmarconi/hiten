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

The simplest way to create a custom corrector is to use the existing `_NewtonOrbitCorrector`:

.. code-block:: python

   from hiten.algorithms.corrector import _NewtonOrbitCorrector
   from hiten.algorithms.corrector.line import _LineSearchConfig

   # Use the ready-to-use corrector with custom configuration
   custom_corrector = _NewtonOrbitCorrector(
       max_attempts=50,
       tol=1e-8,
       line_search_config=_LineSearchConfig(
           armijo_c=1e-4,
           alpha_reduction=0.5
       )
   )
   
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   corrected_state, half_period = custom_corrector.correct(halo)
   print(f"Custom correction successful: {half_period is not None}")
   print(f"Half period: {half_period}")

For more control, you can create a custom corrector by combining interfaces:

.. code-block:: python

   from hiten.algorithms.corrector.newton import _NewtonCore
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.line import _LineSearchConfig

   class CustomOrbitCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
       """Custom corrector with specialized configuration."""
       
       def __init__(self, custom_tol=1e-8, **kwargs):
           super().__init__(**kwargs)
           self.custom_tol = custom_tol
       
       def correct(self, orbit, **kwargs):
           """Correct orbit with custom tolerance."""
           # Override default tolerance
           kwargs.setdefault('tol', self.custom_tol)
           return super().correct(orbit, **kwargs)

   # Use the custom corrector
   custom_corrector = CustomOrbitCorrector(custom_tol=1e-12)
   corrected_state, half_period = custom_corrector.correct(halo)

Advanced Custom Corrector with Backend Separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, follow HITEN's architectural patterns with backend separation:

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector, _BaseCorrectionConfig
   from hiten.algorithms.corrector._step_interface import _Stepper
   from abc import ABC, abstractmethod
   from dataclasses import dataclass
   from typing import Optional, Tuple
   import numpy as np

   # Define domain-specific exceptions
   class CustomCorrectionError(Exception):
       """Base exception for custom correction problems."""
       pass

   class ConvergenceError(CustomCorrectionError):
       """Raised when correction fails to converge."""
       pass

   # Configuration following HITEN's pattern
   @dataclass(frozen=True, slots=True)
   class _QuasiNewtonConfig(_BaseCorrectionConfig):
       """Configuration for quasi-Newton correction."""
       jacobian_update_method: str = "broyden"
       initial_jacobian: str = "identity"
       update_threshold: float = 1e-12

   # Backend for computation (following HITEN's backend pattern)
   class _QuasiNewtonBackend(ABC):
       """Backend for quasi-Newton correction computations."""
       
       def __init__(self, config: _QuasiNewtonConfig):
           self.config = config
           self.jacobian = None
           self._prev_residual = None
       
       def initialize_jacobian(self, n: int) -> np.ndarray:
           """Initialize Jacobian matrix."""
           if self.config.initial_jacobian == "identity":
               return np.eye(n)
           else:
               return np.zeros((n, n))
       
       def update_jacobian(self, delta_x: np.ndarray, delta_r: np.ndarray) -> None:
           """Update Jacobian using Broyden's method."""
           if self.jacobian is None:
               return
           
           if np.dot(delta_x, delta_x) > self.config.update_threshold:
               u = delta_r - self.jacobian @ delta_x
               self.jacobian += np.outer(u, delta_x) / np.dot(delta_x, delta_x)
       
       def solve_correction(self, residual: np.ndarray) -> np.ndarray:
           """Solve for correction step."""
           try:
               return np.linalg.solve(self.jacobian, -residual)
           except np.linalg.LinAlgError:
               # Fall back to gradient descent
               return -0.1 * residual

   # Main corrector following HITEN's interface pattern
   class QuasiNewtonCorrector(_Corrector):
       """Quasi-Newton corrector using backend separation."""
       
       def __init__(self, config: _QuasiNewtonConfig, **kwargs):
           super().__init__(**kwargs)
           self.config = config
           self._backend = _QuasiNewtonBackend(config)
       
       def correct(self, x0: np.ndarray, residual_fn, *, jacobian_fn=None, norm_fn=None, **kwargs):
           """Implement the abstract _Corrector.correct method."""
           if norm_fn is None:
               norm_fn = lambda r: np.linalg.norm(r)
           
           x = x0.copy()
           n = len(x)
           self._backend.jacobian = self._backend.initialize_jacobian(n)
           
           max_attempts = kwargs.get('max_attempts', self.config.max_attempts)
           tol = kwargs.get('tol', self.config.tol)
           
           for attempt in range(max_attempts):
               r = residual_fn(x)
               r_norm = norm_fn(r)
               
               if r_norm < tol:
                   return x, {'iterations': attempt, 'residual_norm': r_norm}
               
               # Update Jacobian if not first iteration
               if attempt > 0 and self._backend._prev_residual is not None:
                   delta_x = x - prev_x
                   delta_r = r - self._backend._prev_residual
                   self._backend.update_jacobian(delta_x, delta_r)
               
               # Solve for correction
               correction = self._backend.solve_correction(r)
               prev_x = x.copy()
               x = x + correction
               self._backend._prev_residual = r.copy()
           
           raise ConvergenceError(f"Failed to converge after {max_attempts} iterations")

   # Usage example
   config = _QuasiNewtonConfig(tol=1e-10, max_attempts=30)
   corrector = QuasiNewtonCorrector(config)
   
   # Define residual function for orbit correction
   def orbit_residual(x):
       # This would be the actual orbit constraint residual
       return np.array([x[0]**2 + x[1]**2 - 1.0, x[2]])  # Example constraint
   
   # Use the corrector
   x0 = np.array([0.8, 0.6, 0.0])
   solution, info = corrector.correct(x0, orbit_residual)
   print(f"Solution: {solution}")
   print(f"Converged in {info['iterations']} iterations")

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

Custom Domain Interface with Engine Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a custom interface following HITEN's engine pattern for better separation of concerns:

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector, _BaseCorrectionConfig
   from hiten.algorithms.corrector._step_interface import _Stepper
   from hiten.algorithms.corrector.newton import _NewtonCore
   from abc import ABC, abstractmethod
   from dataclasses import dataclass
   from typing import Optional, Tuple, Dict, Any
   import numpy as np

   # Domain-specific exceptions
   class ConstraintError(Exception):
       """Raised when constraint evaluation fails."""
       pass

   class ProblemSetupError(Exception):
       """Raised when problem setup fails."""
       pass

   # Configuration following HITEN's mixin pattern
   @dataclass(frozen=True, slots=True)
   class _ConstraintConfig:
       """Configuration for constraint handling."""
       constraint_type: str = "equality"
       penalty_weight: float = 1.0
       bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

   @dataclass(frozen=True, slots=True)
   class _CustomProblemConfig(_BaseCorrectionConfig, _ConstraintConfig):
       """Complete configuration for custom problem correction."""
       custom_tolerance: float = 1e-8
       linear_constraints: Optional[np.ndarray] = None

   # Backend for constraint computation
   class _ConstraintBackend(ABC):
       """Backend for constraint computation and evaluation."""
       
       def __init__(self, config: _ConstraintConfig):
           self.config = config
           self._constraint_cache: Dict[str, Any] = {}
       
       def evaluate_constraints(self, x: np.ndarray, constraints: list) -> np.ndarray:
           """Evaluate all constraints at point x."""
           residuals = []
           
           for i, constraint in enumerate(constraints):
               try:
                   if constraint['type'] == 'equality':
                       residual = constraint['function'](x) - constraint['target']
                   elif constraint['type'] == 'inequality':
                       residual = np.maximum(0, constraint['function'](x) - constraint['upper_bound'])
                   else:
                       raise ConstraintError(f"Unknown constraint type: {constraint['type']}")
                   
                   residuals.append(residual)
               except Exception as e:
                   raise ConstraintError(f"Failed to evaluate constraint {i}: {e}")
           
           return np.concatenate(residuals) if residuals else np.array([])
       
       def compute_bounds_penalty(self, x: np.ndarray) -> np.ndarray:
           """Compute penalty for bounds violations."""
           if self.config.bounds is None:
               return np.array([])
           
           lower, upper = self.config.bounds
           violations = np.maximum(0, x - upper) + np.maximum(0, lower - x)
           return self.config.penalty_weight * violations

   # Engine for coordinating correction process
   class _CustomCorrectionEngine:
       """Engine coordinating backend and correction strategy."""
       
       def __init__(self, backend: _ConstraintBackend, config: _CustomProblemConfig):
           self._backend = backend
           self._config = config
           self._problem_state: Optional[Dict[str, Any]] = None
       
       def setup_problem(self, problem_data: Dict[str, Any]) -> None:
           """Setup problem-specific data structures."""
           try:
               self._problem_state = {
                   'initial_guess': problem_data.get('initial_guess'),
                   'constraints': problem_data.get('constraints', []),
                   'objective': problem_data.get('objective'),
               }
           except Exception as e:
               raise ProblemSetupError(f"Failed to setup problem: {e}")
       
       def compute_residual(self, x: np.ndarray) -> np.ndarray:
           """Compute complete residual vector."""
           if self._problem_state is None:
               raise ProblemSetupError("Problem not set up. Call setup_problem first.")
           
           # Evaluate constraints
           constraint_residuals = self._backend.evaluate_constraints(
               x, self._problem_state['constraints']
           )
           
           # Add bounds penalty
           bounds_penalty = self._backend.compute_bounds_penalty(x)
           
           return np.concatenate([constraint_residuals, bounds_penalty])
       
       def compute_jacobian(self, x: np.ndarray) -> np.ndarray:
           """Compute Jacobian using finite differences."""
           h = 1e-8
           n = len(x)
           r0 = self.compute_residual(x)
           m = len(r0)
           
           J = np.zeros((m, n))
           for j in range(n):
               x_plus = x.copy()
               x_minus = x.copy()
               h_j = h * max(1.0, abs(x[j]))
               x_plus[j] += h_j
               x_minus[j] -= h_j
               
               r_plus = self.compute_residual(x_plus)
               r_minus = self.compute_residual(x_minus)
               J[:, j] = (r_plus - r_minus) / (2 * h_j)
           
           return J

   # Main corrector interface
   class CustomProblemCorrector(_Corrector):
       """Custom corrector using engine pattern."""
       
       def __init__(self, config: _CustomProblemConfig, **kwargs):
           super().__init__(**kwargs)
           self.config = config
           self._backend = _ConstraintBackend(config)
           self._engine = _CustomCorrectionEngine(self._backend, config)
       
       def correct(self, x0: np.ndarray, residual_fn, *, jacobian_fn=None, norm_fn=None, **kwargs):
           """Implement the abstract _Corrector.correct method."""
           if norm_fn is None:
               norm_fn = lambda r: np.linalg.norm(r)
           
           x = x0.copy()
           max_attempts = kwargs.get('max_attempts', self.config.max_attempts)
           tol = kwargs.get('tol', self.config.custom_tolerance)
           
           for attempt in range(max_attempts):
               r = residual_fn(x)
               r_norm = norm_fn(r)
               
               if r_norm < tol:
                   return x, {'iterations': attempt, 'residual_norm': r_norm}
               
               if jacobian_fn is not None:
                   J = jacobian_fn(x)
               else:
                   J = self._engine.compute_jacobian(x)
               
               try:
                   delta = np.linalg.solve(J, -r)
                   x = x + delta
               except np.linalg.LinAlgError:
                   # Fallback to gradient descent
                   delta = -J.T @ r
                   alpha = 0.1
                   x = x + alpha * delta
           
           raise ConvergenceError(f"Failed to converge after {max_attempts} iterations")
       
       def solve_custom_problem(self, problem_data: Dict[str, Any], **kwargs):
           """High-level method for solving custom problems."""
           # Setup problem
           self._engine.setup_problem(problem_data)
           
           # Extract initial guess
           x0 = self._engine._problem_state['initial_guess']
           
           # Build residual and Jacobian functions
           def residual_fn(x):
               return self._engine.compute_residual(x)
           
           def jacobian_fn(x):
               return self._engine.compute_jacobian(x)
           
           # Use the correct method
           return self.correct(
               x0=x0,
               residual_fn=residual_fn,
               jacobian_fn=jacobian_fn,
               **kwargs
           )

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

Custom Step Interface with Protocol Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a custom step control strategy following HITEN's protocol pattern:

.. code-block:: python

   from hiten.algorithms.corrector._step_interface import _StepInterface, _Stepper
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.newton import _NewtonCore
   from dataclasses import dataclass
   from typing import Protocol, Optional
   import numpy as np

   # Configuration for trust region method
   @dataclass(frozen=True, slots=True)
   class _TrustRegionConfig:
       """Configuration for trust region step control."""
       initial_radius: float = 1.0
       max_radius: float = 10.0
       min_radius: float = 0.1
       good_ratio_threshold: float = 0.75
       poor_ratio_threshold: float = 0.25
       radius_expansion_factor: float = 2.0
       radius_contraction_factor: float = 0.5

   # Protocol for trust region strategies
   class _TrustRegionStrategy(Protocol):
       """Protocol for trust region step strategies."""
       
       def __call__(self, x: np.ndarray, delta: np.ndarray, current_norm: float) -> tuple[np.ndarray, float, float]:
           """Apply trust region step strategy."""
           ...

   # Backend for trust region computations
   class _TrustRegionBackend:
       """Backend for trust region computations."""
       
       def __init__(self, config: _TrustRegionConfig):
           self.config = config
           self.current_radius = config.initial_radius
           self.radius_history = []
       
       def predict_reduction(self, delta: np.ndarray) -> float:
           """Predict reduction using quadratic model."""
           # Simplified prediction - in practice, you'd use the quadratic model
           return 0.5 * np.linalg.norm(delta)**2
       
       def adapt_radius(self, actual_reduction: float, predicted_reduction: float) -> None:
           """Adapt trust region radius based on reduction ratio."""
           if predicted_reduction <= 0:
               return
           
           ratio = actual_reduction / predicted_reduction
           
           if ratio > self.config.good_ratio_threshold:
               # Good agreement - expand radius
               self.current_radius = min(
                   self.config.radius_expansion_factor * self.current_radius,
                   self.config.max_radius
               )
           elif ratio < self.config.poor_ratio_threshold:
               # Poor agreement - contract radius
               self.current_radius = max(
                   self.config.radius_contraction_factor * self.current_radius,
                   self.config.min_radius
               )
           
           self.radius_history.append(self.current_radius)
       
       def constrain_step(self, x: np.ndarray, delta: np.ndarray) -> tuple[np.ndarray, float]:
           """Constrain step to fit within trust region."""
           delta_norm = np.linalg.norm(delta)
           
           if delta_norm <= self.current_radius:
               return x + delta, 1.0
           
           # Scale step to fit within trust region
           alpha = self.current_radius / delta_norm
           return x + alpha * delta, alpha

   # Custom step interface following HITEN's pattern
   class _TrustRegionStepInterface(_StepInterface):
       """Step interface implementing trust region method."""
       
       def __init__(self, config: _TrustRegionConfig, **kwargs):
           super().__init__(**kwargs)
           self.config = config
           self._backend = _TrustRegionBackend(config)
       
       def _build_line_searcher(self, residual_fn, norm_fn, max_delta):
           """Build trust region stepper following HITEN's protocol pattern."""
           
           def trust_region_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
               """Trust region step with radius adaptation."""
               
               # Compute full Newton step
               x_full = x + delta
               r_full = residual_fn(x_full)
               r_norm_full = norm_fn(r_full)
               
               # Check if full step is within trust region
               if np.linalg.norm(delta) <= self._backend.current_radius:
                   # Full step is acceptable
                   actual_reduction = current_norm - r_norm_full
                   predicted_reduction = self._backend.predict_reduction(delta)
                   
                   # Adapt radius based on reduction ratio
                   self._backend.adapt_radius(actual_reduction, predicted_reduction)
                   
                   return x_full, r_norm_full, 1.0
               
               # Full step outside trust region - compute constrained step
               x_constrained, alpha = self._backend.constrain_step(x, delta)
               r_constrained = residual_fn(x_constrained)
               r_norm_constrained = norm_fn(r_constrained)
               
               return x_constrained, r_norm_constrained, alpha
           
           return trust_region_step

   # Complete corrector combining interfaces
   class TrustRegionCorrector(_PeriodicOrbitCorrectorInterface, _TrustRegionStepInterface, _NewtonCore):
       """Corrector using trust region method with full HITEN architecture."""
       
       def __init__(self, config: _TrustRegionConfig, **kwargs):
           super().__init__(config=config, **kwargs)

   # Usage example
   config = _TrustRegionConfig(
       initial_radius=0.5,
       max_radius=5.0,
       good_ratio_threshold=0.8
   )
   
   trust_corrector = TrustRegionCorrector(config)
   corrected_state, half_period = trust_corrector.correct(orbit)
   
   print(f"Trust region radius history: {trust_corrector._backend.radius_history}")
   print(f"Final radius: {trust_corrector._backend.current_radius}")

Architectural Best Practices
-----------------------------

The custom correction examples above follow HITEN's architectural patterns:

**Configuration Pattern**
    - Use `@dataclass(frozen=True, slots=True)` for immutable configurations
    - Follow mixin pattern for configuration inheritance
    - Separate domain-specific configs from base configs

**Backend Separation**
    - Separate computation logic into backend classes
    - Backends handle low-level numerical operations
    - Enable testing and reuse of computation components

**Engine Coordination**
    - Use engine classes to coordinate backends and strategies
    - Engines handle high-level algorithm orchestration
    - Provide clean separation between interface and implementation

**Protocol-Based Design**
    - Use protocols for flexible strategy implementations
    - Enable different algorithms to share common interfaces
    - Support multiple inheritance cleanly

**Error Handling**
    - Define domain-specific exception hierarchies
    - Provide meaningful error messages for debugging
    - Handle edge cases gracefully

**Multiple Inheritance**
    - Combine interfaces through multiple inheritance
    - Follow HITEN's pattern of mixing domain interfaces with algorithm cores
    - Maintain clean separation of concerns

Next Steps
----------

Once you understand correction methods, you can:

- Learn about continuation algorithms (see :doc:`guide_12_continuation`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study connection analysis (see :doc:`guide_16_connections`)

For more advanced correction techniques, see the HITEN source code in :mod:`hiten.algorithms.corrector`.
