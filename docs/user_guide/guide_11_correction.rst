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

   from hiten.algorithms.corrector.backends.newton import _NewtonBackend
   from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.stepping import make_plain_stepper

   # Create a Newton corrector using the engine pattern
   backend = _NewtonBackend(stepper_factory=make_plain_stepper())
   interface = _PeriodicOrbitCorrectorInterface()
   newton_corrector = _OrbitCorrectionEngine(backend=backend, interface=interface)
   
   # Correct the orbit using simple API
   halo.correct()
   
   print(f"Correction successful: {halo.period is not None}")
   print(f"Final period: {halo.period}")

Finite Difference Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For orbits where analytical Jacobians are difficult to compute, finite difference methods can be used:

.. code-block:: python

   # Use finite difference for vertical orbits
   vertical = l1.create_orbit("vertical", initial_state=[0.8, 0, 0, 0, 0.1, 0])
   
   # Correct using finite difference
   vertical.correct(max_attempts=100, finite_difference=True, tol=1e-10)

Correction Parameters
---------------------------

Control correction behavior through various parameters:

Convergence Criteria
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy correction
   halo.correct(
       max_attempts=50,
       tol=1e-12,        # Very tight tolerance
       max_delta=1e-8    # Small maximum step size
   )

   # Fast correction configuration
   halo.correct(
       max_attempts=10,
       tol=1e-6,         # Looser tolerance
       max_delta=1e-3    # Larger step size
   )

Step Size Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Conservative correction (smaller steps)
   halo.correct(max_attempts=30, max_delta=1e-8)

   # Aggressive correction (larger steps)
   halo.correct(max_attempts=20, max_delta=1e-4)

Line Search Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced control over the line search behavior, you can use the `_LineSearchConfig` class:

.. code-block:: python

   from hiten.algorithms.corrector.config import _LineSearchConfig

   # Custom line search configuration
   line_search_config = _LineSearchConfig(
       armijo_c=1e-4,        # Armijo parameter for sufficient decrease
       alpha_reduction=0.5,  # Step size reduction factor
       min_alpha=1e-4,       # Minimum step size
       max_delta=1e-3        # Maximum step size
   )

   # Note: Line search configuration is primarily for advanced users
   # creating custom correctors. The simple orbit.correct() API uses
   # sensible defaults. For custom correction engines:
   from hiten.algorithms.corrector.backends.newton import _NewtonBackend
   from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.stepping import make_armijo_stepper
   from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
   
   backend = _NewtonBackend(stepper_factory=make_armijo_stepper(line_search_config))
   interface = _PeriodicOrbitCorrectorInterface()
   corrector = _OrbitCorrectionEngine(backend=backend, interface=interface)
   
   config = _OrbitCorrectionConfig(max_attempts=30)
   problem = interface.create_problem(domain_obj=halo, config=config)
   result = corrector.solve(problem)

Creating Custom Correctors
--------------------------------

HITEN's modular design allows you to create custom correctors by implementing the correction interface:

Basic Custom Corrector
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to create a custom corrector is to use the existing `_NewtonBackend`:

.. code-block:: python

   from hiten.algorithms.corrector.backends.newton import _NewtonBackend
   from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.config import _LineSearchConfig
   from hiten.algorithms.corrector.stepping import make_armijo_stepper

   # Use the ready-to-use corrector with custom configuration
   backend = _NewtonBackend(
       stepper_factory=make_armijo_stepper(_LineSearchConfig(armijo_c=1e-4, alpha_reduction=0.5))
   )
   interface = _PeriodicOrbitCorrectorInterface()
   custom_corrector = _OrbitCorrectionEngine(backend=backend, interface=interface)
   
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   config = halo._correction_config
   problem = interface.create_problem(domain_obj=halo, config=config)
   result = custom_corrector.solve(problem)
   print(f"Custom correction successful: {result is not None}")

For more control, you can create a custom corrector engine with specialized behavior:

.. code-block:: python

   from hiten.algorithms.corrector.backends.newton import _NewtonBackend
   from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.config import _LineSearchConfig, _OrbitCorrectionConfig
   from hiten.algorithms.corrector.stepping import make_armijo_stepper

   class CustomOrbitCorrectionEngine(_OrbitCorrectionEngine):
       """Custom correction engine with specialized configuration."""
       
       def __init__(self, custom_tol=1e-8, **kwargs):
           # Create backend with custom stepper
           backend = _NewtonBackend(stepper_factory=make_armijo_stepper(_LineSearchConfig()))
           interface = _PeriodicOrbitCorrectorInterface()
           super().__init__(backend=backend, interface=interface, **kwargs)
           self.custom_tol = custom_tol
           self.interface = interface
       
       def solve_orbit(self, orbit):
           """Solve with custom tolerance."""
           cfg = _OrbitCorrectionConfig(tol=self.custom_tol)
           problem = self.interface.create_problem(domain_obj=orbit, config=cfg)
           return super().solve(problem)

   # Use the custom corrector
   custom_corrector = CustomOrbitCorrectionEngine(custom_tol=1e-12)
   result = custom_corrector.solve_orbit(halo)

Advanced Custom Corrector for Generic Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For generic correction problems (not orbit-specific), you can create custom correctors
by extending the base correction framework:

.. code-block:: python

   from hiten.algorithms.corrector.base import _Corrector, _BaseCorrectionConfig
   from hiten.algorithms.corrector.newton import _NewtonBackend
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

   # Custom corrector extending the Newton core
   class QuasiNewtonCorrector(_NewtonBackend):
       """Quasi-Newton corrector with custom Jacobian update strategy."""
       
       def __init__(self, config: _QuasiNewtonConfig, **kwargs):
           super().__init__(**kwargs)
           self.config = config
           self.jacobian = None
           self._prev_residual = None
       
       def _initialize_jacobian(self, n: int) -> np.ndarray:
           """Initialize Jacobian matrix."""
           if self.config.initial_jacobian == "identity":
               return np.eye(n)
           else:
               return np.zeros((n, n))
       
       def _update_jacobian(self, delta_x: np.ndarray, delta_r: np.ndarray) -> None:
           """Update Jacobian using Broyden's method."""
           if self.jacobian is None:
               return
           
           if np.dot(delta_x, delta_x) > self.config.update_threshold:
               u = delta_r - self.jacobian @ delta_x
               self.jacobian += np.outer(u, delta_x) / np.dot(delta_x, delta_x)
       
       def _compute_jacobian(self, x, residual_fn, jacobian_fn, fd_step):
           """Override Jacobian computation with quasi-Newton update."""
           if jacobian_fn is not None:
               return jacobian_fn(x)
           
           # Use quasi-Newton update if available
           if self.jacobian is not None:
               return self.jacobian
           
           # Fall back to finite difference for first iteration
           return super()._compute_jacobian(x, residual_fn, jacobian_fn, fd_step)


   # Usage example
   config = _QuasiNewtonConfig(tol=1e-10, max_attempts=30)
   corrector = QuasiNewtonCorrector(config)
   
   # Define residual function for generic correction
   def generic_residual(x):
       # Example: solve x^2 + y^2 = 1, z = 0
       return np.array([x[0]**2 + x[1]**2 - 1.0, x[2]])
   
   # Use the corrector
   x0 = np.array([0.8, 0.6, 0.0])
   solution, info = corrector.correct(x0, generic_residual)
   print(f"Solution: {solution}")
   print(f"Converged in {info['iterations']} iterations")

Advanced Correction
-------------------

HITEN's correction system is built on a modular architecture that separates algorithmic components from domain-specific logic. This design enables flexible combinations of different correction strategies with various problem types.

Correction Interfaces
~~~~~~~~~~~~~~~~~~~~~

The correction framework uses several key interfaces:

**Base Corrector Interface** 
    - `_CorrectorBackend`: The abstract base class that defines the core correction algorithm interface. All correctors must implement the `correct` method.

**Domain-Specific Interfaces**

    - `_PeriodicOrbitCorrectorInterface`: Handles orbit-specific correction logic
    - `_InvariantToriCorrectorInterface`: Reserved for future tori correction

**Step Control Interfaces**

    - `_StepInterface`: Abstract base for step-size control strategies
    - `_CorrectorPlainStep`: Simple Newton steps with safeguards
    - `_ArmijoStep`: Armijo line search with backtracking

.. code-block:: python

   from hiten.algorithms.corrector.backends.base import _CorrectorBackend
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.stepping import _ArmijoStep
   from hiten.algorithms.corrector.newton import _NewtonBackend

   # Create a custom corrector engine combining backend and interface
   backend = _NewtonBackend(stepper_factory=make_plain_stepper())
   interface = _PeriodicOrbitCorrectorInterface()
   
   class CustomOrbitCorrectionEngine(_OrbitCorrectionEngine):
       """Custom correction engine with additional logic.
       
       This allows you to add custom pre/post-processing or validation.
       """
       
       def solve(self, problem):
           """Solve with custom logic."""
           # Add custom pre-processing here
           result = super().solve(problem)
           # Add custom post-processing here
           return result
   
   # Use the custom corrector
   from hiten.algorithms.corrector.stepping import make_plain_stepper
   backend = _NewtonBackend(stepper_factory=make_plain_stepper())
   interface = _PeriodicOrbitCorrectorInterface()
   custom_corrector = CustomOrbitCorrectionEngine(backend=backend, interface=interface)
   
   # Correct an orbit
   config = halo._correction_config
   problem = interface.create_problem(domain_obj=halo, config=config)
   result = custom_corrector.solve(problem)

Custom Line Search Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized applications, you can implement custom line search strategies by extending the step interface:

.. code-block:: python

   from hiten.algorithms.corrector.stepping.base import _CorrectorStepBase
   from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
   from hiten.algorithms.corrector.config import _LineSearchConfig
   import numpy as np

   class CustomStepInterface(_CorrectorStepBase):
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

   # Note: Custom step interfaces are advanced. The example above shows the concept,
   # but integrating custom steppers into the correction engine requires careful
   # consideration of the stepper factory pattern. For most use cases, using
   # make_plain_stepper() or make_armijo_stepper() with custom LineSearchConfig
   # is sufficient and much simpler.

Advanced Line Search Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `_LineSearchConfig` class provides fine-grained control over line search behavior:

.. code-block:: python

   from hiten.algorithms.corrector.config import _LineSearchConfig

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
   from hiten.algorithms.corrector.backends.newton import _NewtonBackend
   from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
   from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
   from hiten.algorithms.corrector.stepping import make_armijo_stepper
   from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
   
   backend = _NewtonBackend(stepper_factory=make_armijo_stepper(precise_config))
   interface = _PeriodicOrbitCorrectorInterface()
   corrector = _OrbitCorrectionEngine(backend=backend, interface=interface)
   
   config = _OrbitCorrectionConfig(max_attempts=50)
   problem = interface.create_problem(domain_obj=halo, config=config)
   result = corrector.solve(problem)

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

   # Note: Custom Jacobian computation requires extending the _NewtonBackend class
   # to override the _compute_jacobian method. The example above shows the concept
   # of custom Jacobian elements, but full integration requires subclassing _NewtonBackend.
   # For most applications, the built-in analytical and finite difference Jacobians
   # are sufficient and well-optimized.

Next Steps
----------

Once you understand correction methods, you can:

- Learn about continuation algorithms (see :doc:`guide_12_continuation`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study connection analysis (see :doc:`guide_16_connections`)

For more advanced correction techniques, see the HITEN source code in :mod:`hiten.algorithms.corrector`.
