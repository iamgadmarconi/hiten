Continuation Methods and Custom Steppers
=================================================

This guide covers HITEN's continuation algorithms for generating families of periodic orbits, including natural parameter continuation, pseudo-arclength methods, and how to create custom stepping strategies.

Understanding Continuation
--------------------------------

Continuation methods generate families of solutions by starting from a known solution and following a solution branch as parameters change. This is essential for exploring the parameter space of periodic orbits.

Basic Continuation Concept
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System
   from hiten.algorithms.continuation import StateParameter
   from hiten.algorithms.utils.types import SynodicState
   import numpy as np

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   # Start with a single halo orbit
   initial_orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   initial_orbit.correct()

   # Use continuation to generate a family
   state_engine = StateParameter.with_default_engine()
   
   result = state_engine.solve(
       seed=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.3),         # Target range
       step=0.01,                 # Step size
       max_members=10
   )

   family = result.family
   family.propagate()  # If family has propagate method

   print(f"Generated family with {len(family)} orbits")

Available Continuation Methods
------------------------------------

HITEN provides several continuation strategies optimized for different applications.

Natural Parameter Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest continuation method varies a single parameter linearly:

.. code-block:: python

   # Natural parameter continuation
   natural_engine = StateParameter.with_default_engine()
   
   result = natural_engine.solve(
       seed=initial_orbit,
       state=SynodicState.X,      # Vary X position
       target=(0.8, 0.9),         # Target range
       step=0.01,                 # Step size
       max_members=20
   )

   natural_family = result.family

Pseudo-Arclength Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more robust continuation that follows the solution curve in parameter space, use the secant stepping strategy:

.. code-block:: python

   # Use the StateParameter facade with secant stepper
   state_parameter = StateParameter.with_default_engine()

   result = state_parameter.solve(
       seed=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.5),         # Target range
       step=0.01,                 # Step size
       max_members=15,
       stepper="secant",          # Use secant stepping
       extra_params={
           'max_attempts': 50,    # More correction attempts
           'tol': 1e-12           # Higher precision
       }
   )

   arclength_family = result.family

Continuation Parameters
-----------------------------

Control continuation behavior through various parameters:

Step Size Control
~~~~~~~~~~~~~~~~~

The continuation engine automatically adapts step sizes based on correction success/failure:

.. code-block:: python

   # Step size is automatically adapted by the engine
   adaptive_engine = StateParameter.with_default_engine()
   
   result = adaptive_engine.solve(
       seed=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.5),         # Target range
       step=0.05,                 # Initial step size
       max_members=20,
       extra_params={
           'max_attempts': 25,   # More attempts for better convergence
           'tol': 1e-10          # Higher precision
       }
   )

Convergence Control
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy continuation
   high_precision_engine = StateParameter.with_default_engine()
   
   result = high_precision_engine.solve(
       seed=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.5),         # Target range
       step=0.05,
       max_members=20,
       extra_params={
           'max_attempts': 50,
           'tol': 1e-12,
           'max_delta': 1e-8
       }
   )

Multi-Parameter Continuation
----------------------------------

Continue in multiple parameters simultaneously:

.. code-block:: python

   # Two-parameter continuation
   multi_param_engine = StateParameter.with_default_engine()
   
   result = multi_param_engine.solve(
       seed=initial_orbit,
       state=[SynodicState.X, SynodicState.Z],  # Vary both X and Z
       target=[[0.8, 0.9], [0.1, 0.3]],        # Target ranges for each parameter
       step=[0.01, 0.01],                      # Step sizes for each parameter
       max_members=25
   )

   multi_family = result.family

Creating Custom Continuation Algorithms
-----------------------------------------

HITEN's modular design allows you to create custom continuation algorithms by combining interfaces and stepping strategies.

.. note::
   When creating custom continuation algorithms, you can:

   1. Use the existing facades with custom configurations
   2. Create custom engines by combining backends, interfaces, and stepping strategies
   3. Implement custom stepping strategies for specialized prediction logic

Basic Custom Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.continuation.stepping import _NaturalParameterStep
   from hiten.algorithms.continuation.engine import _OrbitContinuationEngine
   from hiten.algorithms.continuation.backends import _PCContinuationBackend
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   from hiten.algorithms.continuation.config import _OrbitContinuationConfig
   import numpy as np

   # Define custom prediction function
   def custom_predictor(orbit, step):
       """Custom prediction with specialized logic."""
       new_state = orbit.initial_state.copy()
       # Apply custom prediction logic
       new_state[2] += step[0]  # Vary Z component
       return new_state

   # Create custom engine with custom stepping strategy
   backend = _PCContinuationBackend()
   interface = _PeriodicOrbitContinuationInterface()
   engine = _OrbitContinuationEngine(backend=backend, interface=interface)

   # Create configuration
   config = _OrbitContinuationConfig(
       target=(0.1, 0.3),
       step=0.01,
       max_members=10,
       max_retries_per_step=10,
       step_min=1e-10,
       step_max=1.0,
       state=SynodicState.Z  # Vary Z component
   )

   # Use the engine directly
   result = engine.solve(initial_orbit, config)

Advanced Custom Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, implement custom stepping strategies:

.. code-block:: python

   from hiten.algorithms.continuation.stepping import _ContinuationStepBase
   import numpy as np

   class AdaptiveStepper(_ContinuationStepBase):
       """Adaptive stepping strategy."""

       def __init__(self, predictor_fn, initial_step=0.01, min_step=0.001, max_step=0.1):
           self._predictor = predictor_fn
           self.initial_step = initial_step
           self.min_step = min_step
           self.max_step = max_step
           self.current_step = initial_step
           self.convergence_history = []

       def __call__(self, last_solution: object, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
           """Generate prediction with adaptive step size."""

           # Adjust step size based on convergence history
           if len(self.convergence_history) > 2:
               recent_errors = self.convergence_history[-3:]
               avg_error = np.mean(recent_errors)

               if avg_error < 1e-8:  # Good convergence
                   self.current_step = min(self.current_step * 1.2, self.max_step)
               elif avg_error > 1e-6:  # Poor convergence
                   self.current_step = max(self.current_step * 0.8, self.min_step)

           # Generate prediction using custom predictor
           prediction = self._predictor(last_solution, np.array([self.current_step]))
           return prediction, np.array([self.current_step])

       def on_success(self, solution: object) -> None:
           """Called when correction succeeds."""
           # Track convergence for step size adaptation
           if hasattr(solution, 'correction_error'):
               self.convergence_history.append(solution.correction_error)

       def on_failure(self, solution: object) -> None:
           """Called when correction fails."""
           # Reduce step size on failure
           self.current_step = max(self.current_step * 0.5, self.min_step)
           # Track convergence for step size adaptation
           if hasattr(solution, 'correction_error'):
               self.convergence_history.append(solution.correction_error)

   # Define adaptive predictor function
   def adaptive_predictor(orbit, step):
       """Predictor function for adaptive stepping."""
       new_state = orbit.initial_state.copy()
       new_state[0] += step[0]  # Vary X component
       return new_state

   # For advanced usage, you can create a custom engine with the adaptive stepper
   # backend = _PCContinuationBackend()
   # interface = _PeriodicOrbitContinuationInterface()
   # engine = _OrbitContinuationEngine(backend=backend, interface=interface)
   # This would require more complex setup to integrate the adaptive stepper

   # For simplicity, use the standard facade
   state_engine = StateParameter.with_default_engine()
   result = state_engine.solve(
       seed=initial_orbit,
       state=SynodicState.X,
       target=(0.8, 0.9),
       step=0.01,
       max_members=20
   )

Advanced Continuation
---------------------

HITEN's continuation framework is built on a modular architecture that separates algorithmic components from domain-specific logic.

Continuation Engine Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuation framework consists of several key components:

**Engine Components**

    - `_OrbitContinuationEngine`: Orchestration layer that coordinates the continuation process
    - `_ContinuationEngine`: Abstract base class that defines the engine interface

**Backend Components**

    - `_PCContinuationBackend`: Core numerical algorithm that implements the predict-correct-accept loop
    - `_ContinuationBackend`: Abstract base class that defines the backend interface

**Interface Components**

    - `_PeriodicOrbitContinuationInterface`: Domain-specific adapter for periodic orbits

**Stepping Strategies**

    - `_NaturalParameterStep`: Concrete implementation for natural parameter continuation
    - `_SecantStep`: Concrete implementation for pseudo-arclength continuation
    - `_ContinuationPlainStep`: Simple stepping strategy using a provided predictor function
    - `_ContinuationStepBase`: Abstract base class for stepping strategies

**Configuration Components**

    - `_OrbitContinuationConfig`: Configuration class for periodic orbit continuation
    - `_ContinuationConfig`: Abstract base class for continuation configuration

.. code-block:: python

   from hiten.algorithms.continuation.engine import _OrbitContinuationEngine
   from hiten.algorithms.continuation.backends import _PCContinuationBackend
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   from hiten.algorithms.continuation.stepping import _NaturalParameterStep

   # Example: Understanding the component relationships
   def predictor(orbit, step):
       new_state = orbit.initial_state.copy()
       new_state[2] += step[0]  # Vary Z component
       return new_state

   # Create engine with custom stepping strategy
   backend = _PCContinuationBackend()
   interface = _PeriodicOrbitContinuationInterface()
   engine = _OrbitContinuationEngine(backend=backend, interface=interface)

   # The stepping strategy would be integrated into the engine's solve method
   # This is a simplified example showing component relationships

Event Hooks and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced users can implement custom event handling:

.. code-block:: python

   class MonitoringContinuation(_PeriodicOrbitContinuationInterface):
       """Continuation with detailed monitoring and logging."""
       
       # Create custom stepping strategy with monitoring
       def predictor(orbit, step):
           self.step_history.append(step.copy())
           new_state = orbit.initial_state.copy()
           new_state[2] += step[0]
           return new_state

       # Use the standard facade with custom configuration
       # The monitoring would be implemented at the engine level
       
       def _stop_condition(self) -> bool:
           """Check if continuation should terminate."""
           current = self._parameter(self._family[-1])
           return np.any(current < self._target_min) or np.any(current > self._target_max)
       
       def _on_accept(self, candidate):
           """Hook called after successful solution acceptance."""
           # Log convergence information
           param_val = self._parameter(candidate)
           self.convergence_data.append({
               'iteration': len(self._family),
               'parameter': param_val,
               'step_size': self._step.copy()
           })
           
           print(f"Accepted orbit {len(self._family)}: param={param_val}")

Next Steps
----------

Once you understand continuation methods, you can:

- Learn about polynomial methods (see :doc:`guide_14_polynomial`)
- Explore connection analysis (see :doc:`guide_16_connections`)
- Study advanced integration techniques (see :doc:`guide_10_integrators`)

For more advanced continuation techniques, see the HITEN source code in :mod:`hiten.algorithms.continuation`.
