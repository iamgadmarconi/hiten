Continuation Methods and Custom Steppers
=================================================

This guide covers HITEN's continuation algorithms for generating families of periodic orbits, including natural parameter continuation, pseudo-arclength methods, and how to create custom stepping strategies.

Understanding Continuation
--------------------------------

Continuation methods generate families of solutions by starting from a known solution and following a solution branch as parameters change. This is essential for exploring the parameter space of periodic orbits.

Basic Continuation Concept
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import System, OrbitFamily
   from hiten.algorithms import StateParameter
   from hiten.algorithms.utils.types import SynodicState
   import numpy as np

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   # Start with a single halo orbit
   initial_orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   initial_orbit.correct()

   # Use continuation to generate a family
   state_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.3),         # Target range
       step=0.01,                 # Step size
       max_orbits=10
   )

   family = state_engine.run()
   family.propagate()

   print(f"Generated family with {len(family)} orbits")

Available Continuation Methods
------------------------------------

HITEN provides several continuation strategies optimized for different applications.

Natural Parameter Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest continuation method varies a single parameter linearly:

.. code-block:: python

   # Natural parameter continuation
   natural_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=SynodicState.X,      # Vary X position
       target=(0.8, 0.9),         # Target range
       step=0.01,                 # Step size
       max_orbits=20
   )

   natural_family = natural_engine.run()

Pseudo-Arclength Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More robust continuation that follows the solution curve in parameter space using the `_SecantArcLength` algorithm:

.. code-block:: python

   from hiten.algorithms.continuation.strategies._algorithms import _SecantArcLength
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   
   # Create pseudo-arclength continuation class
   class PseudoArcLengthContinuation(_PeriodicOrbitContinuationInterface, _SecantArcLength):
       """Pseudo-arclength continuation for periodic orbits."""
       
       def _representation(self, orbit):
           """Convert orbit to numerical representation for arclength computation."""
           return orbit.initial_state
   
   # Use pseudo-arclength continuation
   arclength_engine = PseudoArcLengthContinuation(
       initial_orbit=initial_orbit,
       parameter_getter=lambda orbit: np.array([orbit.initial_state[2]]),  # Z component
       target=(0.1, 0.5),
       step=0.01,
       max_orbits=15,
       corrector_kwargs={
           'max_attempts': 50,    # More correction attempts
           'tol': 1e-12           # Higher precision
       }
   )

   arclength_family = arclength_engine.run()

Continuation Parameters
-----------------------------

Control continuation behavior through various parameters:

Step Size Control
~~~~~~~~~~~~~~~~~

The continuation engine automatically adapts step sizes based on correction success/failure:

.. code-block:: python

   # Step size is automatically adapted by the engine
   adaptive_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.5),         # Target range
       step=0.05,                 # Initial step size
       max_orbits=20,
       corrector_kwargs={
           'max_attempts': 25,   # More attempts for better convergence
           'tol': 1e-10          # Higher precision
       }
   )

Convergence Control
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy continuation
   high_precision_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=SynodicState.Z,      # Vary Z component
       target=(0.1, 0.5),         # Target range
       step=0.05,
       max_orbits=20,
       corrector_kwargs={
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
   multi_param_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=[SynodicState.X, SynodicState.Z],  # Vary both X and Z
       target=[[0.8, 0.9], [0.1, 0.3]],        # Target ranges for each parameter
       step=[0.01, 0.01],                      # Step sizes for each parameter
       max_orbits=25
   )

   multi_family = multi_param_engine.run()

Creating Custom Continuation Algorithms
-----------------------------------------

HITEN's modular design allows you to create custom continuation algorithms by combining interfaces and strategies.

.. note::
   When creating custom continuation classes, you must:
   
   1. Inherit from both a domain interface (e.g., `_PeriodicOrbitContinuationInterface`) and an algorithm strategy (e.g., `_NaturalParameter`)
   2. Provide a `parameter_getter` function that extracts continuation parameters from solution objects
   3. Implement the `_make_stepper()` method to return your custom stepping strategy
   4. Pass required parameters (`initial_orbit`, `parameter_getter`, `target`, `step`) to the parent constructor

Basic Custom Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.continuation.strategies._algorithms import _NaturalParameter
   from hiten.algorithms.continuation.strategies._stepping import _NaturalParameterStep
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   import numpy as np

   # Define custom prediction function
   def custom_predictor(orbit, step):
       """Custom prediction with specialized logic."""
       new_state = orbit.initial_state.copy()
       # Apply custom prediction logic
       new_state[2] += step[0]  # Vary Z component
       return new_state

   class CustomContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
       """Custom continuation with specialized prediction logic."""
       
       def __init__(self, initial_orbit, parameter_getter, target, step=1e-4, **kwargs):
           # Store custom predictor for use in _make_stepper
           self._custom_predictor = custom_predictor
           
           super().__init__(
               initial_orbit=initial_orbit,
               parameter_getter=parameter_getter,
               target=target,
               step=step,
               **kwargs
           )
       
       def _make_stepper(self):
           """Create custom stepping strategy."""
           return _NaturalParameterStep(self._custom_predictor)
       
       def _stop_condition(self) -> bool:
           """Check if continuation should terminate."""
           current = self._parameter(self._family[-1])
           return np.any(current < self._target_min) or np.any(current > self._target_max)

   # Use custom continuation
   custom_engine = CustomContinuation(
       initial_orbit=initial_orbit,
       parameter_getter=lambda orbit: np.array([orbit.initial_state[2]]),  # Z component
       target=(0.1, 0.5),
       step=0.05,
       max_orbits=20
   )

Advanced Custom Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, implement custom stepping strategies with event hooks:

.. code-block:: python

   from hiten.algorithms.continuation.strategies._step_interface import _ContinuationStep
   from hiten.algorithms.continuation.strategies._algorithms import _NaturalParameter
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   import numpy as np

   class AdaptiveStepper:
       """Adaptive stepping strategy implementing _ContinuationStep protocol."""
       
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

   # Define adaptive predictor function
   def adaptive_predictor(orbit, step):
       """Predictor function for adaptive stepping."""
       new_state = orbit.initial_state.copy()
       new_state[0] += step[0]  # Vary X component
       return new_state

   # Create custom continuation algorithm using the adaptive stepper
   class AdaptiveContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
       """Custom continuation with adaptive stepping."""
       
       def __init__(self, initial_orbit, parameter_getter, target, step=1e-4, **kwargs):
           # Store adaptive stepper for use in _make_stepper
           self._adaptive_stepper = AdaptiveStepper(adaptive_predictor)
           super().__init__(
               initial_orbit=initial_orbit,
               parameter_getter=parameter_getter,
               target=target,
               step=step,
               **kwargs
           )
       
       def _make_stepper(self):
           """Create adaptive stepping strategy."""
           return self._adaptive_stepper
       
       def _stop_condition(self) -> bool:
           """Check if continuation should terminate."""
           current = self._parameter(self._family[-1])
           return np.any(current < self._target_min) or np.any(current > self._target_max)

   # Use adaptive continuation
   adaptive_engine = AdaptiveContinuation(
       initial_orbit=initial_orbit,
       parameter_getter=lambda orbit: np.array([orbit.initial_state[0]]),  # X component
       target=(0.8, 0.9),
       step=0.01,
       max_orbits=20
   )

Advanced Continuation
---------------------

HITEN's continuation framework is built on a modular architecture that separates algorithmic components from domain-specific logic.

Continuation Engine Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuation framework consists of several key components:

**Base Engine** 

    - `_ContinuationEngine`: The concrete base class that implements the core predict-correct algorithm with step size adaptation and termination criteria.

**Domain Interfaces** 

    - `_PeriodicOrbitContinuationInterface`: Mix-in classes that provide domain-specific implementations for instantiation, correction, and parameter extraction.

**Algorithm Strategies** 

    - `_NaturalParameter`: Abstract class that implements natural parameter continuation.
    - `_SecantArcLength`: Abstract class that implements pseudo-arclength continuation.

**Stepping Strategies**

    - `_NaturalParameterStep`: Concrete implementation that handles the prediction phase of natural parameter continuation.
    - `_SecantStep`: Concrete implementation that handles the prediction phase of pseudo-arclength continuation.

.. code-block:: python

   from hiten.algorithms.continuation.base import _ContinuationEngine
   from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
   from hiten.algorithms.continuation.strategies._algorithms import _NaturalParameter
   from hiten.algorithms.continuation.strategies._stepping import _NaturalParameterStep

   # Example: Understanding the component relationships
   class MyContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
       """Custom continuation combining interface and algorithm."""
       
       def __init__(self, initial_orbit, parameter_getter, target, step=1e-4, **kwargs):
           # Store predictor for use in _make_stepper
           def predictor(orbit, step):
               new_state = orbit.initial_state.copy()
               new_state[2] += step[0]  # Vary Z component
               return new_state
           
           self._predictor = predictor
           super().__init__(
               initial_orbit=initial_orbit,
               parameter_getter=parameter_getter,
               target=target,
               step=step,
               **kwargs
           )
       
       def _make_stepper(self):
           """Create stepping strategy."""
           return _NaturalParameterStep(self._predictor)

Event Hooks and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

Advanced users can implement custom event handling:

.. code-block:: python

   class MonitoringContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
       """Continuation with detailed monitoring and logging."""
       
       def __init__(self, initial_orbit, parameter_getter, target, step=1e-4, **kwargs):
           self.convergence_data = []
           self.step_history = []
           
           # Create stepping strategy with monitoring
           def predictor(orbit, step):
               self.step_history.append(step.copy())
               new_state = orbit.initial_state.copy()
               new_state[2] += step[0]
               return new_state
           
           self._predictor = predictor
           super().__init__(
               initial_orbit=initial_orbit,
               parameter_getter=parameter_getter,
               target=target,
               step=step,
               **kwargs
           )
       
       def _make_stepper(self):
           """Create stepping strategy with monitoring."""
           return _NaturalParameterStep(self._predictor)
       
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
