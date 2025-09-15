Advanced Continuation Methods and Custom Steppers
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
       state=(SynodicState.Z,),  # Vary Z amplitude
       amplitude=True,
       target=(0.1, 0.5),        # From 0.1 to 0.5
       step=0.05,                # Step size
       max_orbits=10
   )

   state_engine.run()
   family = OrbitFamily.from_engine(state_engine)
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
       state=(SynodicState.X,),  # Vary X position
       amplitude=False,
       target=(0.8, 0.9),        # From X=0.8 to X=0.9
       step=0.01,
       max_orbits=20
   )

   natural_engine.run()
   natural_family = OrbitFamily.from_engine(natural_engine)

Pseudo-Arclength Continuation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More robust continuation that follows the solution curve in parameter space:

.. code-block:: python

   from hiten.algorithms.continuation.strategies import PseudoArclengthStep

   # Pseudo-arclength continuation
   arclength_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.Z,),
       amplitude=True,
       target=(0.1, 0.5),
       step=0.05,
       max_orbits=15,
       stepping_strategy=PseudoArclengthStep()
   )

   arclength_engine.run()
   arclength_family = OrbitFamily.from_engine(arclength_engine)

Continuation Parameters
-----------------------------

Control continuation behavior through various parameters:

Step Size Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Adaptive step size
   adaptive_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.Z,),
       amplitude=True,
       target=(0.1, 0.5),
       step=0.05,
       max_orbits=20,
       adaptive_step=True,       # Enable adaptive stepping
       min_step=0.01,           # Minimum step size
       max_step=0.1             # Maximum step size
   )

Convergence Control
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy continuation
   high_precision_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.Z,),
       amplitude=True,
       target=(0.1, 0.5),
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
       state=(SynodicState.X, SynodicState.Z),  # Vary both X and Z
       amplitude=False,
       target=([0.8, 0.2], [0.9, 0.4]),        # From (0.8, 0.2) to (0.9, 0.4)
       step=(0.01, 0.02),                       # Different steps for each parameter
       max_orbits=25
   )

   multi_param_engine.run()
   multi_family = OrbitFamily.from_engine(multi_param_engine)

Creating Custom Stepping Strategies
-----------------------------------------

HITEN's modular design allows you to create custom stepping strategies:

Basic Custom Stepper
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.continuation.strategies._step_interface import _ContinuationStep
   import numpy as np

   class LinearStepper(_ContinuationStep):
       """Simple linear stepping strategy."""
       
       def __init__(self, step_size=0.01):
           self.step_size = step_size
       
       def __call__(self, last_solution, step):
           """Generate next solution prediction."""
           
           # Simple linear extrapolation
           current_state = last_solution.initial_state.copy()
           new_state = current_state.copy()
           
           # Apply step in the direction of the parameter
           new_state[0] += step[0]  # Assume varying X coordinate
           
           return new_state, step

   # Use custom stepper
   custom_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.X,),
       amplitude=False,
       target=(0.8, 0.9),
       step=0.01,
       max_orbits=20,
       stepping_strategy=LinearStepper(step_size=0.01)
   )

Advanced Custom Stepper
~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, implement adaptive stepping:

.. code-block:: python

   class AdaptiveStepper(_ContinuationStep):
       """Adaptive stepping strategy with step size control."""
       
       def __init__(self, initial_step=0.01, min_step=0.001, max_step=0.1):
           self.initial_step = initial_step
           self.min_step = min_step
           self.max_step = max_step
           self.current_step = initial_step
           self.convergence_history = []
       
       def __call__(self, last_solution, step):
           """Generate prediction with adaptive step size."""
           
           # Adjust step size based on convergence history
           if len(self.convergence_history) > 2:
               recent_errors = self.convergence_history[-3:]
               avg_error = np.mean(recent_errors)
               
               if avg_error < 1e-8:  # Good convergence
                   self.current_step = min(self.current_step * 1.2, self.max_step)
               elif avg_error > 1e-6:  # Poor convergence
                   self.current_step = max(self.current_step * 0.8, self.min_step)
           
           # Generate prediction
           current_state = last_solution.initial_state.copy()
           new_state = current_state.copy()
           new_state[0] += self.current_step
           
           return new_state, np.array([self.current_step])
       
       def on_success(self, solution):
           """Called when correction succeeds."""
           # Track convergence for step size adaptation
           if hasattr(solution, 'correction_error'):
               self.convergence_history.append(solution.correction_error)
       
       def on_failure(self, solution):
           """Called when correction fails."""
           # Reduce step size on failure
           self.current_step = max(self.current_step * 0.5, self.min_step)

   # Use adaptive stepper
   adaptive_engine = StateParameter(
       initial_orbit=initial_orbit,
       state=(SynodicState.X,),
       amplitude=False,
       target=(0.8, 0.9),
       step=0.01,
       max_orbits=20,
       stepping_strategy=AdaptiveStepper()
   )

Next Steps
----------

Once you understand continuation methods, you can:

- Learn about polynomial methods (see :doc:`guide_14_polynomial`)
- Explore connection analysis (see :doc:`guide_16_connections`)
- Study advanced integration techniques (see :doc:`guide_10_integrators`)

For more advanced continuation techniques, see the HITEN source code in :mod:`hiten.algorithms.continuation`.
