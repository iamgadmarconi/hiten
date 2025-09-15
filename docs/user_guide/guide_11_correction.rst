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

   from hiten.algorithms.corrector import NewtonOrbitCorrector

   # Create a Newton corrector
   newton_corrector = NewtonOrbitCorrector(
       max_attempts=25,
       tol=1e-10,
       max_delta=1e-6
   )

   # Correct an orbit
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   newton_corrector.correct(halo)
   
   print(f"Correction successful: {halo.period is not None}")
   print(f"Final period: {halo.period}")

Finite Difference Correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For orbits where analytical Jacobians are difficult to compute, finite difference methods can be used:

.. code-block:: python

   # Use finite difference for vertical orbits
   vertical = l1.create_orbit("vertical", initial_state=[0.8, 0, 0, 0, 0.1, 0])
   
   # Correct with finite difference
   vertical.correct(
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
   halo.correct(
       max_attempts=50,
       tol=1e-12,        # Very tight tolerance
       max_delta=1e-8    # Small maximum step size
   )

   # Fast correction
   halo.correct(
       max_attempts=10,
       tol=1e-6,         # Looser tolerance
       max_delta=1e-3    # Larger step size
   )

Step Size Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Conservative correction (smaller steps)
   halo.correct(
       max_attempts=30,
       max_delta=1e-8,
       line_search=True  # Use line search for better convergence
   )

   # Aggressive correction (larger steps)
   halo.correct(
       max_attempts=20,
       max_delta=1e-4,
       line_search=False
   )

Correction Strategies by Orbit Type
------------------------------------------

Different orbit types require different correction strategies:

Halo Orbits
~~~~~~~~~~~

.. code-block:: python

   # Halo orbits typically converge well with Newton's method
   halo = l1.create_orbit("halo", amplitude_z=0.3, zenith="northern")
   halo.correct(
       max_attempts=25,
       tol=1e-10,
       max_delta=1e-6
   )

Lyapunov Orbits
~~~~~~~~~~~~~~~

.. code-block:: python

   # Lyapunov orbits are usually well-behaved
   lyapunov = l1.create_orbit("lyapunov", amplitude_x=0.05)
   lyapunov.correct(
       max_attempts=20,
       tol=1e-10
   )

Vertical Orbits
~~~~~~~~~~~~~~~

.. code-block:: python

   # Vertical orbits often need finite difference methods
   vertical = l1.create_orbit("vertical", initial_state=[0.8, 0, 0, 0, 0.1, 0])
   vertical.correct(
       max_attempts=100,
       finite_difference=True,
       tol=1e-10,
       max_delta=1e-8
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

   class SimpleFixedPointCorrector(_Corrector):
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
                   return True
               
               # Apply correction with relaxation
               new_state = current_state + self.relaxation * error
               orbit.initial_state = new_state
           
           return False

   # Use the custom corrector
   custom_corrector = SimpleFixedPointCorrector(relaxation=0.3)
   halo = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   
   success = custom_corrector.correct(halo)
   print(f"Custom correction successful: {success}")

Advanced Custom Corrector
~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated methods, you can implement quasi-Newton or other advanced algorithms:

.. code-block:: python

   class QuasiNewtonCorrector(_Corrector):
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
                   return True
               
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
           
           return False

Next Steps
----------

Once you understand correction methods, you can:

- Learn about continuation algorithms (see :doc:`guide_12_continuation`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study connection analysis (see :doc:`guide_16_connections`)

For more advanced correction techniques, see the HITEN source code in :mod:`hiten.algorithms.corrector`.
