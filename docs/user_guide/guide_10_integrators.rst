Integration Methods and Custom Integrators
===================================================

This guide covers HITEN's advanced integration methods, including Runge-Kutta schemes, symplectic integrators, and how to create custom integrators for specialized applications.

Available Integrators
---------------------------

HITEN provides several integrators optimized for different types of dynamical systems. The main user-facing classes are factory classes that create the appropriate integrator instances.

Integration Methods
~~~~~~~~~~~~~~~~~~~

The primary way to use integrators in HITEN is through the :meth:`~hiten.algorithms.dynamics.base._propagate_dynsys` method, which supports the following methods:

- ``"fixed"``: Fixed-step Runge-Kutta methods (orders 4, 6, 8)
- ``"adaptive"``: Adaptive step-size Runge-Kutta methods (orders 5, 8)
- ``"symplectic"``: High-order symplectic integrators (orders 2, 4, 6, 8)

For direct access to integrator classes, use the factory classes from :mod:`~hiten.algorithms.integrators`.

Runge-Kutta Methods
~~~~~~~~~~~~~~~~~~~

HITEN includes several explicit Runge-Kutta schemes with different orders of accuracy. These are accessed through factory classes:

.. code-block:: python

   from hiten.algorithms.integrators.rk import RungeKutta, AdaptiveRK
   from hiten.algorithms.integrators.symplectic import ExtendedSymplectic
   from hiten import System
   import numpy as np

   system = System.from_bodies("earth", "moon")
   initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])

   # Fixed-step Runge-Kutta methods
   rk4 = RungeKutta(order=4)
   rk6 = RungeKutta(order=6)
   rk8 = RungeKutta(order=8)

   # Adaptive step-size methods
   rk45 = AdaptiveRK(order=5)  # Dormand-Prince 5(4) with custom error estimation
   dop853 = AdaptiveRK(order=8)  # Dormand-Prince 8(5,3) with dual error estimation

   # Integration using System.propagate() method
   traj_rk4 = system.propagate(initial_state, tf=2*np.pi, method="fixed", order=4)
   traj_rk8 = system.propagate(initial_state, tf=2*np.pi, method="fixed", order=8)
   traj_adaptive = system.propagate(initial_state, tf=2*np.pi, method="adaptive", order=8)

Adaptive Step-Size Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

HITEN's adaptive integrators automatically adjust the step size to maintain a specified error tolerance. This is particularly useful for systems with varying time scales or when you need guaranteed accuracy.

The adaptive methods use embedded Runge-Kutta schemes that compute both high-order and low-order solutions to estimate the local truncation error. HITEN's implementation uses custom error estimation methods rather than traditional embedded coefficient approaches for better numerical stability. The available methods are:

- **RK45**: Dormand-Prince 5(4) method with custom error estimation using embedded coefficients (not traditional B_LOW approach)
- **DOP853**: Dormand-Prince 8(5,3) method with dual error estimation using both 5th and 3rd order estimates for improved robustness

.. code-block:: python

   from hiten.algorithms.integrators.rk import AdaptiveRK

   # Create adaptive integrators
   rk45 = AdaptiveRK(order=5, rtol=1e-6, atol=1e-8)
   dop853 = AdaptiveRK(order=8, rtol=1e-9, atol=1e-11)
   
   # Note: RK45 uses custom _rk_embedded_step() method with E coefficients
   # DOP853 uses _estimate_error() with both 5th and 3rd order estimates

   # Integration with error control
   traj_rk45 = system.propagate(initial_state, tf=2*np.pi, method="adaptive", order=5)
   traj_dop853 = system.propagate(initial_state, tf=2*np.pi, method="adaptive", order=8)

Symplectic Integrators
~~~~~~~~~~~~~~~~~~~~~~

Symplectic integrators are specialized numerical methods designed for Hamiltonian systems that preserve the symplectic structure of phase space. This preservation leads to excellent long-term energy conservation and stability, making them ideal for celestial mechanics and dynamical systems applications. Symplectic integrators have several key advantages:

1. **Exact Symplectic Structure Preservation**: They maintain the geometric properties of Hamiltonian flow
2. **Superior Long-term Stability**: Energy errors remain bounded rather than growing linearly with time
3. **Phase Space Volume Conservation**: Preserves the volume element in phase space
4. **Backward Error Analysis**: Small energy errors can be interpreted as small perturbations to the original Hamiltonian

Background
~~~~~~~~~~~~~~~~~~~~~~~

Symplectic integrators are based on the mathematical theory of symplectic geometry. In Hamiltonian systems, the phase space is equipped with a symplectic 2-form:

.. math::
   \omega = \sum_{i=1}^{n} dp_i \wedge dq_i

where :math:`q_i` are the generalized coordinates and :math:`p_i` are the conjugate momenta. A symplectic transformation :math:`\phi` preserves this form:

.. math::
   \phi^*\omega = \omega

This geometric property ensures that:

- **Volume preservation**: The phase space volume element is conserved
- **Energy bounds**: Energy errors remain bounded rather than growing linearly
- **Qualitative accuracy**: The qualitative behavior of the system is preserved

HITEN uses the extended phase space technique proposed by Tao (2016), which allows for high-order symplectic integration of non-separable Hamiltonians. The method works by:

1. **Extended Phase Space**: Doubles the phase space dimension by introducing auxiliary variables
2. **Operator Splitting**: Decomposes the Hamiltonian into separable parts
3. **Recursive Composition**: Uses composition methods to achieve high-order accuracy
4. **Polynomial Evaluation**: Leverages polynomial representations for efficient computation

The key advantage is that this approach can handle Hamiltonians that are naturally non-separable. HITEN implements high-order explicit symplectic integrators based on the recursive operator-splitting strategy proposed by Tao (2016). These integrators use an extended phase space technique to achieve high-order accuracy while maintaining exact symplecticity.

.. code-block:: python

   from hiten.algorithms.integrators.symplectic import ExtendedSymplectic

   # Create symplectic integrators of different orders
   symp2 = ExtendedSymplectic(order=2)  # 2nd order
   symp4 = ExtendedSymplectic(order=4)  # 4th order
   symp6 = ExtendedSymplectic(order=6)  # 6th order (default)
   symp8 = ExtendedSymplectic(order=8)  # 8th order

   # Advanced configuration with custom parameters
   symp_high_precision = ExtendedSymplectic(
       order=8,
       c_omega_heuristic=25.0  # Higher value for better energy conservation
   )

   # Integration using System.propagate() method
   traj_symp = system.propagate(
       initial_state, 
       tf=2*np.pi, 
       method="symplectic", 
       order=6
   )

RHS Function Compilation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance, HITEN's integrators use JIT compilation (Numba) for right-hand side (RHS) functions. The compilation requirements vary by integration method:

**General Requirements:**

- RHS functions must have signature ``(t, y)`` where ``t`` is time and ``y`` is the state vector
- Functions should be compatible with Numba compilation
- HITEN automatically handles compilation through the system wrapper

**For Runge-Kutta Methods:**

- RHS functions are compiled with explicit Numba signatures
- The system wrapper ensures proper compilation automatically
- No manual compilation is required from the user

**For Symplectic Methods:**

- RHS functions are provided through the Hamiltonian structure
- The ``_hamiltonian_rhs`` function handles the compilation internally
- Users don't need to provide explicit RHS functions for Hamiltonian systems

Symplectic integrators require systems with specific Hamiltonian structure. They must implement the following attributes:

- ``jac_H``: Jacobian of the Hamiltonian as a list of polynomial coefficients
- ``clmo_H``: Coefficient layout mapping objects for polynomial evaluation
- ``n_dof``: Number of degrees of freedom (must equal 3 for this implementation)
- ``dim``: System dimension (must equal 2 * n_dof)

Symplectic integrators excel at long-term energy conservation and are particularly well-suited for:

1. **Long-term Integration**: When you need to integrate over many orbital periods
2. **Hamiltonian Systems**: Systems that can be expressed in Hamiltonian form
3. **Energy Conservation**: Applications where energy conservation is critical

However, they have limitations:

1. **System Requirements**: Require specific Hamiltonian structure (jac_H, clmo_H, n_dof)
2. **Computational Cost**: Higher-order methods require more function evaluations
3. **Implementation Complexity**: More complex to implement than standard Runge-Kutta methods
4. **Limited Applicability**: Not suitable for non-Hamiltonian systems

Event-Based Integration
~~~~~~~~~~~~~~~~~~~~~~~

HITEN's integrators support event detection during integration, allowing you to find specific conditions or stop integration when certain criteria are met. Events are particularly useful for:

- Detecting when trajectories cross specific surfaces (Poincaré sections)
- Stopping integration when systems reach certain states
- Monitoring system behavior for specific conditions
- Adaptive time stepping based on system events

Basic Event Detection
~~~~~~~~~~~~~~~~~~~~~

Events are defined as scalar functions ``g(t, y)`` that trigger when ``g(t, y) = 0``. HITEN supports three types of event detection:

**Event Directionality:**

- ``direction=0``: Detect any sign change (default)
- ``direction=+1``: Only detect increasing crossings (g goes from negative to positive)
- ``direction=-1``: Only detect decreasing crossings (g goes from positive to negative)

**Event Configuration:**
Events are configured using the :class:`~hiten.algorithms.types.events._EventConfig` class:

.. code-block:: python

   from hiten.algorithms.types.events import _EventConfig

   # Detect any sign change and stop integration
   event_cfg = _EventConfig(direction=0, terminal=True)

   # Detect only increasing crossings and continue integration
   event_cfg = _EventConfig(direction=+1, terminal=False)

   # High precision event detection
   event_cfg = _EventConfig(
       direction=0,
       terminal=True,
       xtol=1e-15,  # Time tolerance
       gtol=1e-15   # Function tolerance
   )

**Simple Event Example:**

.. code-block:: python

   import numpy as np
   from hiten.algorithms.integrators.rk import AdaptiveRK
   from hiten.algorithms.types.events import _EventConfig

   def rhs(t, y):
       return np.array([-y[1], y[0]])  # Harmonic oscillator

   def event_func(t, y):
       """Detect when x = 0.5"""
       return y[0] - 0.5  # Trigger when x = 0.5

   # Create integrator and event configuration
   rk45 = AdaptiveRK(order=5)
   event_cfg = _EventConfig(direction=0, terminal=True)

   # Initial conditions
   y0 = np.array([0.0, 1.0])
   t_vals = np.linspace(0, 10, 1000)

   # Integration with event detection
   solution = rk45.integrate(
       system=rhs,
       y0=y0,
       t_vals=t_vals,
       event_fn=event_func,
       event_cfg=event_cfg
   )

   # Integration stops at the event
   print(f"Event detected at t = {solution.times[-1]:.4f}")
   print(f"State at event: {solution.states[-1]}")

**Advanced Event Features:**

- **Dense Interpolation**: Events use cubic Hermite interpolation for precise timing
- **Multiple Events**: Can detect multiple events in a single integration
- **Hamiltonian Support**: Events work with both standard and Hamiltonian systems
- **Event Refinement**: Automatic refinement of event timing using bisection

**Event-Based Poincaré Sections:**
Events are ideal for creating Poincaré sections in dynamical systems:

.. code-block:: python

   @numba.njit(cache=False)
   def plane_crossing_event(t, y):
       """Detect crossing of the x=0 plane"""
       return y[0]  # Zero when x = 0

   # Configure for plane crossing detection
   event_cfg = _EventConfig(direction=+1, terminal=False)

   # Each integration step that crosses x=0 will be recorded
   solution = rk45.integrate(
       system=rhs,
       y0=y0,
       t_vals=t_vals,
       event_fn=plane_crossing_event,
       event_cfg=event_cfg
   )

Creating Custom Integrators
---------------------------------

HITEN's modular design allows you to create custom integrators by implementing the :class:`~hiten.algorithms.integrators.base._Integrator` interface. However, custom integrators must be properly integrated with HITEN's architecture to work correctly with the framework's direction handling, state validation, and system wrapping.

.. warning::
   Custom integrators should not be used directly in most cases. The recommended approach is to extend the existing factory classes or integrate them through the framework's propagation system.

Basic Custom Integrator
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.integrators.base import _Integrator, _Solution
   from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
   import numpy as np

   class EulerIntegrator(_Integrator):
       """Simple first-order explicit Euler method."""
       
       def __init__(self):
           super().__init__("Explicit Euler")
       
       @property
       def order(self):
           return 1
       
       def integrate(self, system: _DynamicalSystemProtocol, y0: np.ndarray, 
                    t_vals: np.ndarray, **kwargs) -> _Solution:
           """Integrate using explicit Euler method."""
           
           # Validate inputs
           self.validate_inputs(system, y0, t_vals)
           
           # Use the system's RHS method directly
           rhs_func = system.rhs
           
           # Initialize solution arrays with derivatives for Hermite interpolation
           states = np.zeros((len(t_vals), len(y0)), dtype=np.float64)
           derivatives = np.zeros_like(states)
           states[0] = y0.copy()
           derivatives[0] = rhs_func(t_vals[0], y0)
           
           # Euler integration
           for i in range(len(t_vals) - 1):
               dt = t_vals[i+1] - t_vals[i]
               states[i+1] = states[i] + dt * rhs_func(t_vals[i], states[i])
               derivatives[i+1] = rhs_func(t_vals[i+1], states[i+1])
           
           return _Solution(t_vals.copy(), states, derivatives)

   # Use the custom integrator with proper HITEN architecture
   from hiten.algorithms.dynamics.base import _DirectedSystem
   
   euler = EulerIntegrator()
   
   # Wrap system for direction handling (required by HITEN)
   dynsys_dir = _DirectedSystem(system._dynsys, forward=1)
   
   # Use custom integrator with wrapped system
   solution_euler = euler.integrate(dynsys_dir, initial_state, times)

Custom Symplectic Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating custom symplectic integrators requires understanding the underlying mathematical structure. HITEN's symplectic integrators are based on operator splitting methods that decompose the Hamiltonian into separable parts.

Basic Symplectic Integrator Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A symplectic integrator must preserve the symplectic 2-form ω = dp ∧ dq. This is typically achieved through operator splitting:

.. code-block:: python

   from hiten.algorithms.integrators.base import _Integrator, _Solution
   from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
   import numpy as np

   class CustomSymplecticIntegrator(_Integrator):
       """Custom symplectic integrator using operator splitting."""
       
       def __init__(self, order=2, name="Custom Symplectic"):
           if order < 2 or order % 2 != 0:
               raise ValueError("Symplectic order must be even and >= 2")
           super().__init__(name)
           self._order = order
       
       @property
       def order(self):
           return self._order
       
       def integrate(self, system: _DynamicalSystemProtocol, y0: np.ndarray, 
                    t_vals: np.ndarray, **kwargs) -> _Solution:
           """Integrate using custom symplectic method."""
           
           self.validate_inputs(system, y0, t_vals)
           
           # Use the system's RHS method directly
           rhs_func = system.rhs
           
           # Initialize solution with derivatives
           states = np.zeros((len(t_vals), len(y0)), dtype=np.float64)
           derivatives = np.zeros_like(states)
           states[0] = y0.copy()
           derivatives[0] = rhs_func(t_vals[0], y0)
           
           # Symplectic integration using operator splitting
           for i in range(len(t_vals) - 1):
               dt = t_vals[i+1] - t_vals[i]
               states[i+1] = self._symplectic_step(system, states[i], dt)
               derivatives[i+1] = rhs_func(t_vals[i+1], states[i+1])
           
           return _Solution(t_vals.copy(), states, derivatives)
       
       def _symplectic_step(self, system, y, dt):
           """Single symplectic step using operator splitting."""
           # This is a simplified example - real implementation would be more complex
           # and would depend on the specific Hamiltonian structure
           
           # Split into position and momentum updates
           n_dof = system.n_dof
           q = y[:n_dof]
           p = y[n_dof:]
           
           # Half-step momentum update
           p_half = p - 0.5 * dt * self._gradient_H_q(system, q, p)
           
           # Full-step position update
           q_new = q + dt * self._gradient_H_p(system, q, p_half)
           
           # Half-step momentum update
           p_new = p_half - 0.5 * dt * self._gradient_H_q(system, q_new, p_half)
           
           return np.concatenate([q_new, p_new])
       
       def _gradient_H_q(self, system, q, p):
           """Compute gradient of Hamiltonian with respect to position."""
           # This would need to be implemented based on the specific Hamiltonian
           # For now, return zeros as placeholder
           return np.zeros_like(q)
       
       def _gradient_H_p(self, system, q, p):
           """Compute gradient of Hamiltonian with respect to momentum."""
           # This would need to be implemented based on the specific Hamiltonian
           # For now, return zeros as placeholder
           return np.zeros_like(p)

   # Usage example with proper HITEN architecture
   from hiten.algorithms.dynamics.base import _DirectedSystem
   
   custom_symplectic = CustomSymplecticIntegrator(order=4)
   
   # Wrap system for direction handling (required for symplectic integrators)
   dynsys_dir = _DirectedSystem(system._dynsys, forward=1)
   
   # Use custom integrator with wrapped system
   times = np.linspace(0, 2*np.pi, 1000)
   solution = custom_symplectic.integrate(dynsys_dir, initial_state, times)

Next Steps
----------

Once you understand integration methods, you can:

- Learn about orbit correction methods (see :doc:`guide_11_correction`)
- Explore continuation algorithms (see :doc:`guide_12_continuation`)
- Study polynomial methods (see :doc:`guide_14_polynomial`)

For more advanced integration techniques, see the HITEN source code in :mod:`hiten.algorithms.integrators`.
