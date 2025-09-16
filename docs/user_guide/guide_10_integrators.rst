Advanced Integration Methods and Custom Integrators
===================================================

This guide covers HITEN's advanced integration methods, including Runge-Kutta schemes, symplectic integrators, and how to create custom integrators for specialized applications.

Available Integrators
---------------------------

HITEN provides several high-quality integrators optimized for different types of dynamical systems. The main user-facing classes are factory classes that create the appropriate integrator instances.

Integration Methods
~~~~~~~~~~~~~~~~~~~

The primary way to use integrators in HITEN is through the :meth:`~hiten.system.base.System.propagate` method, which supports the following methods:

- ``"scipy"``: Uses SciPy's DOP853 adaptive integrator (default)
- ``"rk"``: Fixed-step Runge-Kutta methods (orders 4, 6, 8)
- ``"adaptive"``: Adaptive step-size Runge-Kutta methods (orders 5, 8)
- ``"symplectic"``: High-order symplectic integrators (orders 2, 4, 6, 8)

For direct access to integrator classes, use the factory classes from :mod:`hiten.algorithms.integrators`.

Runge-Kutta Methods
~~~~~~~~~~~~~~~~~~~

HITEN includes several explicit Runge-Kutta schemes with different orders of accuracy. These are accessed through factory classes:

.. code-block:: python

   from hiten.algorithms.integrators import RungeKutta, AdaptiveRK
   from hiten import System
   import numpy as np

   system = System.from_bodies("earth", "moon")
   initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])

   # Fixed-step Runge-Kutta methods
   rk4 = RungeKutta(order=4)
   rk6 = RungeKutta(order=6)
   rk8 = RungeKutta(order=8)

   # Adaptive step-size methods
   rk45 = AdaptiveRK(order=5)  # Dormand-Prince 5(4)
   dop853 = AdaptiveRK(order=8)  # Dormand-Prince 8(5,3)

   # Integration using System.propagate() method
   times, states_rk4 = system.propagate(initial_state, tf=2*np.pi, method="rk", order=4)
   times, states_rk8 = system.propagate(initial_state, tf=2*np.pi, method="rk", order=8)
   times, states_adaptive = system.propagate(initial_state, tf=2*np.pi, method="adaptive", order=8)

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

   from hiten.algorithms.integrators import ExtendedSymplectic

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
   times, states_symp = system.propagate(
       initial_state, 
       tf=2*np.pi, 
       method="symplectic", 
       order=6
   )

Symplectic integrators require systems with specific Hamiltonian structure. They must implement the following attributes:

- ``jac_H``: Jacobian of the Hamiltonian
- ``clmo_H``: Coefficient layout mapping objects
- ``n_dof``: Number of degrees of freedom

Energy Conservation Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symplectic integrators excel at long-term energy conservation:

.. code-block:: python

   import numpy as np
   from hiten.algorithms.dynamics.utils.energy import crtbp_energy

   def compare_energy_conservation(system, initial_state, tf=100*np.pi):
       """Compare energy conservation between different integrators."""
       
       # Runge-Kutta integration
       times_rk, states_rk = system.propagate(
           initial_state, tf=tf, steps=10000, method="rk", order=8
       )
       
       # Symplectic integration
       times_symp, states_symp = system.propagate(
           initial_state, tf=tf, steps=10000, method="symplectic", order=6
       )
       
       # Calculate energy errors
       initial_energy = crtbp_energy(initial_state, system.mu)
       
       rk_energies = [crtbp_energy(state, system.mu) for state in states_rk]
       symp_energies = [crtbp_energy(state, system.mu) for state in states_symp]
       
       rk_error = np.max(np.abs(np.array(rk_energies) - initial_energy))
       symp_error = np.max(np.abs(np.array(symp_energies) - initial_energy))
       
       print(f"RK8 maximum energy error: {rk_error:.2e}")
       print(f"Symplectic6 maximum energy error: {symp_error:.2e}")
       print(f"Symplectic advantage: {rk_error/symp_error:.1f}x better")

   # Run comparison
   compare_energy_conservation(system, initial_state)

Symplectic integrators are particularly well-suited for:

1. **Long-term Integration**: When you need to integrate over many orbital periods
2. **Hamiltonian Systems**: Systems that can be expressed in Hamiltonian form
3. **Energy Conservation**: Applications where energy conservation is critical

However, they have limitations:

1. **System Requirements**: Require specific Hamiltonian structure (jac_H, clmo_H, n_dof)
2. **Computational Cost**: Higher-order methods require more function evaluations
3. **Implementation Complexity**: More complex to implement than standard Runge-Kutta methods
4. **Limited Applicability**: Not suitable for non-Hamiltonian systems

Creating Custom Integrators
---------------------------------

HITEN's modular design allows you to create custom integrators by implementing the :class:`~hiten.algorithms.integrators.base._Integrator` interface:

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
           self.validate_system(system)
           self.validate_inputs(system, y0, t_vals)
           
           # Initialize solution arrays
           states = np.zeros((len(t_vals), len(y0)))
           states[0] = y0
           
           # Euler integration
           for i in range(len(t_vals) - 1):
               dt = t_vals[i+1] - t_vals[i]
               states[i+1] = states[i] + dt * system.rhs(t_vals[i], states[i])
           
           return _Solution(t_vals, states)

   # Use the custom integrator
   euler = EulerIntegrator()
   solution_euler = euler.integrate(system.dynsys, initial_state, times)

Advanced Custom Integrator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more advanced custom integrators, you can implement adaptive step-size control:

.. code-block:: python

   class AdaptiveEulerIntegrator(_Integrator):
       """Adaptive Euler method with simple step size control."""
       
       def __init__(self, rtol=1e-6, atol=1e-8):
           super().__init__("Adaptive Euler")
           self.rtol = rtol
           self.atol = atol
       
       @property
       def order(self):
           return 1
       
       def integrate(self, system: _DynamicalSystemProtocol, y0: np.ndarray, 
                    t_vals: np.ndarray, **kwargs) -> _Solution:
           """Integrate using adaptive Euler method."""
           
           self.validate_system(system)
           self.validate_inputs(system, y0, t_vals)
           
           # Simple adaptive implementation
           states = [y0.copy()]
           times = [t_vals[0]]
           
           for i in range(len(t_vals) - 1):
               t_curr = t_vals[i]
               t_next = t_vals[i + 1]
               dt = t_next - t_curr
               
               # Single Euler step
               y_curr = states[-1]
               dy = system.rhs(t_curr, y_curr)
               y_next = y_curr + dt * dy
               
               states.append(y_next)
               times.append(t_next)
           
           return _Solution(np.array(times), np.array(states))

Integration with System Propagation
-----------------------------------------

Custom integrators can be integrated with HITEN's system-level propagation by using them directly:

.. code-block:: python

   # Create custom integrator
   custom_integrator = AdaptiveEulerIntegrator(rtol=1e-8)
   
   # Use directly with system's dynamical system
   times = np.linspace(0, 2*np.pi, 1000)
   solution = custom_integrator.integrate(system.dynsys, initial_state, times)
   
   print(f"Custom integrator: {custom_integrator.name}")
   print(f"Solution shape: {solution.states.shape}")

Custom Symplectic Integrators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating custom symplectic integrators requires understanding the underlying mathematical structure. HITEN's symplectic integrators are based on operator splitting methods that decompose the Hamiltonian into separable parts.

Basic Symplectic Integrator Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A symplectic integrator must preserve the symplectic 2-form ω = dp ∧ dq. This is typically achieved through operator splitting:

.. code-block:: python

   from hiten.algorithms.integrators.base import _Integrator, _Solution
   from hiten.algorithms.dynamics.base import _DynamicalSystem
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
       
       def integrate(self, system: _DynamicalSystem, y0: np.ndarray, 
                    t_vals: np.ndarray, **kwargs) -> _Solution:
           """Integrate using custom symplectic method."""
           
           self.validate_inputs(system, y0, t_vals)
           
           # For symplectic integrators, we need Hamiltonian structure
           if not hasattr(system, 'jac_H'):
               raise ValueError("System must provide Hamiltonian structure for symplectic integration")
           
           # Initialize solution
           states = np.zeros((len(t_vals), len(y0)))
           states[0] = y0.copy()
           
           # Symplectic integration using operator splitting
           for i in range(len(t_vals) - 1):
               dt = t_vals[i+1] - t_vals[i]
               states[i+1] = self._symplectic_step(system, states[i], dt)
           
           return _Solution(t_vals, states)
       
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

Advanced Symplectic Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more sophisticated symplectic integrators, you can implement higher-order composition methods:

.. code-block:: python

   class HighOrderSymplecticIntegrator(_Integrator):
       """High-order symplectic integrator using composition methods."""
       
       def __init__(self, order=4, composition_type="suzuki"):
           if order < 2 or order % 2 != 0:
               raise ValueError("Order must be even and >= 2")
           super().__init__(f"High-Order Symplectic {order}")
           self._order = order
           self._composition_type = composition_type
           self._coefficients = self._get_composition_coefficients()
       
       def _get_composition_coefficients(self):
           """Get composition coefficients for high-order methods."""
           if self._composition_type == "suzuki":
               # Suzuki composition for 4th order
               if self._order == 4:
                   return [1/(4-4**(1/3)), 1-2/(4-4**(1/3)), 1/(4-4**(1/3))]
               # Add more orders as needed
           elif self._composition_type == "yoshida":
               # Yoshida composition
               if self._order == 4:
                   return [1/(2-2**(1/3)), -2**(1/3)/(2-2**(1/3)), 1/(2-2**(1/3))]
           
           # Default to 2nd order
           return [0.5, 0.5]
       
       def integrate(self, system: _DynamicalSystem, y0: np.ndarray, 
                    t_vals: np.ndarray, **kwargs) -> _Solution:
           """Integrate using high-order symplectic composition."""
           
           self.validate_inputs(system, y0, t_vals)
           
           states = np.zeros((len(t_vals), len(y0)))
           states[0] = y0.copy()
           
           for i in range(len(t_vals) - 1):
               dt = t_vals[i+1] - t_vals[i]
               states[i+1] = self._composition_step(system, states[i], dt)
           
           return _Solution(t_vals, states)
       
       def _composition_step(self, system, y, dt):
           """Single step using composition method."""
           current_y = y.copy()
           
           for coeff in self._coefficients:
               current_y = self._basic_symplectic_step(system, current_y, coeff * dt)
           
           return current_y
       
       def _basic_symplectic_step(self, system, y, dt):
           """Basic 2nd order symplectic step."""
           # Implement basic symplectic step here
           # This is a placeholder - real implementation would be more complex
           return y

Next Steps
----------

Once you understand integration methods, you can:

- Learn about orbit correction methods (see :doc:`guide_11_correction`)
- Explore continuation algorithms (see :doc:`guide_12_continuation`)
- Study polynomial methods (see :doc:`guide_14_polynomial`)

For more advanced integration techniques, see the HITEN source code in :mod:`hiten.algorithms.integrators`.
