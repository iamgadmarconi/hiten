Advanced Integration Methods and Custom Integrators
===================================================

This guide covers HITEN's advanced integration methods, including Runge-Kutta schemes, symplectic integrators, and how to create custom integrators for specialized applications.

Available Integrators
---------------------------

HITEN provides several high-quality integrators optimized for different types of dynamical systems.

Runge-Kutta Methods
~~~~~~~~~~~~~~~~~~~

HITEN includes several explicit Runge-Kutta schemes with different orders of accuracy:

.. code-block:: python

   from hiten.algorithms.integrators import RK4Integrator, RK6Integrator, RK8Integrator, DOP853Integrator
   from hiten import System
   import numpy as np

   system = System.from_bodies("earth", "moon")
   initial_state = np.array([0.8, 0.0, 0.0, 0.0, 0.1, 0.0])

   # 4th order Runge-Kutta
   rk4 = RK4Integrator()
   times = np.linspace(0, 2*np.pi, 1000)
   solution_rk4 = rk4.integrate(system.dynsys, initial_state, times)

   # 6th order Runge-Kutta
   rk6 = RK6Integrator()
   solution_rk6 = rk6.integrate(system.dynsys, initial_state, times)

   # 8th order Runge-Kutta
   rk8 = RK8Integrator()
   solution_rk8 = rk8.integrate(system.dynsys, initial_state, times)

   # Dormand-Prince 8(5)3 adaptive method
   dop853 = DOP853Integrator()
   solution_dop853 = dop853.integrate(system.dynsys, initial_state, times)

Symplectic Integrators
~~~~~~~~~~~~~~~~~~~~~~

For Hamiltonian systems, symplectic integrators preserve the symplectic structure and provide excellent long-term energy conservation:

.. code-block:: python

   from hiten.algorithms.integrators import SymplecticIntegrator

   # Create symplectic integrator (order 4)
   symp = SymplecticIntegrator(order=4)
   
   # Symplectic integrators work best with polynomial Hamiltonian systems
   # For CR3BP, you might need to convert to Hamiltonian form
   solution_symp = symp.integrate(system.dynsys, initial_state, times)

Integration Parameters and Control
----------------------------------------

Control integration accuracy and performance through various parameters:

Step Size Control
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fixed step size integration
   times_fixed = np.linspace(0, 2*np.pi, 1000)
   solution_fixed = rk8.integrate(system.dynsys, initial_state, times_fixed)

   # Adaptive step size (for DOP853)
   dop853_adaptive = DOP853Integrator(
       rtol=1e-8,  # Relative tolerance
       atol=1e-10  # Absolute tolerance
   )
   solution_adaptive = dop853_adaptive.integrate(system.dynsys, initial_state, times)

Tolerance Settings
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High accuracy integration
   high_precision = DOP853Integrator(
       rtol=1e-12,
       atol=1e-14,
       max_step=0.01
   )

   # Fast integration
   fast_integration = RK4Integrator()
   # RK4 uses fixed step size, so accuracy is controlled by step count

Performance Comparison
---------------------------

Compare different integrators for your specific application:

.. code-block:: python

   import time

   def benchmark_integrator(integrator, system, initial_state, times, name):
       start_time = time.time()
       solution = integrator.integrate(system.dynsys, initial_state, times)
       end_time = time.time()
       
       # Check energy conservation
       from hiten.algorithms.dynamics.utils.energy import crtbp_energy
       initial_energy = crtbp_energy(initial_state, system.mu)
       final_energy = crtbp_energy(solution.states[-1], system.mu)
       energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
       
       print(f"{name}:")
       print(f"  Time: {end_time - start_time:.4f} seconds")
       print(f"  Energy error: {energy_error:.2e}")
       print(f"  Order: {integrator.order}")
       print()

   # Benchmark different integrators
   times_bench = np.linspace(0, 10*np.pi, 5000)
   
   benchmark_integrator(RK4Integrator(), system, initial_state, times_bench, "RK4")
   benchmark_integrator(RK8Integrator(), system, initial_state, times_bench, "RK8")
   benchmark_integrator(DOP853Integrator(), system, initial_state, times_bench, "DOP853")

Creating Custom Integrators
---------------------------------

HITEN's modular design allows you to create custom integrators by implementing the `_Integrator` interface:

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

Symplectic Integrators
~~~~~~~~~~~~~~~~~~~~~~~

For Hamiltonian systems, symplectic integrators preserve the symplectic structure and provide excellent long-term energy conservation:

.. code-block:: python

   from hiten.algorithms.integrators.symplectic import ExtendedSymplectic

   # Create symplectic integrators of different orders
   symp2 = ExtendedSymplectic(order=2)  # 2nd order
   symp4 = ExtendedSymplectic(order=4)  # 4th order
   symp6 = ExtendedSymplectic(order=6)  # 6th order (default)
   symp8 = ExtendedSymplectic(order=8)  # 8th order

   # Symplectic integrators work with polynomial Hamiltonian systems
   # They require the system to provide jac_H, clmo_H, and n_dof attributes
   solution_symp = symp6.integrate(hamiltonian_system, initial_state, times)

   print(f"Symplectic integrator order: {symp6.order}")
   print(f"Solution shape: {solution_symp.states.shape}")

Integration with System Propagation
-----------------------------------------

Custom integrators can be integrated with HITEN's system-level propagation:

.. code-block:: python

   class CustomSystem(System):
       """System with custom integrator support."""
       
       def propagate_with_integrator(self, initial_conditions, tf, integrator, **kwargs):
           """Propagate using a custom integrator."""
           
           # Create time array
           if 'steps' in kwargs:
               times = np.linspace(0, tf, kwargs['steps'])
           else:
               times = np.linspace(0, tf, 1000)
           
           # Use custom integrator
           solution = integrator.integrate(self.dynsys, initial_conditions, times)
           
           return solution.times, solution.states

   # Example usage
   custom_system = CustomSystem.from_bodies("earth", "moon")
   custom_integrator = AdaptiveRK4Integrator(rtol=1e-8)
   
   times, trajectory = custom_system.propagate_with_integrator(
       initial_state, 2*np.pi, custom_integrator, steps=2000
   )

Best Practices
--------------------

1. **Choose the right integrator**:
   - Use RK4 for simple, fast integration
   - Use RK8 or DOP853 for high accuracy
   - Use symplectic methods for long-term Hamiltonian integration

2. **Set appropriate tolerances**:
   - Balance accuracy with computational cost
   - Use relative tolerances around 1e-6 to 1e-8 for most applications
   - Use absolute tolerances around 1e-8 to 1e-10

3. **Monitor energy conservation**:
   - Check energy conservation for Hamiltonian systems
   - Use symplectic integrators for long-term integration

4. **Profile performance**:
   - Benchmark different integrators for your specific problem
   - Consider the trade-off between accuracy and speed

5. **Custom integrator design**:
   - Implement proper error estimation for adaptive methods
   - Use efficient data structures for large-scale problems
   - Consider parallelization for computationally intensive methods

Next Steps
----------

Once you understand integration methods, you can:

- Learn about orbit correction methods (see :doc:`guide_11_correction`)
- Explore continuation algorithms (see :doc:`guide_12_continuation`)
- Study polynomial methods (see :doc:`guide_14_polynomial`)

For more advanced integration techniques, see the HITEN source code in :mod:`hiten.algorithms.integrators`.
