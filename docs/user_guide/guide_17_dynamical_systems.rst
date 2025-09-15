Custom Dynamical Systems
=========================

This guide covers how to create custom dynamical systems in HITEN, including both standard dynamical systems and Hamiltonian systems. Understanding how to implement custom systems is essential for extending HITEN's capabilities to new problems and research applications.

Understanding Dynamical Systems
-------------------------------

HITEN provides a flexible framework for defining dynamical systems through the :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` abstract base class. All systems must implement the :class:`hiten.algorithms.dynamics.base._DynamicalSystemProtocol` interface.

Creating Standard Dynamical Systems
-----------------------------------

For general dynamical systems, subclass :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`:

Basic Custom System
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics.base import _DynamicalSystem
   from hiten.algorithms.integrators.base import _Solution
   import numpy as np
   from numba import njit
   from hiten.algorithms.utils.config import FASTMATH

   class SimpleOscillator(_DynamicalSystem):
       """Simple harmonic oscillator: d²x/dt² + ω²x = 0."""
       
       def __init__(self, omega=1.0, name="Simple Oscillator"):
           super().__init__(dim=2)  # [position, velocity]
           self.omega = omega
           self.name = name
           
           # Create JIT-compiled RHS function
           @njit(fastmath=FASTMATH, cache=False)
           def _oscillator_rhs(t: float, y: np.ndarray, _omega=omega) -> np.ndarray:
               # y = [x, v], dy/dt = [v, -ω²x]
               return np.array([y[1], -_omega**2 * y[0]])
           
           self._rhs = _oscillator_rhs
       
       @property
       def rhs(self):
           """Right-hand side function for the oscillator."""
           return self._rhs
       
       def __repr__(self):
           return f"SimpleOscillator(omega={self.omega}, name='{self.name}')"

   # Use the custom system
   oscillator = SimpleOscillator(omega=2.0)
   
   # Integrate the system
   from hiten.algorithms.integrators.rk import RK8Integrator
   
   initial_state = np.array([1.0, 0.0])  # Start at x=1, v=0
   times = np.linspace(0, 4*np.pi, 1000)
   
   integrator = RK8Integrator()
   solution = integrator.integrate(oscillator, initial_state, times)
   
   print(f"Oscillator period: {2*np.pi/oscillator.omega:.4f}")

Advanced Custom System
~~~~~~~~~~~~~~~~~~~~~~

For more complex systems with parameters and validation:

.. code-block:: python

   class DuffingOscillator(_DynamicalSystem):
       """Duffing oscillator"""
       
       def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, omega=1.0, 
                    delta=0.0, name="Duffing Oscillator"):
           super().__init__(dim=2)  # [position, velocity]
           self.alpha = alpha
           self.beta = beta
           self.gamma = gamma
           self.omega = omega
           self.delta = delta
           self.name = name
           
           # Create JIT-compiled RHS function
           @njit(fastmath=FASTMATH, cache=False)
           def _duffing_rhs(t: float, y: np.ndarray, _alpha=alpha, _beta=beta, 
                           _gamma=gamma, _omega=omega, _delta=delta) -> np.ndarray:
               # y = [x, v], dy/dt = [v, -δv - αx - βx³ + γcos(ωt)]
               x, v = y[0], y[1]
               forcing = _gamma * np.cos(_omega * t) if _gamma != 0 else 0.0
               return np.array([v, -_delta*v - _alpha*x - _beta*x**3 + forcing])
           
           self._rhs = _duffing_rhs
       
       @property
       def rhs(self):
           """Right-hand side function for the Duffing oscillator."""
           return self._rhs
       
       def validate_parameters(self):
           """Validate system parameters."""
           if self.alpha < 0 and self.beta < 0:
               raise ValueError("System parameters lead to unstable dynamics")
           return True
       
       def __repr__(self):
           return (f"DuffingOscillator(alpha={self.alpha}, beta={self.beta}, "
                   f"gamma={self.gamma}, omega={self.omega}, delta={self.delta})")

   # Use the advanced system
   duffing = DuffingOscillator(alpha=1.0, beta=0.1, gamma=0.2, omega=1.2)
   duffing.validate_parameters()
   
   # Integrate with different initial conditions
   initial_states = [
       np.array([1.0, 0.0]),  # Small amplitude
       np.array([2.0, 0.0]),  # Large amplitude
   ]
   
   for i, y0 in enumerate(initial_states):
       solution = integrator.integrate(duffing, y0, times)
       print(f"Initial condition {i+1}: Final state = {solution.states[-1]}")

Creating Hamiltonian Systems
----------------------------

For Hamiltonian systems that can be used with symplectic integrators, implement additional attributes:

Basic Hamiltonian System
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SimplePendulum(_DynamicalSystem):
       """Simple pendulum: H = p²/(2m) - mgl cos(θ)."""
       
       def __init__(self, m=1.0, g=9.81, l=1.0, name="Simple Pendulum"):
           super().__init__(dim=2)  # [θ, p]
           self.m = m
           self.g = g
           self.l = l
           self.name = name
           self.n_dof = 1  # Required for symplectic integrators
           
           # Create JIT-compiled RHS function
           @njit(fastmath=FASTMATH, cache=False)
           def _pendulum_rhs(t: float, y: np.ndarray, _m=m, _g=g, _l=l) -> np.ndarray:
               # y = [θ, p], dy/dt = [p/(ml²), -mgl sin(θ)]
               theta, p = y[0], y[1]
               return np.array([p/(_m*_l**2), -_m*_g*_l*np.sin(theta)])
           
           self._rhs = _pendulum_rhs
           
           # For symplectic integration, we need polynomial representations
           # This is a simplified example - real implementation would be more complex
           self.jac_H = None  # Jacobian of Hamiltonian
           self.clmo_H = None  # Coefficient layout mapping
       
       @property
       def rhs(self):
           """Right-hand side function for the pendulum."""
           return self._rhs
       
       def hamiltonian(self, y):
           """Compute the Hamiltonian at state y."""
           theta, p = y[0], y[1]
           return p**2/(2*self.m*self.l**2) - self.m*self.g*self.l*np.cos(theta)
       
       def __repr__(self):
           return f"SimplePendulum(m={self.m}, g={self.g}, l={self.l})"

   # Use the Hamiltonian system
   pendulum = SimplePendulum(m=1.0, g=9.81, l=1.0)
   
   # Check energy conservation
   initial_state = np.array([np.pi/4, 0.0])  # Start at θ=π/4, p=0
   initial_energy = pendulum.hamiltonian(initial_state)
   
   solution = integrator.integrate(pendulum, initial_state, times)
   final_energy = pendulum.hamiltonian(solution.states[-1])
   
   print(f"Energy conservation error: {abs(final_energy - initial_energy):.2e}")

Polynomial Hamiltonian System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For systems that work with symplectic integrators, you need polynomial representations:

.. code-block:: python

   from hiten.algorithms.polynomial.base import _init_index_tables, _make_poly
   from hiten.algorithms.polynomial.operations import _polynomial_evaluate, _polynomial_variable

   class PolynomialHamiltonianSystem(_DynamicalSystem):
       """Example polynomial Hamiltonian system for symplectic integration."""
       
       def __init__(self, degree=4, name="Polynomial Hamiltonian"):
           super().__init__(dim=6)  # 3 DOF system
           self.n_dof = 3  # Required for symplectic integrators
           self.degree = degree
           self.name = name
           
           # Initialize polynomial tables
           self.psi, self.clmo = _init_index_tables(degree)
           
           # Create polynomial Hamiltonian (simplified example)
           self._setup_polynomial_hamiltonian()
           
           # Create RHS function
           @njit(fastmath=FASTMATH, cache=False)
           def _polynomial_rhs(t: float, y: np.ndarray) -> np.ndarray:
               # This would compute derivatives from polynomial Hamiltonian
               # Simplified implementation
               return np.zeros(6)
           
           self._rhs = _polynomial_rhs
       
       def _setup_polynomial_hamiltonian(self):
           """Set up polynomial representation of Hamiltonian."""
           # This is a simplified example - real implementation would be more complex
           # Create dummy polynomial coefficients for demonstration
           self.jac_H = []
           for i in range(6):
               # Create polynomial for each variable
               var_poly = _polynomial_variable(i, self.degree, self.psi, 
                                              self.clmo, self._ENCODE_DICT_GLOBAL)
               self.jac_H.append(var_poly)
       
       @property
       def rhs(self):
           """Right-hand side function."""
           return self._rhs
       
       def __repr__(self):
           return f"PolynomialHamiltonianSystem(degree={self.degree})"

System Integration and Testing
------------------------------

Test your custom systems thoroughly:

System Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_system_validation(system, test_states):
       """Test system validation and error handling."""
       
       print(f"Testing {system.name}...")
       
       # Test valid states
       for i, state in enumerate(test_states):
           try:
               system.validate_state(state)
               print(f"  Valid state {i+1}: ✓")
           except ValueError as e:
               print(f"  Invalid state {i+1}: {e}")
       
       # Test RHS function
       try:
           test_state = test_states[0]
           derivative = system.rhs(0.0, test_state)
           print(f"  RHS evaluation: ✓ (shape: {derivative.shape})")
       except Exception as e:
           print(f"  RHS evaluation failed: {e}")

   # Test the oscillator system
   oscillator = SimpleOscillator(omega=2.0)
   test_states = [
       np.array([1.0, 0.0]),  # Valid
       np.array([0.5, 0.5]),  # Valid
       np.array([1.0]),       # Invalid dimension
   ]
   
   test_system_validation(oscillator, test_states)

Integration Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_integration(system, initial_state, times, integrator):
       """Test system integration with different integrators."""
       
       print(f"Testing integration of {system.name}...")
       
       try:
           solution = integrator.integrate(system, initial_state, times)
           print(f"  Integration successful: ✓")
           print(f"  Solution shape: {solution.states.shape}")
           print(f"  Time range: [{solution.times[0]:.2f}, {solution.times[-1]:.2f}]")
           
           # Check for NaN or infinite values
           if np.any(np.isnan(solution.states)) or np.any(np.isinf(solution.states)):
               print(f"  Warning: Solution contains NaN or infinite values")
           
           return solution
           
       except Exception as e:
           print(f"  Integration failed: {e}")
           return None

   # Test integration
   from hiten.algorithms.integrators.rk import RK4Integrator, RK8Integrator
   
   integrators = [RK4Integrator(), RK8Integrator()]
   initial_state = np.array([1.0, 0.0])
   times = np.linspace(0, 2*np.pi, 100)
   
   for integrator in integrators:
       solution = test_integration(oscillator, initial_state, times, integrator)

Best Practices
--------------

1. **Use Numba JIT Compilation**:
   - Always use `@njit` decorators for RHS functions
   - Use `fastmath=FASTMATH` for better performance
   - Avoid Python objects in compiled functions

Troubleshooting
---------------

Next Steps
----------

Once you understand custom dynamical systems, you can:

- Learn about advanced integration techniques (see :doc:`guide_10_integrators`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study Hamiltonian mechanics (see :doc:`guide_07_center_manifold`)

For more advanced dynamical system techniques, see the HITEN source code in :mod:`hiten.algorithms.dynamics`.
