Custom Dynamical Systems
=========================

This guide covers how to create and work with dynamical systems in HITEN, including both built-in systems and custom implementations. The dynamics framework provides a flexible interface for defining, analyzing, and integrating dynamical systems with emphasis on astrodynamics applications.

Understanding the Dynamics Framework
------------------------------------

HITEN provides a comprehensive framework for dynamical systems through several key components:

- **Core Framework**: :class:`~hiten.algorithms.dynamics.base._DynamicalSystem` abstract base class and :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol` interface
- **Built-in Systems**: CR3BP equations, Hamiltonian systems, and generic RHS adapters
- **Utilities**: Energy analysis, stability analysis, and linear algebra tools
- **Integration**: Compatible with all HITEN integrators

Built-in Dynamical Systems
---------------------------

HITEN provides several pre-built dynamical systems for common applications:

CR3BP Systems
~~~~~~~~~~~~~

Several built-in systems are available in HITEN:

.. code-block:: python

   from hiten.algorithms.dynamics import rtbp_dynsys, jacobian_dynsys, variational_dynsys
   import numpy as np
   from scipy.integrate import solve_ivp

   # Create CR3BP system (Earth-Moon system)
   system = rtbp_dynsys(mu=0.01215, name="Earth-Moon")
   
   # Initial state near L1 point
   initial_state = np.array([0.8, 0, 0, 0, 0.1, 0])
   
   # Integrate using SciPy
   times = np.linspace(0, 10, 1000)
   sol = solve_ivp(system.rhs, [0, 10], initial_state, t_eval=times)
   
   print(f"System dimension: {system.dim}")
   print(f"Mass parameter: {system.mu}")

   # Create Jacobian evaluation system
   jac_system = jacobian_dynsys(mu=0.01215)
   jacobian = jac_system.rhs(0.0, initial_state[:3])  # Position only
   
   # Create variational equations system for STM propagation
   var_system = variational_dynsys(mu=0.01215)

Generic RHS Systems
~~~~~~~~~~~~~~~~~~~

For custom ODE systems, use the RHS adapter:

.. code-block:: python

   from hiten.algorithms.dynamics import create_rhs_system
   import numpy as np

   def harmonic_oscillator(t, y):
       """Simple harmonic oscillator: dy/dt = [v, -ω²x]"""
       return np.array([y[1], -y[0]])

   def duffing_oscillator(t, y):
       """Duffing oscillator with forcing"""
       x, v = y[0], y[1]
       alpha, beta, gamma, omega = 1.0, 0.1, 0.2, 1.2
       return np.array([v, -alpha*x - beta*x**3 + gamma*np.cos(omega*t)])

   # Create systems
   harmonic = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic")
   duffing = create_rhs_system(duffing_oscillator, dim=2, name="Duffing")
   
   # Test the systems
   state = np.array([1.0, 0.0])
   print(f"Harmonic derivative: {harmonic.rhs(0.0, state)}")
   print(f"Duffing derivative: {duffing.rhs(0.0, state)}")

Hamiltonian Systems
~~~~~~~~~~~~~~~~~~~

For polynomial Hamiltonian systems from center manifold reduction:

.. code-block:: python

   from hiten.algorithms.dynamics import create_hamiltonian_system
   from hiten.algorithms.polynomial.base import _init_index_tables
   import numpy as np

   # Example: Create a simple polynomial Hamiltonian system
   # (In practice, you would get these from center manifold computation)
   degree = 4
   n_dof = 3
   
   # Initialize polynomial tables
   psi_table, clmo_table = _init_index_tables(degree)
   
   # Create dummy H_blocks (in practice from normal form computation)
   H_blocks = [np.zeros(1) for _ in range(degree + 1)]
   encode_dict_list = [{} for _ in range(degree + 1)]
   
   # Create Hamiltonian system
   ham_system = create_hamiltonian_system(
       H_blocks, degree, psi_table, clmo_table, 
       encode_dict_list, n_dof=3, name="Center Manifold"
   )
   
   print(f"Hamiltonian system dimension: {ham_system.dim}")
   print(f"Degrees of freedom: {ham_system.n_dof}")

Creating Custom Dynamical Systems
---------------------------------

HITEN provides the :func:`~hiten.algorithms.dynamics.create_rhs_system` function for creating custom dynamical systems from user-defined ODE functions. This is the recommended approach rather than subclassing the base classes directly.

Basic Custom System
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.dynamics import create_rhs_system
   import numpy as np

   def harmonic_oscillator(t, y, omega=1.0):
       """Simple harmonic oscillator: dy/dt = [v, -ω²x]"""
       return np.array([y[1], -omega**2 * y[0]])

   def duffing_oscillator(t, y, alpha=1.0, beta=0.1, gamma=0.2, omega=1.2, delta=0.0):
       """Duffing oscillator with forcing"""
       x, v = y[0], y[1]
       forcing = gamma * np.cos(omega * t) if gamma != 0 else 0.0
       return np.array([v, -delta*v - alpha*x - beta*x**3 + forcing])

   # Create systems using the factory function
   oscillator = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic Oscillator")
   duffing = create_rhs_system(duffing_oscillator, dim=2, name="Duffing Oscillator")
   
   # Test the systems
   state = np.array([1.0, 0.0])
   print(f"Harmonic derivative: {oscillator.rhs(0.0, state)}")
   print(f"Duffing derivative: {duffing.rhs(0.0, state)}")

Advanced Custom System with Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For systems requiring parameter validation or additional methods, you can create a wrapper class:

.. code-block:: python

   class ParameterizedOscillator:
       """Wrapper for parameterized oscillator with validation."""
       
       def __init__(self, omega=1.0, damping=0.0, forcing_amplitude=0.0, forcing_freq=1.0):
           self.omega = omega
           self.damping = damping
           self.forcing_amplitude = forcing_amplitude
           self.forcing_freq = forcing_freq
           
           # Validate parameters
           if omega <= 0:
               raise ValueError("Natural frequency must be positive")
           if damping < 0:
               raise ValueError("Damping coefficient must be non-negative")
           
           # Create the RHS function with parameters
           def rhs_func(t, y):
               x, v = y[0], y[1]
               forcing = self.forcing_amplitude * np.cos(self.forcing_freq * t)
               return np.array([v, -self.omega**2 * x - self.damping * v + forcing])
           
           # Create the dynamical system
           self.system = create_rhs_system(rhs_func, dim=2, name="Parameterized Oscillator")
       
       @property
       def rhs(self):
           """Access to the RHS function."""
           return self.system.rhs
       
       @property
       def dim(self):
           """System dimension."""
           return self.system.dim
       
       def energy(self, y):
           """Compute system energy (for conservative case)."""
           if self.damping == 0 and self.forcing_amplitude == 0:
               x, v = y[0], y[1]
               return 0.5 * v**2 + 0.5 * self.omega**2 * x**2
           return None
       
       def __repr__(self):
           return (f"ParameterizedOscillator(omega={self.omega}, damping={self.damping}, "
                   f"forcing_amp={self.forcing_amplitude}, forcing_freq={self.forcing_freq})")

   # Use the parameterized system
   osc = ParameterizedOscillator(omega=2.0, damping=0.1, forcing_amplitude=0.5, forcing_freq=1.5)
   
   # Test the system
   state = np.array([1.0, 0.0])
   derivative = osc.rhs(0.0, state)
   energy = osc.energy(state)
   print(f"Derivative: {derivative}")
   print(f"Energy: {energy}")

Working with High-Level Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In practice, most HITEN users work with high-level `System` objects rather than low-level dynamical systems:

.. code-block:: python

   from hiten.system import System
   from hiten.algorithms.dynamics import create_rhs_system
   import numpy as np

   # Create a physical system (Earth-Moon CR3BP)
   system = System.from_bodies("earth", "moon")
   
   # Access the underlying dynamical system
   dynsys = system.dynsys  # This is an _RTBPRHS instance
   var_dynsys = system.var_dynsys  # Variational equations
   
   # Create custom systems for specialized analysis
   def custom_perturbation(t, y, base_system, perturbation_strength=0.01):
       """Add small perturbation to base system."""
       base_derivative = base_system.rhs(t, y)
       # Add small random perturbation
       perturbation = perturbation_strength * np.random.normal(0, 1, len(y))
       return base_derivative + perturbation
   
   # Create perturbed system
   perturbed_system = create_rhs_system(
       lambda t, y: custom_perturbation(t, y, dynsys, 0.001),
       dim=6, 
       name="Perturbed CR3BP"
   )
   
   print(f"Base system: {dynsys}")
   print(f"Perturbed system: {perturbed_system}")

Integration with HITEN Integrators
------------------------------------

HITEN provides several integrators that work with dynamical systems:

Using Runge-Kutta Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.integrators.rk import RungeKutta, AdaptiveRK
   from hiten.algorithms.dynamics import create_rhs_system
   import numpy as np

   # Create a system using the RHS adapter
   def harmonic_oscillator(t, y, omega=2.0):
       return np.array([y[1], -omega**2 * y[0]])
   
   oscillator = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic")
   initial_state = np.array([1.0, 0.0])
   times = np.linspace(0, 2*np.pi, 100)

   # Fixed-step Runge-Kutta (orders 4, 6, 8)
   rk4 = RungeKutta(order=4)
   rk8 = RungeKutta(order=8)
   
   # Adaptive Runge-Kutta (orders 5, 8)
   rk45 = AdaptiveRK(order=5)
   dop853 = AdaptiveRK(order=8)

   # Integrate with different methods
   sol_rk4 = rk4.integrate(oscillator, initial_state, times)
   sol_rk8 = rk8.integrate(oscillator, initial_state, times)
   sol_adaptive = rk45.integrate(oscillator, initial_state, times)

   print(f"RK4 solution shape: {sol_rk4.states.shape}")
   print(f"RK8 solution shape: {sol_rk8.states.shape}")
   print(f"Adaptive solution shape: {sol_adaptive.states.shape}")

Using SciPy Integration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy.integrate import solve_ivp
   import numpy as np

   # Create system
   system = rtbp_dynsys(mu=0.01215)
   initial_state = np.array([0.8, 0, 0, 0, 0.1, 0])
   
   # Integrate with SciPy
   times = np.linspace(0, 10, 1000)
   sol = solve_ivp(system.rhs, [0, 10], initial_state, t_eval=times, 
                   method='DOP853', rtol=1e-12, atol=1e-12)
   
   print(f"Integration successful: {sol.success}")
   print(f"Number of function evaluations: {sol.nfev}")

Energy and Stability Analysis
-----------------------------

HITEN provides utilities for analyzing dynamical systems:

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.common.energy import (crtbp_energy, kinetic_energy, 
                                                effective_potential, hill_region)
   import numpy as np

   # CR3BP energy analysis
   state = np.array([0.8, 0, 0, 0, 0.1, 0])
   mu = 0.01215
   
   # Compute different energy components
   total_energy = crtbp_energy(state, mu)
   kinetic = kinetic_energy(state)
   potential = effective_potential(state, mu)
   
   print(f"Total energy: {total_energy:.6f}")
   print(f"Kinetic energy: {kinetic:.6f}")
   print(f"Effective potential: {potential:.6f}")
   
   # Generate Hill region for visualization
   X, Y, Z = hill_region(mu=mu, C=total_energy, n_grid=100)
   print(f"Hill region shape: {Z.shape}")

Stability Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.linalg.backend import _LinalgBackend
   from hiten.algorithms.dynamics import jacobian_dynsys
   import numpy as np

   # Create Jacobian system
   jac_system = jacobian_dynsys(mu=0.01215)
   
   # Evaluate Jacobian at a point
   state = np.array([0.8, 0, 0])
   jacobian = jac_system.rhs(0.0, state)
   
   # Analyze stability using linalg backend
   linalg_backend = _LinalgBackend()
   stable_vals, unstable_vals, center_vals, Ws, Wu, Wc = linalg_backend.eigenvalue_decomposition(jacobian)
   
   print(f"Stable eigenvalues: {stable_vals}")
   print(f"Unstable eigenvalues: {unstable_vals}")
   print(f"Center eigenvalues: {center_vals}")
   print(f"Stable subspace dimension: {Ws.shape[1]}")
   print(f"Unstable subspace dimension: {Wu.shape[1]}")
   print(f"Center subspace dimension: {Wc.shape[1]}")

System Testing and Validation
-----------------------------

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
   def harmonic_oscillator(t, y, omega=2.0):
       return np.array([y[1], -omega**2 * y[0]])
   
   oscillator = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic")
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

   # Test integration with different methods
   from hiten.algorithms.integrators.rk import RungeKutta, AdaptiveRK
   from hiten.algorithms.dynamics import create_rhs_system
   
   # Create test system
   def harmonic_oscillator(t, y, omega=2.0):
       return np.array([y[1], -omega**2 * y[0]])
   
   oscillator = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic")
   
   integrators = [
       RungeKutta(order=4),
       RungeKutta(order=8), 
       AdaptiveRK(order=5)
   ]
   initial_state = np.array([1.0, 0.0])
   times = np.linspace(0, 2*np.pi, 100)
   
   for integrator in integrators:
       solution = test_integration(oscillator, initial_state, times, integrator)

Best Practices
--------------

1. **Use the RHS Adapter Pattern**:
   - Use :func:`~hiten.algorithms.dynamics.create_rhs_system` for custom systems
   - Let HITEN handle JIT compilation automatically
   - Avoid direct subclassing of `_DynamicalSystem` unless absolutely necessary

2. **System Design**:
   - Create simple RHS functions with clear parameter handling
   - Use wrapper classes for complex parameter validation
   - Follow the :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol` interface

3. **High-Level vs Low-Level**:
   - Use high-level `System` objects for CR3BP applications
   - Use low-level dynamical systems only for specialized analysis
   - Leverage built-in systems (`rtbp_dynsys`, `variational_dynsys`) when possible

4. **Testing**:
   - Test with various initial conditions
   - Validate state dimensions
   - Check for numerical stability

5. **Performance**:
   - Use built-in systems when possible (CR3BP, RHS adapters)
   - Leverage HITEN's optimized integrators
   - Consider adaptive step sizes for complex dynamics

Troubleshooting
---------------

Common issues and solutions:

- **Dimension Mismatch**: Ensure state vectors match system dimension
- **JIT Compilation Errors**: Avoid Python objects in compiled functions
- **Integration Failures**: Check for singularities or numerical instabilities
- **Performance Issues**: Use appropriate integrator for your problem type

Next Steps
----------

Once you understand the dynamics framework, you can:

- Learn about advanced integration techniques (see :doc:`guide_10_integrators`)
- Explore polynomial methods (see :doc:`guide_14_polynomial`)
- Study Hamiltonian mechanics (see :doc:`guide_07_center_manifold`)
- Analyze periodic orbits and manifolds (see :doc:`guide_05_manifolds`)

For more advanced dynamical system techniques, see the HITEN source code in :mod:`hiten.algorithms.dynamics`.
