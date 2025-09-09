Algorithms Module
=================

The algorithms module contains the core computational algorithms for dynamical 
systems analysis in the CR3BP.

Dynamics
--------

The dynamics module provides a comprehensive framework for defining, analyzing, and
integrating dynamical systems with emphasis on applications in astrodynamics.

.. currentmodule:: hiten.algorithms.dynamics

base.py
~~~~~~~

The base module provides the core framework for dynamical systems and their interfaces. 
An abstract base class for dynamical systems is provided, as well as a protocol for dynamical systems.

_DynamicalSystem()
^^^^^^^^^^^^^^^^^^

The :class:`_DynamicalSystem` class provides the foundation for implementing
concrete dynamical systems that can be integrated numerically. It defines the
minimal interface required for compatibility with the integrator framework
and includes utilities for state validation and dimension checking.

.. autoclass:: _DynamicalSystem()
   :members:
   :undoc-members:
   :exclude-members: __init__

_DirectedSystem()
^^^^^^^^^^^^^^^^^^

The :class:`_DirectedSystem` class provides a directional wrapper for forward/backward time integration. 
It wraps another dynamical system to enable forward or backward time integration with selective component sign handling. 
Particularly useful for Hamiltonian systems where momentum variables change sign under time reversal.

.. autoclass:: _DirectedSystem()
   :members:
   :undoc-members:
   :exclude-members: __init__, dim

_DynamicalSystemProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_DynamicalSystemProtocol` class provides the protocol for the minimal interface for dynamical systems.
It defines the required attributes that any dynamical system must implement to be compatible with the integrator framework.
It uses structural typing to allow duck typing while maintaining type safety.

.. autoclass:: _DynamicalSystemProtocol()
   :members:
   :undoc-members:
   :exclude-members: __init__

_propagate_dynsys()
^^^^^^^^^^^^^^^^^^^

The :func:`_propagate_dynsys` function provides a generic propagation function for dynamical systems.
It handles state validation, directional wrapping, and delegation to various integration backends.
Supports multiple numerical methods with consistent interface.

.. autofunction:: _propagate_dynsys()

_validate_initial_state()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_validate_initial_state` function provides a function to validate the initial state of a dynamical system.
It validates the dimension of the state vector and raises an error if it does not match the expected dimension.

.. autofunction:: _validate_initial_state()

rhs.py
~~~~~~

The rhs module provides the base classes for dynamical systems and their interfaces. 
An abstract base class for dynamical systems is provided, as well as a protocol for dynamical systems.

.. currentmodule:: hiten.algorithms.dynamics.rhs

_RHSSystem()
^^^^^^^^^^^^^

The :class:`_RHSSystem` class provides the foundation for implementing
concrete dynamical systems that can be integrated numerically. It defines the
minimal interface required for compatibility with the integrator framework
and includes utilities for state validation and dimension checking.


.. autoclass:: _RHSSystem()
   :members:
   :undoc-members:
   :exclude-members: __init__, dim

The class automatically compiles the RHS function using Numba if it is not already a compiled Numba dispatcher.

create_rhs_system()
^^^^^^^^^^^^^^^^^^^

The :func:`create_rhs_system` function provides a factory function for creating RHS-systems.

.. autofunction:: create_rhs_system()

hamiltonian.py
~~~~~~~~~~~~~~

The hamiltonian module provides polynomial Hamiltonian systems for center manifold dynamics.

.. currentmodule:: hiten.algorithms.dynamics.hamiltonian

_HamiltonianSystemProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_HamiltonianSystemProtocol` class extends the base dynamical system protocol with Hamiltonian-specific methods required by symplectic integrators.

.. autoclass:: _HamiltonianSystemProtocol()
   :members: n_dof, dH_dQ, dH_dP, poly_H
   :exclude-members: __init__

_HamiltonianSystem()
^^^^^^^^^^^^^^^^^^^^

The :class:`_HamiltonianSystem` class implements a polynomial Hamiltonian system for numerical integration.

.. autoclass:: _HamiltonianSystem()
   :members: n_dof, jac_H, clmo_H, rhs, clmo, dH_dQ, dH_dP, poly_H, _validate_coordinates, _validate_polynomial_data
   :exclude-members: __init__

create_hamiltonian_system()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`create_hamiltonian_system` function creates polynomial Hamiltonian system from coefficient data.

.. autofunction:: create_hamiltonian_system()

rtbp.py
~~~~~~~

The rtbp module provides Circular Restricted Three-Body Problem (CR3BP) dynamics implementation.

.. currentmodule:: hiten.algorithms.dynamics.rtbp

_RTBPRHS()
^^^^^^^^^^

The :class:`_RTBPRHS` class defines the Circular Restricted Three-Body Problem equations of motion.

.. autoclass:: _RTBPRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

_VarEqRHS()
^^^^^^^^^^^

The :class:`_VarEqRHS` class provides the CR3BP variational equations for state transition matrix propagation.

.. autoclass:: _VarEqRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

_JacobianRHS()
^^^^^^^^^^^^^^

The :class:`_JacobianRHS` class provides a dynamical system for CR3BP Jacobian matrix evaluation.

.. autoclass:: _JacobianRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

rtbp_dynsys()
^^^^^^^^^^^^^

The :func:`rtbp_dynsys` function creates CR3BP dynamical system.

.. autofunction:: rtbp_dynsys()

variational_dynsys()
^^^^^^^^^^^^^^^^^^^^

The :func:`variational_dynsys` function creates CR3BP variational equations system.

.. autofunction:: variational_dynsys()

jacobian_dynsys()
^^^^^^^^^^^^^^^^^

The :func:`jacobian_dynsys` function creates CR3BP Jacobian evaluation system.

.. autofunction:: jacobian_dynsys()

utils
~~~~~

The utils module provides utility functions for dynamical systems analysis.

.. currentmodule:: hiten.algorithms.dynamics.utils

Energy Functions
^^^^^^^^^^^^^^^^

The energy module provides energy and potential functions for the CR3BP.

.. currentmodule:: hiten.algorithms.dynamics.utils.energy

crtbp_energy()
^^^^^^^^^^^^^^

The :func:`crtbp_energy` function computes Hamiltonian energy of a state in the CR3BP.

.. autofunction:: crtbp_energy()

effective_potential()
^^^^^^^^^^^^^^^^^^^^^

The :func:`effective_potential` function computes effective potential in the CR3BP rotating frame.

.. autofunction:: effective_potential()

kinetic_energy()
^^^^^^^^^^^^^^^^

The :func:`kinetic_energy` function computes kinetic energy of a state.

.. autofunction:: kinetic_energy()

gravitational_potential()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`gravitational_potential` function computes gravitational potential energy of test particle.

.. autofunction:: gravitational_potential()

hill_region()
^^^^^^^^^^^^^

The :func:`hill_region` function computes Hill region for zero-velocity surface analysis.

.. autofunction:: hill_region()

energy_to_jacobi()
^^^^^^^^^^^^^^^^^^

The :func:`energy_to_jacobi` function converts Hamiltonian energy to Jacobi constant.

.. autofunction:: energy_to_jacobi()

jacobi_to_energy()
^^^^^^^^^^^^^^^^^^

The :func:`jacobi_to_energy` function converts Jacobi constant to Hamiltonian energy.

.. autofunction:: jacobi_to_energy()

primary_distance()
^^^^^^^^^^^^^^^^^^

The :func:`primary_distance` function computes distance from test particle to primary body.

.. autofunction:: primary_distance()

secondary_distance()
^^^^^^^^^^^^^^^^^^^^^

The :func:`secondary_distance` function computes distance from test particle to secondary body.

.. autofunction:: secondary_distance()

pseudo_potential_at_point()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`pseudo_potential_at_point` function evaluates pseudo-potential Omega at a planar point.

.. autofunction:: pseudo_potential_at_point()

Linear Algebra Functions
^^^^^^^^^^^^^^^^^^^^^^^^

The linalg module provides linear algebra utilities for dynamical systems analysis.

.. currentmodule:: hiten.algorithms.dynamics.utils.linalg

eigenvalue_decomposition()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`eigenvalue_decomposition` function classifies eigenvalue-eigenvector pairs into stable, unstable, and center subspaces.

.. autofunction:: eigenvalue_decomposition()

_stability_indices()
^^^^^^^^^^^^^^^^^^^^

The :func:`_stability_indices` function computes Floquet stability indices for periodic orbit analysis.

.. autofunction:: _stability_indices()

Integrators
-----------

The integrators module provides numerical integration methods for dynamical systems.

.. currentmodule:: hiten.algorithms.integrators

base.py
~~~~~~~

The base module provides abstract interfaces for numerical time integration.

.. currentmodule:: hiten.algorithms.integrators.base

_Solution()
^^^^^^^^^^^

The :class:`_Solution` class stores a discrete solution returned by an integrator.

.. autoclass:: _Solution()
   :members: times, states, derivatives, interpolate
   :exclude-members: __init__, __post_init__

_Integrator()
^^^^^^^^^^^^^

The :class:`_Integrator` class defines the minimal interface that every concrete integrator must satisfy.

.. autoclass:: _Integrator()
   :members: name, options, order, integrate, validate_system, validate_inputs
   :exclude-members: __init__

rk.py
~~~~~

The rk module provides explicit Runge-Kutta integrators used throughout the project.

.. currentmodule:: hiten.algorithms.integrators.rk

_RungeKuttaBase()
^^^^^^^^^^^^^^^^^

The :class:`_RungeKuttaBase` class provides shared functionality of explicit Runge-Kutta schemes.

.. autoclass:: _RungeKuttaBase()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _rk_embedded_step
   :exclude-members: __init__

_FixedStepRK()
^^^^^^^^^^^^^^

The :class:`_FixedStepRK` class implements an explicit fixed-step Runge-Kutta scheme.

.. autoclass:: _FixedStepRK()
   :members: order, integrate
   :exclude-members: __init__

_AdaptiveStepRK()
^^^^^^^^^^^^^^^^^

The :class:`_AdaptiveStepRK` class implements an embedded adaptive Runge-Kutta integrator with PI controller.

.. autoclass:: _AdaptiveStepRK()
   :members: SAFETY, MIN_FACTOR, MAX_FACTOR, order, integrate, _select_initial_step, _update_factor
   :exclude-members: __init__

_RK4()
^^^^^^

The :class:`_RK4` class implements the classical 4th-order Runge-Kutta method.

.. autoclass:: _RK4()
   :members:
   :exclude-members: __init__

_RK6()
^^^^^^

The :class:`_RK6` class implements a 6th-order Runge-Kutta method.

.. autoclass:: _RK6()
   :members:
   :exclude-members: __init__

_RK8()
^^^^^^

The :class:`_RK8` class implements an 8th-order Runge-Kutta method.

.. autoclass:: _RK8()
   :members:
   :exclude-members: __init__

_RK45()
^^^^^^^

The :class:`_RK45` class implements the Dormand-Prince 5(4) adaptive Runge-Kutta method.

.. autoclass:: _RK45()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _err_exp, _E, _rk_embedded_step
   :exclude-members: __init__

_DOP853()
^^^^^^^^^

The :class:`_DOP853` class implements the Dormand-Prince 8(5,3) adaptive Runge-Kutta method.

.. autoclass:: _DOP853()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _err_exp, _E3, _E5, _N_STAGES, _rk_embedded_step, _estimate_error, _estimate_error_norm
   :exclude-members: __init__

RungeKutta()
^^^^^^^^^^^^

The :class:`RungeKutta` class implements a factory class for creating fixed-step Runge-Kutta integrators.

.. autoclass:: RungeKutta()
   :members: _map
   :exclude-members: __init__

AdaptiveRK()
^^^^^^^^^^^^

The :class:`AdaptiveRK` class implements a factory class for creating adaptive step-size Runge-Kutta integrators.

.. autoclass:: AdaptiveRK()
   :members: _map
   :exclude-members: __init__

_build_rhs_wrapper()
^^^^^^^^^^^^^^^^^^^^

The :func:`_build_rhs_wrapper` function returns a JIT friendly wrapper around the system RHS.

.. autofunction:: _build_rhs_wrapper()

symplectic.py
~~~~~~~~~~~~~

The symplectic module provides high-order explicit symplectic integrators for polynomial Hamiltonian systems.

.. currentmodule:: hiten.algorithms.integrators.symplectic

_get_tao_omega()
^^^^^^^^^^^^^^^^

The :func:`_get_tao_omega` function calculates the frequency parameter for the symplectic integrator.

.. autofunction:: _get_tao_omega()

_construct_6d_eval_point()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_construct_6d_eval_point` function constructs a 6D evaluation point from N-DOF position and momentum vectors.

.. autofunction:: _construct_6d_eval_point()

_eval_dH_dQ()
^^^^^^^^^^^^^

The :func:`_eval_dH_dQ` function evaluates derivatives of Hamiltonian with respect to generalized position variables.

.. autofunction:: _eval_dH_dQ()

_eval_dH_dP()
^^^^^^^^^^^^^

The :func:`_eval_dH_dP` function evaluates derivatives of Hamiltonian with respect to generalized momentum variables.

.. autofunction:: _eval_dH_dP()

_phi_H_a_update_poly()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_H_a_update_poly` function applies the first Hamiltonian splitting operator (phi_a) in the symplectic scheme.

.. autofunction:: _phi_H_a_update_poly()

_phi_H_b_update_poly()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_H_b_update_poly` function applies the second Hamiltonian splitting operator (phi_b) in the symplectic scheme.

.. autofunction:: _phi_H_b_update_poly()

_phi_omega_H_c_update_poly()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_omega_H_c_update_poly` function applies the rotation operator (phi_c) in the symplectic scheme.

.. autofunction:: _phi_omega_H_c_update_poly()

_recursive_update_poly()
^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_recursive_update_poly` function applies recursive symplectic update of specified order.

.. autofunction:: _recursive_update_poly()

_integrate_symplectic()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_integrate_symplectic` function integrates Hamilton's equations using a high-order symplectic integrator.

.. autofunction:: _integrate_symplectic()

_ExtendedSymplectic()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ExtendedSymplectic` class implements high-order explicit Tao symplectic integrator for polynomial Hamiltonian systems.

.. autoclass:: _ExtendedSymplectic()
   :members: name, order, c_omega_heuristic, validate_system, integrate
   :exclude-members: __init__

ExtendedSymplectic()
^^^^^^^^^^^^^^^^^^^^

The :class:`ExtendedSymplectic` class implements a factory for extended symplectic integrators.

.. autoclass:: ExtendedSymplectic()
   :members: _map
   :exclude-members: __init__

coefficients
~~~~~~~~~~~~

The coefficients module provides Butcher tableaux for various Runge-Kutta methods.

.. currentmodule:: hiten.algorithms.integrators.coefficients

DOP853 Coefficients
^^^^^^^^^^^^^^^^^^^

The DOP853 module provides the Butcher tableau for the Dormand-Prince 8(5,3) Runge-Kutta method.

.. currentmodule:: hiten.algorithms.integrators.coefficients.dop853

.. autodata:: N_STAGES
.. autodata:: N_STAGES_EXTENDED
.. autodata:: INTERPOLATOR_POWER
.. autodata:: C
.. autodata:: A
.. autodata:: B
.. autodata:: E3
.. autodata:: E5
.. autodata:: D

RK4 Coefficients
^^^^^^^^^^^^^^^^

The RK4 module provides the Butcher tableau for the classical fourth-order Runge-Kutta method.

.. currentmodule:: hiten.algorithms.integrators.coefficients.rk4

.. autodata:: A
.. autodata:: B
.. autodata:: C

RK45 Coefficients
^^^^^^^^^^^^^^^^^

The RK45 module provides the Butcher tableau for the Dormand-Prince 4(5) Runge-Kutta method.

.. currentmodule:: hiten.algorithms.integrators.coefficients.rk45

.. autodata:: C
.. autodata:: A
.. autodata:: B_HIGH
.. autodata:: B_LOW
.. autodata:: B
.. autodata:: E
.. autodata:: P

RK6 Coefficients
^^^^^^^^^^^^^^^^

The RK6 module provides the Butcher tableau for a sixth-order Runge-Kutta method.

.. currentmodule:: hiten.algorithms.integrators.coefficients.rk6

.. autodata:: A
.. autodata:: B
.. autodata:: C

RK8 Coefficients
^^^^^^^^^^^^^^^^

The RK8 module provides the Butcher tableau for an eighth-order Runge-Kutta method.

.. currentmodule:: hiten.algorithms.integrators.coefficients.rk8

.. autodata:: A
.. autodata:: B
.. autodata:: C

Continuation
------------

The continuation module provides methods for numerical continuation of solutions.

.. currentmodule:: hiten.algorithms.continuation

Corrector
---------

The corrector module provides methods for correcting approximate solutions.

.. currentmodule:: hiten.algorithms.corrector

Bifurcation
-----------

The bifurcation module provides methods for detecting and analyzing bifurcations.

.. currentmodule:: hiten.algorithms.bifurcation

Connections
-----------

The connections module provides functionality for discovering ballistic and impulsive 
transfers between manifolds in the CR3BP.

.. currentmodule:: hiten.algorithms.connections

Fourier
-------

The Fourier module provides methods for Fourier analysis of periodic solutions.

.. currentmodule:: hiten.algorithms.fourier

Hamiltonian
-----------

The Hamiltonian module provides methods for Hamiltonian systems analysis.

.. currentmodule:: hiten.algorithms.hamiltonian

Poincare Maps
-------------

The Poincare module provides methods for Poincare maps and center manifolds.

.. currentmodule:: hiten.algorithms.poincare

Polynomial
----------

The polynomial module provides methods for polynomial operations in Hamiltonian systems.

.. currentmodule:: hiten.algorithms.polynomial

Tori
----

The tori module provides methods for invariant tori analysis.

.. currentmodule:: hiten.algorithms.tori

Utilities
---------

The utils module provides utility functions for algorithms.

.. currentmodule:: hiten.algorithms.utils
