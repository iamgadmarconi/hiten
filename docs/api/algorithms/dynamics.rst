Dynamics Module
===============

The dynamics module provides a comprehensive framework for defining, analyzing, and
integrating dynamical systems with emphasis on applications in astrodynamics.

.. currentmodule:: hiten.algorithms.dynamics

base.py
~~~~~~~

The base module provides the core framework for dynamical systems and their interfaces. 
An abstract base class for dynamical systems is provided, as well as a protocol for dynamical systems.

.. currentmodule:: hiten.algorithms.dynamics.base

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
