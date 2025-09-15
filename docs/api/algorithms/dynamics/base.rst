Base Dynamical Systems
======================

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
   :exclude-members: __init__, dim, __repr__, __getattr__

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
