Base Dynamical Systems
======================

The base module provides the core framework for dynamical systems.

.. currentmodule:: hiten.algorithms.dynamics.base

_DynamicalSystem()
^^^^^^^^^^^^^^^^^^

The :class:`_DynamicalSystem` class provides the abstract base class for implementing concrete dynamical systems that can be integrated numerically. It defines the minimal interface required for compatibility with the integrator framework and includes utilities for state validation and dimension checking.

.. autoclass:: _DynamicalSystem()
   :members: dim, rhs, validate_state, _compile_rhs_function
   :exclude-members: __init__

_DirectedSystem()
^^^^^^^^^^^^^^^^^^

The :class:`_DirectedSystem` class provides a directional wrapper for forward/backward time integration. It wraps another dynamical system to enable forward or backward time integration with selective component sign handling. Particularly useful for Hamiltonian systems where momentum variables change sign under time reversal.

.. autoclass:: _DirectedSystem()
   :members: _fwd, _flip_idx, _flip_idx_norm, _build_rhs_impl
   :exclude-members: __init__, dim, __repr__, __getattr__

_propagate_dynsys()
^^^^^^^^^^^^^^^^^^^

The :func:`_propagate_dynsys` function provides a generic propagation function for dynamical systems. It handles state validation, directional wrapping, and delegation to various integration backends. Supports multiple numerical methods with consistent interface.

.. autofunction:: _propagate_dynsys()

_validate_initial_state()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_validate_initial_state` function validates and normalizes initial state vectors for dynamical systems.

.. autofunction:: _validate_initial_state()
