Right-Hand Side Systems
========================

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
