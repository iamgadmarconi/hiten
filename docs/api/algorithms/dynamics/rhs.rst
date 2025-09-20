Right-Hand Side Systems
========================

The rhs module provides adapters for wrapping user-defined right-hand side functions into dynamical systems compatible with the integrator framework.

.. currentmodule:: hiten.algorithms.dynamics.rhs

_RHSSystem()
^^^^^^^^^^^^^

The :class:`_RHSSystem` class provides an adapter for generic right-hand side functions. It converts arbitrary Python callables representing ODE systems dy/dt = f(t, y) into objects compatible with the dynamical systems framework. It automatically handles JIT compilation for optimal performance in numerical integrators.

.. autoclass:: _RHSSystem()
   :members: name
   :exclude-members: __init__, dim, __repr__

create_rhs_system()
^^^^^^^^^^^^^^^^^^^

The :func:`create_rhs_system` function provides a factory function for creating RHS systems using a functional interface.

.. autofunction:: create_rhs_system()
