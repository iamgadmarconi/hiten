Base Integrators
================

The base module provides abstract interfaces for numerical time integration.

.. currentmodule:: hiten.algorithms.integrators.base

_Integrator()
^^^^^^^^^^^^^

The :class:`_Integrator` class defines the minimal interface that every concrete integrator must satisfy.

.. autoclass:: _Integrator()
   :members: name, options, order, integrate, validate_system, validate_inputs, _maybe_constant_solution
   :exclude-members: __init__
