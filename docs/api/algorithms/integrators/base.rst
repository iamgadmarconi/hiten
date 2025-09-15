Base Integrators
================

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
