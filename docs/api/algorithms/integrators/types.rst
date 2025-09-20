Types and Data Structures
=========================

The types module provides data structures and result containers for numerical integration.

.. currentmodule:: hiten.algorithms.integrators.types

EventResult()
^^^^^^^^^^^^^

The :class:`EventResult` class is a container for event detection results.

.. autoclass:: EventResult()
   :members:
   :exclude-members: __init__

_Solution()
^^^^^^^^^^^

The :class:`_Solution` class stores a discrete solution returned by an integrator.

.. autoclass:: _Solution()
   :members: times, states, derivatives, interpolate
   :exclude-members: __init__, __post_init__
