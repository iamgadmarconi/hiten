Continuation Algorithm Strategies
==================================

The algorithms module provides abstract base classes for different continuation algorithm strategies. Each strategy defines a particular approach to parameter continuation with specialized stepping logic, stopping criteria, and parameter handling.

.. currentmodule:: hiten.algorithms.continuation.strategies.algorithms

_NaturalParameter()
^^^^^^^^^^^^^^^^^^^^

The :class:`_NaturalParameter` class provides an abstract base class for natural parameter continuation algorithms.

.. autoclass:: _NaturalParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__

_SecantArcLength()
^^^^^^^^^^^^^^^^^^

The :class:`_SecantArcLength` class provides an abstract base class for pseudo-arclength continuation algorithms.

.. autoclass:: _SecantArcLength()
   :members:
   :undoc-members:
   :exclude-members: __init__
