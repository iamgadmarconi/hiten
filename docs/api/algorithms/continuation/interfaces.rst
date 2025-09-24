Continuation Interfaces
=======================

The interfaces module provides interface classes that adapt the generic continuation engine to specific problem domains in dynamical systems. These interfaces implement the abstract methods required by the continuation framework for particular types of solutions (periodic orbits, invariant tori, etc.).

.. currentmodule:: hiten.algorithms.continuation.interfaces

Periodic Orbit Continuation Interface
--------------------------------------

.. currentmodule:: hiten.algorithms.continuation.interfaces

_PeriodicOrbitContinuationInterface()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_PeriodicOrbitContinuationInterface` class adapts periodic-orbit continuation in the CR3BP by providing methods for solution representation, instantiation, parameter extraction, and prediction.

.. autoclass:: _PeriodicOrbitContinuationInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__
