Continuation Engine
===================

The engine module provides orchestration layers that combine backends and interfaces for complete continuation workflows. These engines handle the high-level coordination of the continuation process.

.. toctree::
   :maxdepth: 2

.. currentmodule:: hiten.algorithms.continuation.engine

Base Engine
-----------

.. currentmodule:: hiten.algorithms.continuation.engine.base

_ContinuationEngine()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationEngine` class provides an abstract base class for continuation engines that defines the interface for solving continuation problems.

.. autoclass:: _ContinuationEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__

Orbit Continuation Engine
-------------------------

.. currentmodule:: hiten.algorithms.continuation.engine.engine

_OrbitContinuationEngine()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_OrbitContinuationEngine` class orchestrates periodic-orbit continuation by delegating domain translations to `_PeriodicOrbitContinuationInterface` and numerical work to `_PCContinuationBackend`.

.. autoclass:: _OrbitContinuationEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__
