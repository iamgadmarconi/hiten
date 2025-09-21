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

The :class:`_OrbitContinuationEngine` class implements an engine for periodic orbit continuation that orchestrates the predict-instantiate-correct-accept loop using a backend and interface. This engine uses a backend to solve the continuation problem and an interface to build the necessary closures for domain-specific operations.

.. autoclass:: _OrbitContinuationEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__
