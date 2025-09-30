Continuation Engine
===================

The engine module provides orchestration layers that combine backends and interfaces for complete continuation workflows.

.. currentmodule:: hiten.algorithms.continuation.engine

Base Engine
-----------

_ContinuationEngine()
^^^^^^^^^^^^^^^^^^^^^

Abstract base class for continuation engines.

.. autoclass:: _ContinuationEngine
   :members:
   :undoc-members:
   :exclude-members: __init__

Orbit Continuation Engine
-------------------------

_OrbitContinuationEngine()
^^^^^^^^^^^^^^^^^^^^^^^^^^

Engine orchestrating periodic orbit continuation via backend and interface.

.. autoclass:: _OrbitContinuationEngine
   :members:
   :undoc-members:
   :exclude-members: __init__
