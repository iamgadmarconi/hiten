Continuation Backends
=====================

The backends module provides the core numerical algorithms that drive the continuation process.

.. currentmodule:: hiten.algorithms.continuation.backends

Base Backend
------------

_ContinuationBackend()
^^^^^^^^^^^^^^^^^^^^^^

Abstract base class for continuation backends.

.. autoclass:: _ContinuationBackend
   :members:
   :undoc-members:
   :exclude-members: __init__

Predict-Correct Backend
-----------------------

_PCContinuationBackend()
^^^^^^^^^^^^^^^^^^^^^^^^

Implement a predict-correct continuation backend.

.. autoclass:: _PCContinuationBackend
   :members:
   :undoc-members:
   :exclude-members: __init__
