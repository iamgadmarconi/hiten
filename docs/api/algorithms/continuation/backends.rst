Continuation Backends
=====================

The backends module provides the core numerical algorithms that drive the continuation process. These backends implement the low-level continuation loops and handle the predict-correct-accept cycle.

.. toctree::
   :maxdepth: 2

.. currentmodule:: hiten.algorithms.continuation.backends

Base Backend
------------

.. currentmodule:: hiten.algorithms.continuation.backends.base

_ContinuationBackend()
^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationBackend` class provides an abstract base class for continuation backends that defines the interface for running continuation using purely numerical inputs and callables.

.. autoclass:: _ContinuationBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

Predict-Correct Backend
-----------------------

.. currentmodule:: hiten.algorithms.continuation.backends.pc

_PCContinuationBackend()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_PCContinuationBackend` class implements a predict-correct continuation backend that drives a simple predict-correct-accept loop using user-provided predictor and corrector functions, adapting the step size based on success/failure and stopping when either the member limit is reached or parameters exit the configured bounds.

.. autoclass:: _PCContinuationBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__
