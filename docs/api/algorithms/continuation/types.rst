Continuation Types
==================

The types module defines the standardized result and problem objects used by continuation engines, following the shared architecture used across algorithms.

.. currentmodule:: hiten.algorithms.continuation.types

Continuation Result
-------------------

.. currentmodule:: hiten.algorithms.continuation.types

ContinuationResult()
^^^^^^^^^^^^^^^^^^^^

The :class:`ContinuationResult` class provides a standardized result for a continuation run, including accepted and rejected solution counts, success rate, the family of solutions, parameter values, and iteration counts.

.. autoclass:: ContinuationResult()
   :members:
   :undoc-members:
   :exclude-members: __init__

Continuation Problem
--------------------

.. currentmodule:: hiten.algorithms.continuation.types

_ContinuationProblem()
^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationProblem` class defines the inputs for a continuation run, including the initial solution, parameter getter function, target bounds, step sizes, and various control parameters for the continuation process.

.. autoclass:: _ContinuationProblem()
   :members:
   :undoc-members:
   :exclude-members: __init__
