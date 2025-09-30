Exceptions
==========

The exceptions module defines the custom exception hierarchy for the Hiten framework.

.. currentmodule:: hiten.algorithms.types.exceptions

Exception Hierarchy
-------------------

HitenError()
^^^^^^^^^^^^

Base exception for Hiten errors.

.. autoclass:: HitenError
   :members:
   :undoc-members:

ConvergenceError()
^^^^^^^^^^^^^^^^^^

Raised when an algorithm fails to converge.

.. autoclass:: ConvergenceError
   :members:
   :undoc-members:

BackendError()
^^^^^^^^^^^^^^

An error occurred within a Backend algorithm.

.. autoclass:: BackendError
   :members:
   :undoc-members:

EngineError()
^^^^^^^^^^^^^

An error occurred during an Engine's workflow.

.. autoclass:: EngineError
   :members:
   :undoc-members:
