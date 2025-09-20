Continuation Interface Protocols
=================================

The interfaces module defines protocol and base classes for continuation stepping strategies. These protocols specify the required interface for all stepping strategies used in continuation algorithms.

.. currentmodule:: hiten.algorithms.continuation.strategies._step_interface

_ContinuationStepBase()
^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationStepBase` class defines the protocol for continuation stepping strategies.

.. autoclass:: _ContinuationStepBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

_CorrectorPlainStep()
^^^^^^^^^^^^

The :class:`_CorrectorPlainStep` class implements a simple stepping strategy using a provided predictor function.

.. autoclass:: _CorrectorPlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__
