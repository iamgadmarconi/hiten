Continuation Interface Protocols
=================================

The interfaces module defines protocol and base classes for continuation stepping strategies. These protocols specify the required interface for all stepping strategies used in continuation algorithms.

.. currentmodule:: hiten.algorithms.continuation.strategies._step_interface

_ContinuationStep()
^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationStep` class defines the protocol for continuation stepping strategies.

.. autoclass:: _ContinuationStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

_PlainStep()
^^^^^^^^^^^^

The :class:`_PlainStep` class implements a simple stepping strategy using a provided predictor function.

.. autoclass:: _PlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__
