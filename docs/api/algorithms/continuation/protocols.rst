Continuation Protocols
======================

The protocols module defines runtime-checkable protocols that formalize the interfaces for stepping strategies and engines in the continuation architecture.

.. currentmodule:: hiten.algorithms.continuation.protocols

Continuation Step Protocol
--------------------------

.. currentmodule:: hiten.algorithms.continuation.protocols

ContinuationStepProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ContinuationStepProtocol` defines the protocol for continuation step strategies with optional hooks. Implementations generate the next prediction from the last accepted solution and the current step size, and may adapt internal state via hooks.

.. autoclass:: ContinuationStepProtocol()
   :members:
   :undoc-members:
   :exclude-members: __init__

Continuation Engine Protocol
----------------------------

.. currentmodule:: hiten.algorithms.continuation.protocols

ContinuationEngineProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`ContinuationEngineProtocol` defines the protocol for continuation engines. Engines drive the predict-instantiate-correct-accept loop and should return a standardized result object upon completion.

.. autoclass:: ContinuationEngineProtocol()
   :members:
   :undoc-members:
   :exclude-members: __init__
