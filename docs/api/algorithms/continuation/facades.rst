Continuation Facades
====================

The facades module provides user-facing interfaces for continuation workflows. These facades assemble the engine, backend, and interface using dependency injection and provide a simple API to run continuation with domain-friendly inputs.

.. currentmodule:: hiten.algorithms.continuation.facades

State Parameter Continuation
----------------------------

.. currentmodule:: hiten.algorithms.continuation.facades

StateParameter()
^^^^^^^^^^^^^^^^

The :class:`StateParameter` class provides a facade for natural-parameter continuation varying selected state components. Users supply an engine (via dependency injection). The class also provides a factory method to construct a default engine wired with the generic predict-correct backend and the periodic-orbit interface.

.. autoclass:: StateParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__
