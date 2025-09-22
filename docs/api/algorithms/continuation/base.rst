Continuation Base 
=================

The base module provides the base classes for continuation workflows.

.. currentmodule:: hiten.algorithms.continuation.base

State Parameter Continuation
----------------------------

.. currentmodule:: hiten.algorithms.continuation.base

StateParameter()
^^^^^^^^^^^^^^^^

The :class:`StateParameter` class provides a facade for periodic orbit continuation that varies selected state components. It wraps a continuation engine and provides a convenient interface for configuring and running continuation with domain-friendly parameters like state indices and target ranges.

The facade internally creates a default engine with the predict-correct backend and periodic orbit interface. Users can access the underlying engine via dependency injection if needed.

.. autoclass:: StateParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__
