Continuation Facades
====================

The facades module provides user-facing interfaces for continuation workflows. These facades wrap the continuation engine and provide a convenient API for domain-specific continuation tasks.

.. currentmodule:: hiten.algorithms.continuation.facades

State Parameter Continuation
----------------------------

.. currentmodule:: hiten.algorithms.continuation.facades

StateParameter()
^^^^^^^^^^^^^^^^

The :class:`StateParameter` class provides a facade for periodic orbit continuation that varies selected state components. It wraps a continuation engine and provides a convenient interface for configuring and running continuation with domain-friendly parameters like state indices and target ranges.

The facade internally creates a default engine with the predict-correct backend and periodic orbit interface. Users can access the underlying engine via dependency injection if needed.

.. autoclass:: StateParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__
