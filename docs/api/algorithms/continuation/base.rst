Continuation Base
=================

The base module provides user-facing facades for continuation workflows.

.. currentmodule:: hiten.algorithms.continuation.base

Continuation Pipeline
---------------------

ContinuationPipeline()
^^^^^^^^^^^^^^^^^^^^^^

Facade for natural-parameter continuation varying selected state components. Users supply an engine (DI). Use `ContinuationPipeline.with_default_engine()` to construct a default engine wired with the generic predict-correct backend and the periodic-orbit interface.

.. autoclass:: ContinuationPipeline
   :members:
   :undoc-members:
   :exclude-members: __init__
