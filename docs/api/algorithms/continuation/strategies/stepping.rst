Continuation Stepping Strategies
=================================

The stepping module provides concrete implementations of stepping strategies used in continuation algorithms. These strategies handle the prediction phase of the continuation process, generating numerical representations of the next solution based on the current solution and step size.

.. currentmodule:: hiten.algorithms.continuation.strategies._stepping

_StepStrategy()
^^^^^^^^^^^^^^^

The :class:`_StepStrategy` class defines an extended protocol for stepping strategies with event hooks.

.. autoclass:: _StepStrategy()
   :members:
   :undoc-members:
   :exclude-members: __init__

_NaturalParameterStep()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_NaturalParameterStep` class implements a natural parameter stepping strategy with user-supplied predictor.

.. autoclass:: _NaturalParameterStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

_SecantStep()
^^^^^^^^^^^^^

The :class:`_SecantStep` class implements a secant-based stepping strategy for pseudo-arclength continuation.

.. autoclass:: _SecantStep()
   :members:
   :undoc-members:
   :exclude-members: __init__
