Continuation Stepping Strategies
=================================

The stepping module provides concrete implementations of stepping strategies used in continuation algorithms.

.. currentmodule:: hiten.algorithms.continuation.stepping

Base Step Interface
-------------------

_ContinuationStepBase()
^^^^^^^^^^^^^^^^^^^^^^^

Define the protocol for continuation stepping strategies.

.. autoclass:: _ContinuationStepBase
   :members:
   :undoc-members:
   :exclude-members: __init__

Plain Stepping
--------------

_ContinuationPlainStep()
^^^^^^^^^^^^^^^^^^^^^^^^

Implement a simple stepping strategy using a provided predictor function.

.. autoclass:: _ContinuationPlainStep
   :members:
   :undoc-members:
   :exclude-members: __init__

Natural Parameter Stepping
--------------------------

_NaturalParameterStep()
^^^^^^^^^^^^^^^^^^^^^^^

Implement a natural parameter stepping strategy with user-supplied predictor.

.. autoclass:: _NaturalParameterStep
   :members:
   :undoc-members:
   :exclude-members: __init__

Secant Stepping
---------------

_SecantStep()
^^^^^^^^^^^^^

Stateless secant step using an external tangent provider.

.. autoclass:: _SecantStep
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Functions
-----------------

make_natural_stepper()
^^^^^^^^^^^^^^^^^^^^^^

Factory for a natural-parameter stepper.

.. autofunction:: make_natural_stepper

make_secant_stepper()
^^^^^^^^^^^^^^^^^^^^^

Factory for a secant stepper using an external tangent provider.

.. autofunction:: make_secant_stepper
