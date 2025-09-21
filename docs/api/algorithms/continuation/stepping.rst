Continuation Stepping Strategies
=================================

The stepping module provides concrete implementations of stepping strategies used in continuation algorithms. These strategies handle the prediction phase of the continuation process, generating numerical representations of the next solution based on the current solution and step size.

.. toctree::
   :maxdepth: 2

.. currentmodule:: hiten.algorithms.continuation.stepping

Base Step Interface
-------------------

.. currentmodule:: hiten.algorithms.continuation.stepping.base

_ContinuationStepBase()
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationStepBase` class defines the protocol for continuation stepping strategies that specifies the required interface for all stepping strategies used in continuation algorithms.

.. autoclass:: _ContinuationStepBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

Plain Stepping
--------------

.. currentmodule:: hiten.algorithms.continuation.stepping.plain

_ContinuationPlainStep()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationPlainStep` class implements a simple stepping strategy using a provided predictor function. This strategy delegates prediction to a user-provided function and returns the step size unchanged, making it suitable for cases where step adaptation is handled elsewhere or not needed.

.. autoclass:: _ContinuationPlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

Natural Parameter Stepping
--------------------------

.. currentmodule:: hiten.algorithms.continuation.stepping.np

_NaturalParameterStep()
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_NaturalParameterStep` class implements a natural parameter stepping strategy with user-supplied predictor. This class implements a simple stepping strategy for natural parameter continuation that delegates prediction to a user-supplied function and keeps the step size unchanged.

.. autoclass:: _NaturalParameterStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

Secant Stepping
---------------

.. currentmodule:: hiten.algorithms.continuation.stepping.sc

_SecantStep()
^^^^^^^^^^^^^

The :class:`_SecantStep` class implements a stateless secant step using an external tangent provider. The backend owns history and tangent computation, and this stepper simply uses the provided tangent to form a prediction and returns the step hint unchanged.

.. autoclass:: _SecantStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Functions
-----------------

The stepping module also provides factory functions for creating common stepping strategies:

.. autofunction:: make_natural_stepper

.. autofunction:: make_secant_stepper
