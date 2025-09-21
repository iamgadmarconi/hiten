Step Control Strategies
=======================

The stepping module provides step-size control interfaces for Newton-type correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.stepping

Base Step Interface
-------------------

.. currentmodule:: hiten.algorithms.corrector.stepping.base

_CorrectorStepBase()
^^^^^^^^^^^^^^^^^^^^

The :class:`_CorrectorStepBase` class provides an abstract base class for step-size control strategy interfaces.

.. autoclass:: _CorrectorStepBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

Plain Stepping
--------------

.. currentmodule:: hiten.algorithms.corrector.stepping.plain

_CorrectorPlainStep()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_CorrectorPlainStep` class provides a step interface for plain Newton updates with safeguards.

.. autoclass:: _CorrectorPlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

Armijo Line Search
------------------

.. currentmodule:: hiten.algorithms.corrector.stepping.armijo

_ArmijoStep()
^^^^^^^^^^^^^

The :class:`_ArmijoStep` class provides a step interface with Armijo line search for robust convergence.

.. autoclass:: _ArmijoStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

_ArmijoLineSearch()
^^^^^^^^^^^^^^^^^^^

The :class:`_ArmijoLineSearch` class implements Armijo line search with backtracking for Newton methods.

.. autoclass:: _ArmijoLineSearch()
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Functions
-----------------

.. currentmodule:: hiten.algorithms.corrector.stepping

The stepping module also provides factory functions for creating step control strategies:

make_plain_stepper()
^^^^^^^^^^^^^^^^^^^^

Factory function for creating plain Newton steppers.

.. autofunction:: make_plain_stepper

make_armijo_stepper()
^^^^^^^^^^^^^^^^^^^^^

Factory function for creating Armijo line search steppers.

.. autofunction:: make_armijo_stepper

Utility Functions
-----------------

.. currentmodule:: hiten.algorithms.corrector.stepping.armijo

_default_norm()
^^^^^^^^^^^^^^^

The :func:`_default_norm` function computes L2 norm of residual vector.

.. autofunction:: _default_norm

_infinity_norm()
^^^^^^^^^^^^^^^^

The :func:`_infinity_norm` function computes infinity norm of residual vector.

.. autofunction:: _infinity_norm
