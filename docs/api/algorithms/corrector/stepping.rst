Corrector Stepping
==================

The stepping module provides step-size control interfaces for Newton-type correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.stepping

Base Step Interface
-------------------

_CorrectorStepBase()
^^^^^^^^^^^^^^^^^^^^

Abstract base class for step-size control strategy interfaces.

.. autoclass:: _CorrectorStepBase
   :members:
   :undoc-members:
   :exclude-members: __init__

Plain Stepping
--------------

_CorrectorPlainStep()
^^^^^^^^^^^^^^^^^^^^^

Step interface for plain Newton updates with safeguards.

.. autoclass:: _CorrectorPlainStep
   :members:
   :undoc-members:
   :exclude-members: __init__

Armijo Line Search
------------------

_ArmijoStep()
^^^^^^^^^^^^^

Step interface with Armijo line search for robust convergence.

.. autoclass:: _ArmijoStep
   :members:
   :undoc-members:
   :exclude-members: __init__

_ArmijoLineSearch()
^^^^^^^^^^^^^^^^^^^

Implement Armijo line search with backtracking for Newton methods.

.. autoclass:: _ArmijoLineSearch
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Functions
-----------------

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

_default_norm()
^^^^^^^^^^^^^^^

Compute L2 norm of residual vector.

.. autofunction:: _default_norm

_infinity_norm()
^^^^^^^^^^^^^^^^

Compute infinity norm of residual vector.

.. autofunction:: _infinity_norm
