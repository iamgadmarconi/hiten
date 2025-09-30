Corrector Types
===============

The types module provides data structures and type definitions for the correction framework.

.. currentmodule:: hiten.algorithms.corrector.types

Correction Result
-----------------

CorrectionResult()
^^^^^^^^^^^^^^^^^^

Standardized result for a backend correction run.

.. autoclass:: CorrectionResult
   :members:
   :undoc-members:
   :exclude-members: __init__

Orbit Correction Result
-----------------------

OrbitCorrectionResult()
^^^^^^^^^^^^^^^^^^^^^^^

Result for an orbit correction run.

.. autoclass:: OrbitCorrectionResult
   :members:
   :undoc-members:
   :exclude-members: __init__

Correction Problem
------------------

_CorrectionProblem()
^^^^^^^^^^^^^^^^^^^^

Defines the inputs for a backend correction run.

.. autoclass:: _CorrectionProblem
   :members:
   :undoc-members:
   :exclude-members: __init__

Orbit Correction Problem
------------------------

_OrbitCorrectionProblem()
^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines the inputs for a backend orbit correction run.

.. autoclass:: _OrbitCorrectionProblem
   :members:
   :undoc-members:
   :exclude-members: __init__

Type Aliases
------------

ResidualFn
^^^^^^^^^^

Type alias for residual function signatures.

.. data:: ResidualFn

JacobianFn
^^^^^^^^^^

Type alias for Jacobian function signatures.

.. data:: JacobianFn

NormFn
^^^^^^

Type alias for norm function signatures.

.. data:: NormFn

StepperFactory
^^^^^^^^^^^^^^

Type alias for stepper factory function signatures.

.. data:: StepperFactory
