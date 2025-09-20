Types and Data Structures
=========================

The types module provides data structures and type definitions for the correction framework.

.. currentmodule:: hiten.algorithms.corrector.types

CorrectionResult()
^^^^^^^^^^^^^^^^^^

The :class:`CorrectionResult` class provides standardized result for a backend correction run.

.. autoclass:: CorrectionResult()
   :members:
   :undoc-members:
   :exclude-members: __init__

_CorrectionProblem()
^^^^^^^^^^^^^^^^^^^^

The :class:`_CorrectionProblem` class defines the inputs for a backend correction run.

.. autoclass:: _CorrectionProblem()
   :members:
   :undoc-members:
   :exclude-members: __init__

Type Aliases
^^^^^^^^^^^^

The module also provides several type aliases for function signatures:

.. automodule:: hiten.algorithms.corrector.types
   :members: ResidualFn, JacobianFn, NormFn
   :undoc-members:
