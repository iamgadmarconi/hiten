Step Control Strategies
=======================

The stepping module provides step-size control interfaces for Newton-type correction algorithms.

.. toctree::
   :maxdepth: 2

   base
   plain
   armijo

base.py
^^^^^^^

.. currentmodule:: hiten.algorithms.corrector.stepping.base

_CorrectorSteppingBase()
^^^^^^^^^^^^^^^

The :class:`_CorrectorSteppingBase` class provides an abstract base class for step-size control strategy interfaces.

.. autoclass:: _CorrectorSteppingBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

plain.py
^^^^^^^^

.. currentmodule:: hiten.algorithms.corrector.stepping.plain

_PlainStep()
^^^^^^^^^^^^

The :class:`_PlainStep` class provides a step interface for plain Newton updates with safeguards.

.. autoclass:: _PlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

make_plain_stepper()
^^^^^^^^^^^^^^^^^^^^

Factory function for creating plain Newton steppers.

.. autofunction:: make_plain_stepper()

armijo.py
^^^^^^^^^

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

make_armijo_stepper()
^^^^^^^^^^^^^^^^^^^^^

Factory function for creating Armijo line search steppers.

.. autofunction:: make_armijo_stepper()

_default_norm()
^^^^^^^^^^^^^^^

The :func:`_default_norm` function computes L2 norm of residual vector.

.. autofunction:: _default_norm()

_infinity_norm()
^^^^^^^^^^^^^^^^

The :func:`_infinity_norm` function computes infinity norm of residual vector.

.. autofunction:: _infinity_norm()
