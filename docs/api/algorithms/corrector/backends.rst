Backend Algorithms
==================

The backends module provides core correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.backends

Base Backend
------------

.. currentmodule:: hiten.algorithms.corrector.backends.base

_CorrectorBackend()
^^^^^^^^^^^^^^^^^^^

The :class:`_CorrectorBackend` class defines an abstract base class for iterative correction algorithms.

.. autoclass:: _CorrectorBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

Newton-Raphson Backend
----------------------

.. currentmodule:: hiten.algorithms.corrector.backends.newton

_NewtonBackend()
^^^^^^^^^^^^^^^^

The :class:`_NewtonBackend` class implements the Newton-Raphson algorithm with robust linear algebra and step control.

.. autoclass:: _NewtonBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__
