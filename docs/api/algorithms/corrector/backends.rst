Corrector Backends
==================

The backends module provides core correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.backends

Base Backend
------------

_CorrectorBackend()
^^^^^^^^^^^^^^^^^^^

Abstract base class for iterative correction algorithms.

.. autoclass:: _CorrectorBackend
   :members:
   :undoc-members:
   :exclude-members: __init__

Newton-Raphson Backend
----------------------

_NewtonBackend()
^^^^^^^^^^^^^^^^

Implement the Newton-Raphson algorithm with robust linear algebra and step control.

.. autoclass:: _NewtonBackend
   :members:
   :undoc-members:
   :exclude-members: __init__
