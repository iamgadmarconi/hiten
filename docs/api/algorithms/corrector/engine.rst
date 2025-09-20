Correction Engine
=================

The engine module provides orchestration for correction workflows.

.. toctree::
   :maxdepth: 2

   base
   engine

base.py
^^^^^^^

.. currentmodule:: hiten.algorithms.corrector.engine.base

_CorrectionEngine()
^^^^^^^^^^^^^^^^^^^

The :class:`_CorrectionEngine` class provides an abstract base class for correction engines.

.. autoclass:: _CorrectionEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__

engine.py
^^^^^^^^^

.. currentmodule:: hiten.algorithms.corrector.engine.engine

_OrbitCorrectionEngine()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_OrbitCorrectionEngine` class provides the main engine for orchestrating periodic orbit correction.

.. autoclass:: _OrbitCorrectionEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__
