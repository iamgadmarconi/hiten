Tori Module
===========

The tori module provides high-level utilities for computing invariant tori in the circular restricted three-body problem.

.. currentmodule:: hiten.algorithms.tori

Base Classes
~~~~~~~~~~~~

The base module provides comprehensive tools for computing 2D invariant tori that bifurcate from periodic orbits.

.. currentmodule:: hiten.algorithms.tori.base

_ToriCorrectionConfig()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ToriCorrectionConfig` class provides configuration container for invariant-torus Newton solves.

.. autoclass:: _ToriCorrectionConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

_Torus()
^^^^^^^^

The :class:`_Torus` class provides immutable representation of a 2-D invariant torus.

.. autoclass:: _Torus()
   :members:
   :undoc-members:
   :exclude-members: __init__

_InvariantTori()
^^^^^^^^^^^^^^^^

The :class:`_InvariantTori` class provides linear approximation of a 2-D invariant torus bifurcating from a centre component of a periodic orbit.

.. autoclass:: _InvariantTori()
   :members:
   :undoc-members:
   :exclude-members: __init__

