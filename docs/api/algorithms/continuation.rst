Continuation Module
===================

The continuation module provides a comprehensive framework for numerical continuation of solutions in dynamical systems.

.. currentmodule:: hiten.algorithms.continuation

base.py
~~~~~~~

The base module provides the core continuation framework and abstract base class for numerical continuation algorithms.

.. currentmodule:: hiten.algorithms.continuation.base

_ContinuationEngine()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationEngine` class provides the foundation for implementing concrete continuation algorithms that can trace families of solutions in dynamical systems. It defines the minimal interface required for compatibility with the continuation framework and includes utilities for state validation, step size adaptation, and family management.

.. autoclass:: _ContinuationEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__, __repr__

interfaces.py
~~~~~~~~~~~~~

The interfaces module provides domain-specific continuation interfaces for different types of solutions.

.. currentmodule:: hiten.algorithms.continuation.interfaces

_OrbitContinuationConfig()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_OrbitContinuationConfig` class defines configuration parameters for periodic orbit continuation.

.. autoclass:: _OrbitContinuationConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

_PeriodicOrbitContinuationInterface()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_PeriodicOrbitContinuationInterface` class provides an interface for periodic orbit continuation in the CR3BP.

.. autoclass:: _PeriodicOrbitContinuationInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

_InvariantToriContinuationInterface()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_InvariantToriContinuationInterface` class provides an interface for invariant tori continuation (placeholder).

.. autoclass:: _InvariantToriContinuationInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

predictors.py
~~~~~~~~~~~~~

The predictors module provides concrete implementations of continuation algorithms for periodic orbits.

.. currentmodule:: hiten.algorithms.continuation.predictors

_StateParameter()
^^^^^^^^^^^^^^^^^

The :class:`_StateParameter` class implements natural parameter continuation varying initial state components.

.. autoclass:: _StateParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__

_FixedPeriod()
^^^^^^^^^^^^^^

The :class:`_FixedPeriod` class provides a placeholder for fixed-period continuation for periodic orbits.

.. autoclass:: _FixedPeriod()
   :members:
   :undoc-members:
   :exclude-members: __init__

_EnergyLevel()
^^^^^^^^^^^^^^

The :class:`_EnergyLevel` class provides a placeholder for energy-level continuation for periodic orbits.

.. autoclass:: _EnergyLevel()
   :members:
   :undoc-members:
   :exclude-members: __init__

strategies/
~~~~~~~~~~~

The strategies module provides continuation strategies and stepping methods.

.. toctree::
   :maxdepth: 2

   continuation/strategies/index

events.py
~~~~~~~~~

The events module provides event detection for continuation (reserved for future use).

.. currentmodule:: hiten.algorithms.continuation.events
