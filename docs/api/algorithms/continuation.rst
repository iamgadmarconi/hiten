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
   :exclude-members: __init__

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

.. currentmodule:: hiten.algorithms.continuation.strategies

Algorithm Strategies
^^^^^^^^^^^^^^^^^^^^

The algorithm strategies provide different continuation approaches.

.. currentmodule:: hiten.algorithms.continuation.strategies._algorithms

_NaturalParameter()
^^^^^^^^^^^^^^^^^^^

The :class:`_NaturalParameter` class provides an abstract base class for natural parameter continuation algorithms.

.. autoclass:: _NaturalParameter()
   :members:
   :undoc-members:
   :exclude-members: __init__

_SecantArcLength()
^^^^^^^^^^^^^^^^^^

The :class:`_SecantArcLength` class provides an abstract base class for pseudo-arclength continuation algorithms.

.. autoclass:: _SecantArcLength()
   :members:
   :undoc-members:
   :exclude-members: __init__

Stepping Strategies
^^^^^^^^^^^^^^^^^^^

The stepping strategies provide concrete implementations of prediction methods.

.. currentmodule:: hiten.algorithms.continuation.strategies._stepping

_StepStrategy()
^^^^^^^^^^^^^^^

The :class:`_StepStrategy` class defines an extended protocol for stepping strategies with event hooks.

.. autoclass:: _StepStrategy()
   :members:
   :undoc-members:
   :exclude-members: __init__

_NaturalParameterStep()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_NaturalParameterStep` class implements a natural parameter stepping strategy with user-supplied predictor.

.. autoclass:: _NaturalParameterStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

_SecantStep()
^^^^^^^^^^^^^

The :class:`_SecantStep` class implements a secant-based stepping strategy for pseudo-arclength continuation.

.. autoclass:: _SecantStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

Interface Protocols
^^^^^^^^^^^^^^^^^^^

The interface protocols define the required interfaces for stepping strategies.

.. currentmodule:: hiten.algorithms.continuation.strategies._step_interface

_ContinuationStep()
^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationStep` class defines the protocol for continuation stepping strategies.

.. autoclass:: _ContinuationStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

_PlainStep()
^^^^^^^^^^^^

The :class:`_PlainStep` class implements a simple stepping strategy using a provided predictor function.

.. autoclass:: _PlainStep()
   :members:
   :undoc-members:
   :exclude-members: __init__

events.py
~~~~~~~~~

The events module provides event detection for continuation (reserved for future use).

.. currentmodule:: hiten.algorithms.continuation.events
