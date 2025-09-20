Corrector Module
================

The corrector module provides robust iterative correction algorithms for solving nonlinear systems in dynamical systems analysis.

.. currentmodule:: hiten.algorithms.corrector

config.py
~~~~~~~~~

The config module provides configuration classes for iterative correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.config

_BaseCorrectionConfig()
^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_BaseCorrectionConfig` class defines a base configuration class for correction algorithm parameters.

.. autoclass:: _BaseCorrectionConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

_OrbitCorrectionConfig()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_OrbitCorrectionConfig` class defines a configuration for periodic orbit correction.

.. autoclass:: _OrbitCorrectionConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

_LineSearchConfig()
^^^^^^^^^^^^^^^^^^^^

The :class:`_LineSearchConfig` class defines configuration parameters for Armijo line search.

.. autoclass:: _LineSearchConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

base.py
~~~~~~~

The base module provides the core corrector framework and abstract base class for iterative correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.base

_CorrectorBackend()
^^^^^^^^^^^^

The :class:`_CorrectorBackend` class defines an abstract base class for iterative correction algorithms.

.. autoclass:: _CorrectorBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

correctors.py
~~~~~~~~~~~~~

The correctors module provides concrete corrector implementations for specific problem domains.

.. currentmodule:: hiten.algorithms.corrector.correctors

_NewtonOrbitCorrector()
^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_NewtonOrbitCorrector` class implements a Newton-Raphson corrector for periodic orbits.

.. autoclass:: _NewtonOrbitCorrector()
   :members:
   :undoc-members:
   :exclude-members: __init__

interfaces.py
~~~~~~~~~~~~~

The interfaces module provides domain-specific interfaces for correction algorithms.

.. currentmodule:: hiten.algorithms.corrector.interfaces

_PeriodicOrbitCorrectorInterface()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_PeriodicOrbitCorrectorInterface` class provides an interface for periodic orbit differential correction.

.. autoclass:: _PeriodicOrbitCorrectorInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

_InvariantToriCorrectorInterface()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_InvariantToriCorrectorInterface` class provides an interface for invariant tori correction (placeholder).

.. autoclass:: _InvariantToriCorrectorInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

line.py
~~~~~~~

The line module provides line search implementations for robust Newton-type methods.

.. currentmodule:: hiten.algorithms.corrector.line

_ArmijoLineSearch()
^^^^^^^^^^^^^^^^^^^

The :class:`_ArmijoLineSearch` class implements Armijo line search with backtracking for Newton methods.

.. autoclass:: _ArmijoLineSearch()
   :members:
   :undoc-members:
   :exclude-members: __init__

_default_norm()
^^^^^^^^^^^^^^^

The :func:`_default_norm` function computes L2 norm of residual vector.

.. autofunction:: _default_norm()

_infinity_norm()
^^^^^^^^^^^^^^^^

The :func:`_infinity_norm` function computes infinity norm of residual vector.

.. autofunction:: _infinity_norm()

newton.py
~~~~~~~~~

The newton module provides Newton-Raphson correction algorithm with robust linear algebra.

.. currentmodule:: hiten.algorithms.corrector.newton

_NewtonBackend()
^^^^^^^^^^^^^

The :class:`_NewtonBackend` class implements the Newton-Raphson algorithm with robust linear algebra and step control.

.. autoclass:: _NewtonBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

_step_interface.py
~~~~~~~~~~~~~~~~~~

The step interface module provides step-size control interfaces for Newton-type correction algorithms.

.. currentmodule:: hiten.algorithms.corrector._step_interface

_Stepper()
^^^^^^^^^^

The :class:`_Stepper` class defines the protocol for step transformation functions in Newton-type methods.

.. autoclass:: _Stepper()
   :members:
   :undoc-members:
   :exclude-members: __init__

_StepInterface()
^^^^^^^^^^^^^^^^

The :class:`_StepInterface` class provides an abstract base class for step-size control strategy interfaces.

.. autoclass:: _StepInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

_PlainStepInterface()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_PlainStepInterface` class provides a step interface for plain Newton updates with safeguards.

.. autoclass:: _PlainStepInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

_ArmijoStepInterface()
^^^^^^^^^^^^^^^^^^^^^^

The :class:`_ArmijoStepInterface` class provides a step interface with Armijo line search for robust convergence.

.. autoclass:: _ArmijoStepInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__
