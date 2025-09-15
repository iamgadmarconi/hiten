Runge-Kutta Integrators
========================

The rk module provides explicit Runge-Kutta integrators used throughout the project.

.. currentmodule:: hiten.algorithms.integrators.rk

_RungeKuttaBase()
^^^^^^^^^^^^^^^^^

The :class:`_RungeKuttaBase` class provides shared functionality of explicit Runge-Kutta schemes.

.. autoclass:: _RungeKuttaBase()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _rk_embedded_step
   :exclude-members: __init__

_FixedStepRK()
^^^^^^^^^^^^^^

The :class:`_FixedStepRK` class implements an explicit fixed-step Runge-Kutta scheme.

.. autoclass:: _FixedStepRK()
   :members: order, integrate
   :exclude-members: __init__

_AdaptiveStepRK()
^^^^^^^^^^^^^^^^^

The :class:`_AdaptiveStepRK` class implements an embedded adaptive Runge-Kutta integrator with PI controller.

.. autoclass:: _AdaptiveStepRK()
   :members: SAFETY, MIN_FACTOR, MAX_FACTOR, order, integrate, _select_initial_step, _update_factor
   :exclude-members: __init__

_RK4()
^^^^^^

The :class:`_RK4` class implements the classical 4th-order Runge-Kutta method.

.. autoclass:: _RK4()
   :members:
   :exclude-members: __init__

_RK6()
^^^^^^

The :class:`_RK6` class implements a 6th-order Runge-Kutta method.

.. autoclass:: _RK6()
   :members:
   :exclude-members: __init__

_RK8()
^^^^^^

The :class:`_RK8` class implements an 8th-order Runge-Kutta method.

.. autoclass:: _RK8()
   :members:
   :exclude-members: __init__

_RK45()
^^^^^^^

The :class:`_RK45` class implements the Dormand-Prince 5(4) adaptive Runge-Kutta method.

.. autoclass:: _RK45()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _err_exp, _E, _rk_embedded_step
   :exclude-members: __init__

_DOP853()
^^^^^^^^^

The :class:`_DOP853` class implements the Dormand-Prince 8(5,3) adaptive Runge-Kutta method.

.. autoclass:: _DOP853()
   :members: _A, _B_HIGH, _B_LOW, _C, _p, _err_exp, _E3, _E5, _N_STAGES, _rk_embedded_step, _estimate_error, _estimate_error_norm
   :exclude-members: __init__

RungeKutta()
^^^^^^^^^^^^

The :class:`RungeKutta` class implements a factory class for creating fixed-step Runge-Kutta integrators.

.. autoclass:: RungeKutta()
   :members: _map
   :exclude-members: __init__

AdaptiveRK()
^^^^^^^^^^^^

The :class:`AdaptiveRK` class implements a factory class for creating adaptive step-size Runge-Kutta integrators.

.. autoclass:: AdaptiveRK()
   :members: _map
   :exclude-members: __init__

_build_rhs_wrapper()
^^^^^^^^^^^^^^^^^^^^

The :func:`_build_rhs_wrapper` function returns a JIT friendly wrapper around the system RHS.

.. autofunction:: _build_rhs_wrapper()
