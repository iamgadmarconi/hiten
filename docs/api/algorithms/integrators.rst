Integrators Module
==================

The integrators module provides numerical integration methods for dynamical systems.

.. currentmodule:: hiten.algorithms.integrators

base.py
~~~~~~~

The base module provides abstract interfaces for numerical time integration.

.. currentmodule:: hiten.algorithms.integrators.base

_Solution()
^^^^^^^^^^^

The :class:`_Solution` class stores a discrete solution returned by an integrator.

.. autoclass:: _Solution()
   :members: times, states, derivatives, interpolate
   :exclude-members: __init__, __post_init__

_Integrator()
^^^^^^^^^^^^^

The :class:`_Integrator` class defines the minimal interface that every concrete integrator must satisfy.

.. autoclass:: _Integrator()
   :members: name, options, order, integrate, validate_system, validate_inputs
   :exclude-members: __init__

rk.py
~~~~~

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

symplectic.py
~~~~~~~~~~~~~

The symplectic module provides high-order explicit symplectic integrators for polynomial Hamiltonian systems.

.. currentmodule:: hiten.algorithms.integrators.symplectic

_get_tao_omega()
^^^^^^^^^^^^^^^^

The :func:`_get_tao_omega` function calculates the frequency parameter for the symplectic integrator.

.. autofunction:: _get_tao_omega()

_construct_6d_eval_point()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_construct_6d_eval_point` function constructs a 6D evaluation point from N-DOF position and momentum vectors.

.. autofunction:: _construct_6d_eval_point()

_eval_dH_dQ()
^^^^^^^^^^^^^

The :func:`_eval_dH_dQ` function evaluates derivatives of Hamiltonian with respect to generalized position variables.

.. autofunction:: _eval_dH_dQ()

_eval_dH_dP()
^^^^^^^^^^^^^

The :func:`_eval_dH_dP` function evaluates derivatives of Hamiltonian with respect to generalized momentum variables.

.. autofunction:: _eval_dH_dP()

_phi_H_a_update_poly()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_H_a_update_poly` function applies the first Hamiltonian splitting operator (phi_a) in the symplectic scheme.

.. autofunction:: _phi_H_a_update_poly()

_phi_H_b_update_poly()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_H_b_update_poly` function applies the second Hamiltonian splitting operator (phi_b) in the symplectic scheme.

.. autofunction:: _phi_H_b_update_poly()

_phi_omega_H_c_update_poly()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_phi_omega_H_c_update_poly` function applies the rotation operator (phi_c) in the symplectic scheme.

.. autofunction:: _phi_omega_H_c_update_poly()

_recursive_update_poly()
^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_recursive_update_poly` function applies recursive symplectic update of specified order.

.. autofunction:: _recursive_update_poly()

_integrate_symplectic()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_integrate_symplectic` function integrates Hamilton's equations using a high-order symplectic integrator.

.. autofunction:: _integrate_symplectic()

_ExtendedSymplectic()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ExtendedSymplectic` class implements high-order explicit Tao symplectic integrator for polynomial Hamiltonian systems.

.. autoclass:: _ExtendedSymplectic()
   :members: name, order, c_omega_heuristic, validate_system, integrate
   :exclude-members: __init__

ExtendedSymplectic()
^^^^^^^^^^^^^^^^^^^^

The :class:`ExtendedSymplectic` class implements a factory for extended symplectic integrators.

.. autoclass:: ExtendedSymplectic()
   :members: _map
   :exclude-members: __init__

coefficients/
~~~~~~~~~~~~~

The coefficients module provides Butcher tableaux for various Runge-Kutta methods.

.. currentmodule:: hiten.algorithms.integrators.coefficients
