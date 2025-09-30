Runge-Kutta Integrators
========================

The rk module provides explicit Runge-Kutta integrators used throughout the project.

.. currentmodule:: hiten.algorithms.integrators.rk

Base Classes
------------

.. autoclass:: _RungeKuttaBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

Fixed-Step Classes
------------------

.. autoclass:: _FixedStepRK()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _RK4()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _RK6()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _RK8()
   :members:
   :undoc-members:
   :exclude-members: __init__

Adaptive Classes
----------------

.. autoclass:: _AdaptiveStepRK()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _RK45()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _DOP853()
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Classes
---------------

.. autoclass:: FixedRK()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: AdaptiveRK()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: RungeKutta()
   :members:
   :undoc-members:
   :exclude-members: __init__

Kernel Functions
----------------

.. autofunction:: _build_rhs_wrapper()

.. autofunction:: rk_embedded_step_jit_kernel()

.. autofunction:: rk_embedded_step_ham_jit_kernel()

.. autofunction:: rk45_step_jit_kernel()

.. autofunction:: rk45_step_ham_jit_kernel()

.. autofunction:: dop853_step_jit_kernel()

.. autofunction:: dop853_step_ham_jit_kernel()

Dense Output Functions
----------------------

.. autofunction:: _hermite_eval_dense()

.. autofunction:: _rk45_build_Q_cache()

.. autofunction:: _rk45_eval_dense()

.. autofunction:: _dop853_build_dense_cache()

.. autofunction:: _dop853_eval_dense()

Event Functions
---------------

.. autofunction:: _hermite_refine_in_step()

.. autofunction:: _rk45_refine_in_step()

.. autofunction:: _dop853_refine_in_step()

.. autofunction:: _dop853_refine_in_step_ham()
