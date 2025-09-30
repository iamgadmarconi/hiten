Symplectic Integrators
=======================

The symplectic module provides high-order explicit symplectic integrators for polynomial Hamiltonian systems.

.. currentmodule:: hiten.algorithms.integrators.symplectic

Main Classes
------------

.. autoclass:: _ExtendedSymplectic()
   :members:
   :undoc-members:
   :exclude-members: __init__

Factory Classes
---------------

.. autoclass:: ExtendedSymplectic()
   :members:
   :undoc-members:
   :exclude-members: __init__

Functions
---------

.. autofunction:: _get_tao_omega()

.. autofunction:: _construct_6d_eval_point()

.. autofunction:: _eval_dH_dQ()

.. autofunction:: _eval_dH_dP()

.. autofunction:: _phi_H_a_update_poly()

.. autofunction:: _phi_H_b_update_poly()

.. autofunction:: _phi_omega_H_c_update_poly()

.. autofunction:: _recursive_update_poly()

.. autofunction:: _integrate_symplectic()

Event Functions
---------------

.. autofunction:: _hermite_refine_event_symplectic()

.. autofunction:: _integrate_symplectic_until_event()
