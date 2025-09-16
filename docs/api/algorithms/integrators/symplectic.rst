Symplectic Integrators
=======================

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
