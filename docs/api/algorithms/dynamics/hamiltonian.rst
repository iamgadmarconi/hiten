Hamiltonian Systems
====================

The hamiltonian module provides polynomial Hamiltonian systems for center manifold dynamics.

.. currentmodule:: hiten.algorithms.dynamics.hamiltonian

_HamiltonianSystemProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_HamiltonianSystemProtocol` class extends the base dynamical system protocol with Hamiltonian-specific methods required by symplectic integrators.

.. autoclass:: _HamiltonianSystemProtocol()
   :members: n_dof, dH_dQ, dH_dP, poly_H
   :exclude-members: __init__, __repr__

_HamiltonianSystem()
^^^^^^^^^^^^^^^^^^^^

The :class:`_HamiltonianSystem` class implements a polynomial Hamiltonian system for numerical integration.

.. autoclass:: _HamiltonianSystem()
   :members: n_dof, jac_H, clmo_H, rhs, clmo, dH_dQ, dH_dP, poly_H, _validate_coordinates, _validate_polynomial_data
   :exclude-members: __init__

_hamiltonian_rhs()
^^^^^^^^^^^^^^^^^^

The :func:`_hamiltonian_rhs` function computes Hamilton's equations for polynomial Hamiltonian systems.

.. autofunction:: _hamiltonian_rhs()

create_hamiltonian_system()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`create_hamiltonian_system` function creates polynomial Hamiltonian system from coefficient data.

.. autofunction:: create_hamiltonian_system()
