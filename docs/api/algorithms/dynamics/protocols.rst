Dynamical System Protocols
==========================

The protocols module provides abstract interfaces for dynamical systems using Python protocols.

.. currentmodule:: hiten.algorithms.dynamics.protocols

_DynamicalSystemProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_DynamicalSystemProtocol` class defines the minimal interface that any dynamical system must implement to be compatible with the integrator framework.

.. autoclass:: _DynamicalSystemProtocol()
   :members:
   :exclude-members: __init__

_HamiltonianSystemProtocol()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_HamiltonianSystemProtocol` class extends the base dynamical system protocol with Hamiltonian-specific methods required by symplectic integrators.

.. autoclass:: _HamiltonianSystemProtocol()
   :members: n_dof, dH_dQ, dH_dP, poly_H, rhs_params
   :exclude-members: __init__
