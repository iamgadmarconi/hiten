Restricted Three-Body Problem
==============================

The rtbp module provides Circular Restricted Three-Body Problem (CR3BP) dynamics implementation.

.. currentmodule:: hiten.algorithms.dynamics.rtbp

_RTBPRHS()
^^^^^^^^^^

The :class:`_RTBPRHS` class defines the Circular Restricted Three-Body Problem equations of motion.

.. autoclass:: _RTBPRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

_VarEqRHS()
^^^^^^^^^^^

The :class:`_VarEqRHS` class provides the CR3BP variational equations for state transition matrix propagation.

.. autoclass:: _VarEqRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

_JacobianRHS()
^^^^^^^^^^^^^^

The :class:`_JacobianRHS` class provides a dynamical system for CR3BP Jacobian matrix evaluation.

.. autoclass:: _JacobianRHS()
   :members: mu, name, rhs
   :exclude-members: __init__, dim

rtbp_dynsys()
^^^^^^^^^^^^^

The :func:`rtbp_dynsys` function creates CR3BP dynamical system.

.. autofunction:: rtbp_dynsys()

variational_dynsys()
^^^^^^^^^^^^^^^^^^^^

The :func:`variational_dynsys` function creates CR3BP variational equations system.

.. autofunction:: variational_dynsys()

_compute_stm()
^^^^^^^^^^^^^^

The :func:`_compute_stm` function propagates state transition matrix (STM) along CR3BP trajectory.

.. autofunction:: _compute_stm()

_compute_monodromy()
^^^^^^^^^^^^^^^^^^^^

The :func:`_compute_monodromy` function computes monodromy matrix for periodic CR3BP orbit.

.. autofunction:: _compute_monodromy()

_stability_indices()
^^^^^^^^^^^^^^^^^^^^

The :func:`_stability_indices` function computes Floquet stability indices for periodic orbit analysis.

.. autofunction:: _stability_indices()

_crtbp_accel()
^^^^^^^^^^^^^^

The :func:`_crtbp_accel` function computes CR3BP equations of motion in rotating synodic frame.

.. autofunction:: _crtbp_accel()

_jacobian_crtbp()
^^^^^^^^^^^^^^^^^

The :func:`_jacobian_crtbp` function computes analytical Jacobian matrix of CR3BP equations of motion.

.. autofunction:: _jacobian_crtbp()

_var_equations()
^^^^^^^^^^^^^^^^

The :func:`_var_equations` function computes CR3BP variational equations for state transition matrix propagation.

.. autofunction:: _var_equations()

rtbp_dynsys()
^^^^^^^^^^^^^

The :func:`rtbp_dynsys` function creates CR3BP dynamical system.

.. autofunction:: rtbp_dynsys()

variational_dynsys()
^^^^^^^^^^^^^^^^^^^^

The :func:`variational_dynsys` function creates CR3BP variational equations system.

.. autofunction:: variational_dynsys()

jacobian_dynsys()
^^^^^^^^^^^^^^^^^

The :func:`jacobian_dynsys` function creates CR3BP Jacobian evaluation system.

.. autofunction:: jacobian_dynsys()
