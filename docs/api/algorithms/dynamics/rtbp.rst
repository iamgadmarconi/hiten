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

jacobian_dynsys()
^^^^^^^^^^^^^^^^^

The :func:`jacobian_dynsys` function creates CR3BP Jacobian evaluation system.

.. autofunction:: jacobian_dynsys()
