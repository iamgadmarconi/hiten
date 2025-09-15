Center Manifold Normal Forms
=============================

The center module provides partial normal form computations for center manifold analysis.

.. currentmodule:: hiten.algorithms.hamiltonian.center

_lie_transform()
^^^^^^^^^^^^^^^^

The :func:`_lie_transform` function performs Lie series normalization of polynomial Hamiltonian for center manifold.

.. autofunction:: _lie_transform()

_get_homogeneous_terms()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_get_homogeneous_terms` function extracts homogeneous terms of specified degree from polynomial.

.. autofunction:: _get_homogeneous_terms()

_select_terms_for_elimination()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_select_terms_for_elimination` function identifies non-resonant terms for elimination in Lie normalization.

.. autofunction:: _select_terms_for_elimination()

_lie_expansion()
^^^^^^^^^^^^^^^^

The :func:`_lie_expansion` function computes coordinate transformations using Lie series expansions.

.. autofunction:: _lie_expansion()

_apply_coord_transform()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_apply_coord_transform` function applies Lie series transformation to single coordinate polynomial.

.. autofunction:: _apply_coord_transform()

_evaluate_transform()
^^^^^^^^^^^^^^^^^^^^^

The :func:`_evaluate_transform` function evaluates coordinate transformation at specific center manifold point.

.. autofunction:: _evaluate_transform()

_zero_q1p1()
^^^^^^^^^^^^

The :func:`_zero_q1p1` function restricts polynomial expansions to center manifold subspace.

.. autofunction:: _zero_q1p1()
