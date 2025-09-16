Bifurcation Module
==================

The bifurcation module provides methods for detecting and analyzing bifurcations in the circular restricted three-body problem.

.. currentmodule:: hiten.algorithms.bifurcation

Base Module
~~~~~~~~~~~

The base module provides the core bifurcation analysis framework.

.. currentmodule:: hiten.algorithms.bifurcation.base

This module is reserved for future use and will contain the core bifurcation analysis framework.

Analysis Module
~~~~~~~~~~~~~~~

The analysis module provides bifurcation analysis methods.

.. currentmodule:: hiten.algorithms.bifurcation.analysis

This module is reserved for future use and will contain bifurcation analysis methods.

Transformation Functions
~~~~~~~~~~~~~~~~~~~~~~~~

The transforms module provides polynomial transformations for Birkhoff normal form analysis.

.. currentmodule:: hiten.algorithms.bifurcation.transforms

_nf2aa_ee()
^^^^^^^^^^^

The :func:`_nf2aa_ee` function converts Birkhoff normal form polynomial to action-angle form for elliptic-elliptic libration points (L4, L5).

.. autofunction:: _nf2aa_ee()

_nf2aa_sc()
^^^^^^^^^^^

The :func:`_nf2aa_sc` function converts Birkhoff normal form polynomial to action-angle form for saddle-center libration points (L1, L2, L3).

.. autofunction:: _nf2aa_sc()

