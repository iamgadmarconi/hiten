Base Module
===========

The base module provides high-level abstractions for the Circular Restricted Three-Body Problem (CR3BP).

This module bundles the physical information of a binary system, computes the mass parameter mu, instantiates the underlying vector field via :func:`~hiten.algorithms.dynamics.rtbp.rtbp_dynsys`, and pre-computes the five classical Lagrange (libration) points.

.. currentmodule:: hiten.system.base

System
------

The main system class that provides a lightweight wrapper around the CR3BP dynamical system.

.. autoclass:: System
   :members:
   :undoc-members:
   :exclude-members: __init__