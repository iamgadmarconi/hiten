Center Module
=============

The center module provides high-level utilities for computing a polynomial normal form of the centre manifold around a collinear libration point of the spatial circular restricted three body problem (CRTBP).

All heavy algebra is performed symbolically on packed coefficient arrays. Only NumPy is used so the implementation is portable and fast.

.. currentmodule:: hiten.system.center

CenterManifold
--------------

Centre manifold normal-form builder for libration points.

.. autoclass:: CenterManifold
   :members:
   :undoc-members:
   :exclude-members: __init__
