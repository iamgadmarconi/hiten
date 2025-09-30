Connections Backends
====================

The backends module provides backend routines for discovering connections between synodic sections in CR3BP.

.. currentmodule:: hiten.algorithms.connections.backends

Backend Classes
---------------

.. autoclass:: _ConnectionsBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

Functions
---------

.. autofunction:: _pair_counts()

.. autofunction:: _exclusive_prefix_sum()

.. autofunction:: _radpair2d()

.. autofunction:: _radius_pairs_2d()

.. autofunction:: _nearest_neighbor_2d_numba()

.. autofunction:: _nearest_neighbor_2d()

.. autofunction:: _closest_points_on_segments_2d()

.. autofunction:: _refine_pairs_on_section()
