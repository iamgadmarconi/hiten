Connections Module
==================

The connections module provides functionality for discovering ballistic and impulsive transfers between manifolds in the CR3BP.

.. currentmodule:: hiten.algorithms.connections

base.py
~~~~~~~

The base module provides the core connections framework and user-facing interface for connection discovery.

.. currentmodule:: hiten.algorithms.connections.base

Connection()
^^^^^^^^^^^^

The :class:`Connection` class provides a user-facing facade for connection discovery and plotting in CR3BP.

.. autoclass:: Connection()
   :members:
   :undoc-members:
   :exclude-members: __init__

backends.py
~~~~~~~~~~~

The backends module provides backend routines for discovering connections between synodic sections in CR3BP.

.. currentmodule:: hiten.algorithms.connections.backends

_ConnectionsBackend()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ConnectionsBackend` class encapsulates matching/refinement and Delta-V computation for connections.

.. autoclass:: _ConnectionsBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

_pair_counts()
^^^^^^^^^^^^^^

The :func:`_pair_counts` function returns for each query point the number of reference points within radius^2.

.. autofunction:: _pair_counts()

_exclusive_prefix_sum()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_exclusive_prefix_sum` function computes exclusive prefix sum of an integer array.

.. autofunction:: _exclusive_prefix_sum()

_radpair2d()
^^^^^^^^^^^^

The :func:`_radpair2d` function finds all pairs (i,j) where distance(query[i], ref[j]) <= radius in 2D.

.. autofunction:: _radpair2d()

_radius_pairs_2d()
^^^^^^^^^^^^^^^^^^^

The :func:`_radius_pairs_2d` function returns pairs (i,j) where ||query[i]-ref[j]|| <= radius on a 2D plane.

.. autofunction:: _radius_pairs_2d()

_nearest_neighbor_2d_numba()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_nearest_neighbor_2d_numba` function finds the nearest neighbor for each point in a 2D array (numba-accelerated).

.. autofunction:: _nearest_neighbor_2d_numba()

_nearest_neighbor_2d()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_nearest_neighbor_2d` function finds the nearest neighbor for each point in a 2D array.

.. autofunction:: _nearest_neighbor_2d()

_closest_points_on_segments_2d()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_closest_points_on_segments_2d` function finds the closest points between two 2D line segments.

.. autofunction:: _closest_points_on_segments_2d()

_refine_pairs_on_section()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_refine_pairs_on_section` function refines matched pairs using closest points between local segments.

.. autofunction:: _refine_pairs_on_section()

config.py
~~~~~~~~~

The config module provides configuration classes for connection discovery parameters in CR3BP.

.. currentmodule:: hiten.algorithms.connections.config

_SearchConfig()
^^^^^^^^^^^^^^^

The :class:`_SearchConfig` class defines search parameters and tolerances for connection discovery.

.. autoclass:: _SearchConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

ConnectionConfig()
^^^^^^^^^^^^^^^^^^

The :class:`ConnectionConfig` class defines an extended configuration including computational parameters.

.. autoclass:: ConnectionConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

engine.py
~~~~~~~~~

The engine module provides a connection engine for orchestrating manifold transfer discovery in CR3BP.

.. currentmodule:: hiten.algorithms.connections.engine

_ConnectionProblem()
^^^^^^^^^^^^^^^^^^^^

The :class:`_ConnectionProblem` class defines a problem specification for connection discovery between two manifolds.

.. autoclass:: _ConnectionProblem()
   :members:
   :undoc-members:
   :exclude-members: __init__

_ConnectionEngine()
^^^^^^^^^^^^^^^^^^^

The :class:`_ConnectionEngine` class provides the main engine for orchestrating connection discovery between manifolds.

.. autoclass:: _ConnectionEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__

interfaces.py
~~~~~~~~~~~~~

The interfaces module provides interface classes for manifold data access in connection discovery.

.. currentmodule:: hiten.algorithms.connections.interfaces

_ManifoldInterface()
^^^^^^^^^^^^^^^^^^^^

The :class:`_ManifoldInterface` class provides an interface for accessing manifold data in connection discovery.

.. autoclass:: _ManifoldInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

results.py
~~~~~~~~~~

The results module provides result classes for connection discovery data in CR3BP.

.. currentmodule:: hiten.algorithms.connections.results

_ConnectionResult()
^^^^^^^^^^^^^^^^^^^

The :class:`_ConnectionResult` class stores an individual connection result between two manifolds.

.. autoclass:: _ConnectionResult()
   :members:
   :undoc-members:
   :exclude-members: __init__

ConnectionResults()
^^^^^^^^^^^^^^^^^^^

The :class:`ConnectionResults` class provides a collection of connection results with convenient access and formatting.

.. autoclass:: ConnectionResults()
   :members:
   :undoc-members:
   :exclude-members: __init__
