State Types
===========

The states module provides comprehensive type definitions and state vector containers for the circular restricted three-body problem.

.. currentmodule:: hiten.algorithms.types.states

Reference Frames
----------------

ReferenceFrame
^^^^^^^^^^^^^^

Enumeration of available reference frames for state vectors.

.. autoclass:: ReferenceFrame
   :members:

Coordinate Enumerations
-----------------------

SynodicState
^^^^^^^^^^^^

Enumeration for synodic frame coordinates. Defines indices for the 6D state vector in the rotating synodic frame.

.. autoclass:: SynodicState
   :members:

CenterManifoldState
^^^^^^^^^^^^^^^^^^^

Enumeration for center manifold coordinates. Defines indices for the 6D state vector in the center manifold coordinate system.

.. autoclass:: CenterManifoldState
   :members:

RestrictedCenterManifoldState
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enumeration for restricted center manifold coordinates. Defines indices for the 4D state vector in the restricted center manifold coordinate system.

.. autoclass:: RestrictedCenterManifoldState
   :members:

State Vector Containers
-----------------------

_BaseStateContainer()
^^^^^^^^^^^^^^^^^^^^^

Minimal mutable container for a single state vector, indexed by an IntEnum. Base class for all state vector containers.

.. autoclass:: _BaseStateContainer
   :members:
   :undoc-members:
   :exclude-members: __init__

SynodicStateVector()
^^^^^^^^^^^^^^^^^^^^

Container for synodic frame state vectors. Provides convenient access to 6D state vectors in the rotating synodic frame.

.. autoclass:: SynodicStateVector
   :members:
   :undoc-members:
   :exclude-members: __init__

CenterManifoldStateVector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Container for center manifold state vectors. Provides convenient access to 6D state vectors in the center manifold coordinate system.

.. autoclass:: CenterManifoldStateVector
   :members:
   :undoc-members:
   :exclude-members: __init__

RestrictedCenterManifoldStateVector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Container for restricted center manifold state vectors. Provides convenient access to 4D state vectors in the restricted center manifold coordinate system.

.. autoclass:: RestrictedCenterManifoldStateVector
   :members:
   :undoc-members:
   :exclude-members: __init__

Trajectory
----------

Trajectory()
^^^^^^^^^^^^

Container for time-series of state vectors. Provides efficient storage and access to trajectory data.

.. autoclass:: Trajectory
   :members:
   :undoc-members:
   :exclude-members: __init__
