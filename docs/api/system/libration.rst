Libration Points Module
========================

The libration module provides abstract helpers to model Libration points of the Circular Restricted Three-Body Problem (CR3BP).

.. currentmodule:: hiten.system.libration

Base Classes
~~~~~~~~~~~~

The base module provides the core libration point framework.

.. currentmodule:: hiten.system.libration.base

LinearData
^^^^^^^^^^

Container with linearised CR3BP invariants.

.. autoclass:: LinearData
   :members:

LibrationPoint
^^^^^^^^^^^^^^

Abstract base class for Libration points of the CR3BP.

.. autoclass:: LibrationPoint
   :members:
   :undoc-members:
   :exclude-members: __init__

Collinear Points
~~~~~~~~~~~~~~~~

The collinear module provides collinear libration point classes.

.. currentmodule:: hiten.system.libration.collinear

CollinearPoint
^^^^^^^^^^^^^^

Base class for collinear Libration points (L1, L2, L3).

.. autoclass:: CollinearPoint
   :members:
   :undoc-members:
   :exclude-members: __init__

L1Point
^^^^^^^

L1 Libration point, located between the two primary bodies.

.. autoclass:: L1Point
   :members:
   :undoc-members:
   :exclude-members: __init__

L2Point
^^^^^^^

L2 Libration point, located beyond the smaller primary body.

.. autoclass:: L2Point
   :members:
   :undoc-members:
   :exclude-members: __init__

L3Point
^^^^^^^

L3 Libration point, located beyond the larger primary body.

.. autoclass:: L3Point
   :members:
   :undoc-members:
   :exclude-members: __init__

Triangular Points
~~~~~~~~~~~~~~~~~

The triangular module provides triangular libration point classes.

.. currentmodule:: hiten.system.libration.triangular

TriangularPoint
^^^^^^^^^^^^^^^

Abstract helper for the triangular Libration points.

.. autoclass:: TriangularPoint
   :members:
   :undoc-members:
   :exclude-members: __init__

L4Point
^^^^^^^

L4 Libration point, forming an equilateral triangle with the two primary bodies, located above the x-axis (positive y).

.. autoclass:: L4Point
   :members:
   :undoc-members:
   :exclude-members: __init__

L5Point
^^^^^^^

L5 Libration point, forming an equilateral triangle with the two primary bodies, located below the x-axis (negative y).

.. autoclass:: L5Point
   :members:
   :undoc-members:
   :exclude-members: __init__
