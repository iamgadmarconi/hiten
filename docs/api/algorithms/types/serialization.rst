Serialization
=============

The serialization module provides utilities for handling Numba objects and other problematic types during pickling.

.. currentmodule:: hiten.algorithms.types.serialization

Serialize Base
--------------

_SerializeBase()
^^^^^^^^^^^^^^^^

Mixin class providing serialization utilities for objects with Numba dependencies. Handles conversion of problematic objects to serializable formats during pickling and reconstruction during unpickling.

.. autoclass:: _SerializeBase
   :members:
   :undoc-members:
   :exclude-members: __init__
