Base Services
=============

The base services module defines the fundamental service abstractions used throughout the Hiten framework.

.. currentmodule:: hiten.algorithms.types.services.base

Persistence Service
-------------------

_PersistenceServiceBase()
^^^^^^^^^^^^^^^^^^^^^^^^^

Mixin offering a uniform persistence API around plain callables. Handles save, load, and load-in-place operations.

.. autoclass:: _PersistenceServiceBase
   :members:
   :undoc-members:
   :exclude-members: __init__

Dynamics Service
----------------

_DynamicsServiceBase()
^^^^^^^^^^^^^^^^^^^^^^

Mixin offering a uniform dynamics API around plain callables. Provides caching and domain-specific computations.

.. autoclass:: _DynamicsServiceBase
   :members:
   :undoc-members:
   :exclude-members: __init__

Cache Service
-------------

_CacheServiceBase()
^^^^^^^^^^^^^^^^^^^

Helper providing lazy caching for dynamics-oriented adapters. Manages cache operations and key generation.

.. autoclass:: _CacheServiceBase
   :members:
   :undoc-members:
   :exclude-members: __init__

Service Bundle
--------------

_ServiceBundleBase()
^^^^^^^^^^^^^^^^^^^^

Lightweight helper for service bundles offering ergonomic helpers. Coordinates multiple services for domain objects.

.. autoclass:: _ServiceBundleBase
   :members:
   :undoc-members:
   :exclude-members: __init__
