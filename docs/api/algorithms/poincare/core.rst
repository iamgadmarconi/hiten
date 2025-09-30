Core Poincare Infrastructure
============================

The core module provides the fundamental infrastructure for Poincare return map computation.

.. currentmodule:: hiten.algorithms.poincare.core


Backend Classes
---------------

Defines the backend for Poincare map computation.

.. autoclass:: _ReturnMapBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__


Engine Classes
--------------

Defines the engine for Poincare map computation.

.. autoclass:: _ReturnMapEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__

Configuration Classes
---------------------

Defines the base configuration for Poincare map computation.

.. autoclass:: _ReturnMapBaseConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the configuration for integration.

.. autoclass:: _IntegrationConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the configuration for iteration.

.. autoclass:: _IterationConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the configuration for seeding.

.. autoclass:: _SeedingConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the configuration for refinement.

.. autoclass:: _RefineConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the configuration for Poincare map computation.

.. autoclass:: _ReturnMapConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Event Classes
-------------

Defines the event for surface computation.

.. autoclass:: _SurfaceEvent()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the event for plane computation.

.. autoclass:: _PlaneEvent()
   :members:
   :undoc-members:
   :exclude-members: __init__

Interface Classes
-----------------

Defines the interface for section computation.

.. autoclass:: _SectionInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the interface for Poincare map computation.

.. autoclass:: _PoincareBaseInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

Strategy Classes
----------------

Defines the base class for seeding strategies.

.. autoclass:: _SeedingStrategyBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

Defines the protocol for seeding strategies.

.. autoclass:: _SeedingProtocol()
   :members:
   :undoc-members:
   :exclude-members: __init__

Type Classes
------------

The section classes are used to store the results of the Poincare map computation.

.. autoclass:: _Section()
   :members:
   :undoc-members:
   :exclude-members: __init__

The section hit class is used to store the results of a single trajectory-section intersection.

.. autoclass:: _SectionHit()
   :members:
   :undoc-members:
   :exclude-members: __init__

The map results class is used to store the results of the Poincare map computation.

.. autoclass:: _MapResults()
   :members:
   :undoc-members:
   :exclude-members: __init__
