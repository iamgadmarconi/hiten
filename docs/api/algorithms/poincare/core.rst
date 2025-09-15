Core Poincare Map Infrastructure
=================================

The core module provides the fundamental infrastructure for Poincare return map computation, including base classes, configuration management, and common utilities. This module defines the abstract interfaces and common functionality used by all excludeized Poincare map implementations.

.. currentmodule:: hiten.algorithms.poincare.core

.. autoclass:: _ReturnMapBase()
   :members:
   :exclude-members: __init__, __len__, __repr__

Abstract base class for all Poincare return map implementations. Provides common interface for map computation, section management, and caching functionality.

.. autoclass:: _Section()
   :members:
   :exclude-members: __init__, __len__, __repr__

Container class for Poincare section data. Stores points, states, labels, and timing information for a computed section.

.. autoclass:: _ReturnMapBackend()
   :members:
   :exclude-members: __init__, __len__

Abstract base class for return map backends. Defines the interface for numerical integration and section crossing detection that concrete backends must implement.

.. autoclass:: _ReturnMapEngine()
   :members:
   :exclude-members: __init__, __len__, __repr__

Abstract base class for return map computation engines. Coordinates seeding strategies, parallel processing, and iterative map computation.

.. autoclass:: _SeedingStrategyBase()
   :members:
   :exclude-members: __init__

Abstract base class for seeding strategies. Provides common functionality for generating initial conditions and configuration management.

.. autoclass:: _ReturnMapBaseConfig()
   :members:
   :exclude-members: __init__

Base configuration class for return maps. Contains orchestration parameters like compute_on_init and n_workers.

.. autoclass:: _IntegrationConfig()       
   :members:
   :exclude-members: __init__

Configuration class for numerical integration parameters. Defines integration method, order, time step, and other numerical settings.

.. autoclass:: _IterationConfig()
   :members:
   :exclude-members: __init__

Configuration class for iteration control. Defines the number of return map iterations to compute for each initial condition.

.. autoclass:: _SeedingConfig()
   :members:
   :exclude-members: __init__

Configuration class for seeding strategies. Defines the number of initial seeds to generate for return map computation.

.. autoclass:: _ReturnMapConfig()
   :members:
   :exclude-members: __init__

Complete configuration class combining all configuration mixins. Provides a comprehensive set of parameters for return map computation.

.. autoclass:: _SectionConfig()
   :members:
   :exclude-members: __init__

Abstract base class for Poincare section configuration. Defines the interface for section coordinate and plane coordinate specification.

.. autoclass:: _EngineConfigLike()
   :members:
   :exclude-members: __init__

Protocol for engine configuration objects. Defines the minimum interface required for engine configuration objects.

.. autoclass:: _SeedingConfigLike()
   :members:
   :exclude-members: __init__

Protocol for seeding configuration objects. Defines the minimum interface required for seeding configuration objects.

.. autoclass:: _SurfaceEvent()
   :members:
   :exclude-members: __init__

Abstract base class for Poincare section surface events. Defines the interface for detecting trajectory crossings through Poincare sections.

.. autoclass:: _SectionHit()
   :members:
   :exclude-members: __init__, __repr__

Container for a single trajectory-section intersection. Stores time, state vector, and 2D projection coordinates for efficient access.

.. autoclass:: _PlaneEvent()
   :members:
   :exclude-members: __init__

Concrete surface event for hyperplanes defined by setting a coordinate to a constant value. Supports both coordinate name and index-based specification.

.. autoclass:: _SeedingProtocol()
   :members:
   :exclude-members: __init__

Protocol for seeding strategies in Poincare return map computation. Defines the interface that all seeding strategies must implement.
