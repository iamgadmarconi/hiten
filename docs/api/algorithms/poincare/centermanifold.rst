Center Manifold Poincare Maps
=============================

The centermanifold module provides specialized Poincare map computation for center manifolds of collinear libration points in the CR3BP. This module implements efficient algorithms for computing return maps restricted to the center manifold, which is crucial for understanding the local dynamics near libration points.

.. currentmodule:: hiten.algorithms.poincare.centermanifold

.. autoclass:: CenterManifoldMap
   :members:
   :special-members: __init__

Main user-facing class for computing Poincare maps on center manifolds. Provides a high-level interface for generating return maps with various seeding strategies and visualization capabilities.

.. autoclass:: _CenterManifoldMapConfig
   :members:
   :special-members: __init__

Configuration class for center manifold Poincare maps. Combines parameters for integration, iteration control, and seeding strategies specific to center manifold computations.

.. autoclass:: _CenterManifoldSectionConfig
   :members:
   :special-members: __init__

Configuration for Poincare sections on center manifolds. Defines section coordinates, plane mappings, and coordinate transformations for the 4D center manifold phase space.

.. autoclass:: _CenterManifoldBackend
   :members:
   :special-members: __init__

Numba-compiled backend for efficient center manifold computations. Handles numerical integration, section crossing detection, and parallel processing for high-performance map generation.

.. autoclass:: _CenterManifoldEngine
   :members:
   :special-members: __init__

Computation engine for center manifold Poincare maps. Coordinates seeding strategies, parallel processing, and iterative map computation to generate complete return maps.

.. autoclass:: _CenterManifoldSeedingBase
   :members:
   :special-members: __init__

Abstract base class for center manifold seeding strategies. Provides common functionality for Hill boundary validation and seed generation on the center manifold.

.. autoclass:: _SingleAxisSeeding
   :members:
   :special-members: __init__

Seeding strategy that generates seeds along a single coordinate axis. Useful for exploring center manifold dynamics along specific coordinate directions.

.. autoclass:: _AxisAlignedSeeding
   :members:
   :special-members: __init__

Seeding strategy that generates seeds along both coordinate axes. Provides good coverage of coordinate directions with approximately half the seeds on each axis.

.. autoclass:: _LevelSetsSeeding
   :members:
   :special-members: __init__

Seeding strategy based on level sets of coordinates. Creates a grid-like pattern of seeds for uniform coverage of the section plane.

.. autoclass:: _RadialSeeding
   :members:
   :special-members: __init__

Seeding strategy that distributes seeds on concentric circles. Provides radial coverage of the section plane in polar coordinates.

.. autoclass:: _RandomSeeding
   :members:
   :special-members: __init__

Seeding strategy using uniform rejection sampling. Generates random seeds within the Hill boundary for statistical coverage of the section plane.

.. autofunction:: _make_strategy

Factory function for creating concrete seeding strategy instances based on string identifiers.

.. autofunction:: _get_section_config

Utility function for retrieving cached section configuration objects for specific coordinates.
