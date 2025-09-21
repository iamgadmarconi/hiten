Types and Data Structures
=========================

The types module provides core data structures and result containers used throughout the Poincare return map computation system. These classes define the fundamental data types for sections, intersections, and results.

Core Types
----------

.. currentmodule:: hiten.algorithms.poincare.core.types

.. autoclass:: _Section()
   :members:
   :exclude-members: __init__

Immutable container for a single 2D return map slice. Holds intersection points, state vectors, axis labels, and optional integration times.

.. autoclass:: _SectionHit()
   :members:

Named tuple container for a single trajectory-section intersection. Provides both full state vector and 2D projection for efficient access.

.. autoclass:: _MapResults()
   :members:
   :exclude-members: __init__

Base results object for Poincare maps that extends _Section functionality. Serves as a common base for module-specific results.

Center Manifold Types
--------------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.types

.. autoclass:: CenterManifoldMapResults()
   :members:
   :exclude-members: __init__

User-facing results for center manifold Poincare maps. Extends _MapResults with specialized projection methods for center manifold coordinates.

.. autoclass:: _CenterManifoldMapProblem()
   :members:

Immutable problem definition for center manifold map runs. Contains all parameters needed to define a center manifold Poincare map computation.

Synodic Types
-------------

.. currentmodule:: hiten.algorithms.poincare.synodic.types

.. autoclass:: SynodicMapResults()
   :members:
   :exclude-members: __init__

User-facing results for synodic Poincare sections. Extends _MapResults for synodic frame computations.

.. autoclass:: _SynodicMapProblem()
   :members:

Problem definition for synodic section runs. Contains parameters for synodic frame Poincare map computations including plane coordinates, direction filters, and trajectory data.
