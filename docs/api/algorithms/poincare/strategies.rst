Computational Strategies
========================

The strategies modules provide different computational approaches for Poincare return map computation. These strategies optimize performance for different types of dynamical systems and computational requirements.

Core Strategies
---------------

.. currentmodule:: hiten.algorithms.poincare.core.strategies

.. autoclass:: _BaseStrategy()
   :members:
   :exclude-members: __init__

Abstract base class for all computational strategies. Defines the interface that strategies must implement for consistent integration with the Poincare map engines.

.. autoclass:: _ParallelStrategy()
   :members:
   :exclude-members: __init__

Parallel computation strategy for Poincare maps. Implements parallel processing of multiple trajectories using multiprocessing for improved performance on multi-core systems.

Center Manifold Strategies
--------------------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.strategies

.. autoclass:: _CenterManifoldStrategy()
   :members:
   :exclude-members: __init__

Specialized strategy for center manifold Poincare maps. Optimizes computation for trajectories restricted to center manifolds of collinear libration points.

Synodic Strategies
------------------

.. currentmodule:: hiten.algorithms.poincare.synodic.strategies

.. autoclass:: _SynodicStrategy()
   :members:
   :exclude-members: __init__

Strategy optimized for synodic frame computations. Provides efficient handling of rotating coordinate systems and synodic Poincare sections.
