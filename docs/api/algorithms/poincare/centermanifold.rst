Center Manifold Poincare Maps
=============================

The centermanifold module provides Poincare map computation restricted to center manifolds of collinear libration points in the CR3BP.

.. currentmodule:: hiten.algorithms.poincare.centermanifold

Main Classes
------------

.. autoclass:: CenterManifoldMapPipeline()
   :members:
   :undoc-members:
   :exclude-members: __init__

Configuration Classes
---------------------

.. autoclass:: _CenterManifoldMapConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Backend Classes
---------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.backend

.. autoclass:: _CenterManifoldBackend()
   :members:
   :undoc-members:
   :exclude-members: __init__

Engine Classes
--------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.engine

.. autoclass:: _CenterManifoldEngine()
   :members:
   :undoc-members:
   :exclude-members: __init__

Interface Classes
-----------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.interfaces

.. autoclass:: _CenterManifoldInterface()
   :members:
   :undoc-members:
   :exclude-members: __init__

Strategy Classes
----------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.seeding

.. autoclass:: _CenterManifoldSeedingBase()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. currentmodule:: hiten.algorithms.poincare.centermanifold.strategies

.. autofunction:: _make_strategy()

Type Classes
------------

.. currentmodule:: hiten.algorithms.poincare.centermanifold.types

.. autoclass:: CenterManifoldMapResults()
   :members:
   :undoc-members:
   :exclude-members: __init__

.. autoclass:: _CenterManifoldMapProblem()
   :members:
   :undoc-members:
   :exclude-members: __init__
