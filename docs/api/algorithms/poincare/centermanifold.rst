Center Manifold Poincare Maps
=============================

The centermanifold module provides Poincare map computation restricted to center manifolds of collinear libration points in the CR3BP. It supports various seeding strategies and provides visualization capabilities for understanding local dynamics.

.. currentmodule:: hiten.algorithms.poincare.centermanifold

.. autoclass:: CenterManifoldMap()
   :members:
   :exclude-members: __init__

Main user-facing class for center manifold Poincare maps. Provides interface for computing and analyzing Poincare maps restricted to center manifolds with various seeding strategies.

.. autoclass:: _CenterManifoldMapConfig()
   :members:
   :exclude-members: __init__, __post_init__

Configuration class for center manifold Poincare maps. Combines integration, iteration, and seeding parameters with center manifold-specific settings.

.. autoclass:: CenterManifoldMapResults()
   :members:
   :exclude-members: __init__

Results class for center manifold Poincare maps. Extends the base Section class with center manifold-specific projection capabilities.

Backend Classes
===============

.. currentmodule:: hiten.algorithms.poincare.centermanifold.backend

.. autoclass:: _CenterManifoldBackend()
   :members:
   :exclude-members: __init__

Backend for center manifold computations in the CR3BP. Uses Numba-compiled kernels for efficient Hamiltonian integration and Poincare map evaluation.

.. autofunction:: _detect_crossing()

Detect if trajectory crossed the Poincare section using Hermite interpolation.

.. autofunction:: _solve_bracketed()

Pure-Python Brent's method root-finder for bracketed scalar functions.

.. autofunction:: _get_rk_coefficients()

Return Runge-Kutta coefficients for specified order.

.. autofunction:: _integrate_rk_ham()

Integrate Hamiltonian system using Runge-Kutta method.

.. autofunction:: _integrate_map()

Integrate Hamiltonian system using specified method (Runge-Kutta or symplectic).

.. autofunction:: _poincare_step()

Perform one Poincare map step for center manifold integration.

.. autofunction:: _poincare_map()

Compute Poincare map for multiple center manifold seeds in parallel.

Engine and Configuration Classes
=================================

.. currentmodule:: hiten.algorithms.poincare.centermanifold.engine

.. autoclass:: _CenterManifoldEngine()
   :members:
   :exclude-members: __init__

Engine for center manifold Poincare map computation. Coordinates seeding strategy, numerical integration, and parallel processing.

.. currentmodule:: hiten.algorithms.poincare.centermanifold.config

.. autoclass:: _CenterManifoldSectionConfig()
   :members:
   :exclude-members: __init__

Configuration for center manifold Poincare sections. Provides mappings between coordinate names and indices.

.. autofunction:: _get_section_config()

Get a cached section configuration for the specified coordinate.

Interface Classes
=================

.. currentmodule:: hiten.algorithms.poincare.centermanifold.interfaces

.. autoclass:: _CenterManifoldInterface()
   :members:

Stateless adapter for center manifold section computations. Provides domain-level translations between plane points and center manifold states.

.. autofunction:: _solve_bracketed()

Brent-style bracketed scalar root solve in pure Python.

Seeding Strategies
==================

.. currentmodule:: hiten.algorithms.poincare.centermanifold.seeding

.. autoclass:: _CenterManifoldSeedingBase()
   :members:
   :exclude-members: __init__

Base class for center manifold seeding strategies. Provides common functionality for Hill boundary validation.

.. currentmodule:: hiten.algorithms.poincare.centermanifold.strategies

.. autofunction:: _make_strategy()

Factory returning a concrete seeding strategy based on string identifier.
