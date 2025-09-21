Continuation Strategies Module
==============================

The strategies module provides specialized algorithmic strategies for different continuation methods. This module contains abstract base classes and strategy definitions for natural parameter continuation, pseudo-arclength continuation, and related interfaces.

The module is organized into several submodules:

.. toctree::
   :maxdepth: 2

   algorithms
   stepping
   interfaces

.. currentmodule:: hiten.algorithms.continuation.strategies

Overview
~~~~~~~~

The strategies module provides abstract base classes for different continuation algorithm strategies. Each strategy defines a particular approach to parameter continuation with specialized stepping logic, stopping criteria, and parameter handling.

**Algorithm Strategies**
    Abstract base classes for natural parameter and pseudo-arclength continuation algorithms

**Stepping Interfaces**
    Protocol and base classes for continuation stepping strategies

**Stepping Strategies**
    Concrete implementations of stepping strategies used in continuation algorithms
