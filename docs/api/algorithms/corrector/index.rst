Corrector Module
================

The corrector module provides robust iterative correction algorithms for solving nonlinear systems in dynamical systems analysis.

The module implements a modular architecture that separates algorithmic components from domain-specific logic, enabling flexible combinations of different correction strategies with various problem types.

.. toctree::
   :maxdepth: 2

   config
   backends
   interfaces
   stepping
   protocols
   engine
   types

.. currentmodule:: hiten.algorithms.corrector

Overview
~~~~~~~~

The correction framework consists of several key components:

**Backend Algorithms**
    Core numerical algorithms for solving nonlinear systems (Newton-Raphson, etc.)

**Interfaces**
    Domain-specific interfaces that translate between problem objects and the abstract vector representations expected by correction algorithms

**Step Control**
    Step-size control strategies for robust convergence (plain steps, line search, etc.)

**Configuration**
    Configuration classes for algorithm parameters and problem-specific settings

**Engine**
    Orchestration layer that combines backends and interfaces for complete correction workflows

**Types**
    Data structures and type definitions for correction results and problems

**Protocols**
    Protocol definitions for the correction framework components
