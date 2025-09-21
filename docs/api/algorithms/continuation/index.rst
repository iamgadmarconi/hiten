Continuation Module
===================

The continuation module provides a comprehensive framework for numerical continuation of solutions in dynamical systems.

The module implements a modular architecture that separates algorithmic components from domain-specific logic, enabling flexible combinations of different continuation strategies with various problem types.

.. toctree::
   :maxdepth: 2

   config
   backends
   engine
   interfaces
   facades
   stepping
   types
   protocols

.. currentmodule:: hiten.algorithms.continuation

Overview
~~~~~~~~

The continuation framework consists of several key components:

**Backend Algorithms**
    Core numerical algorithms for driving the continuation process (predict-correct loops, etc.)

**Engine**
    Orchestration layer that combines backends and interfaces for complete continuation workflows

**Interfaces**
    Domain-specific interfaces that translate between problem objects and the abstract vector representations expected by continuation algorithms

**Step Control**
    Step-size control strategies for robust continuation (natural parameter, pseudo-arclength, etc.)

**Configuration**
    Configuration classes for algorithm parameters and problem-specific settings

**Facades**
    User-friendly interfaces that assemble engines, backends, and interfaces using dependency injection

**Types**
    Data structures and type definitions for continuation results and problems

**Protocols**
    Protocol definitions for the continuation framework components

**Strategies**
    Specialized algorithmic strategies for different continuation methods (see strategies/ subdirectory)
