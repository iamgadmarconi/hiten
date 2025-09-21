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
    Core numerical algorithms that implement the continuation loops (predict-correct-accept cycles)

**Engine**
    Orchestration layer that coordinates the continuation process using backends and interfaces

**Interfaces**
    Domain-specific adapters that provide instantiation, correction, and parameter extraction methods

**Stepping Strategies**
    Concrete implementations of prediction strategies (natural parameter, secant-based, etc.)

**Configuration**
    Configuration classes that encapsulate continuation parameters and settings

**Facades**
    User-friendly wrapper classes that provide convenient APIs for common continuation tasks

**Types**
    Data structures and type definitions for continuation results and problems

**Protocols**
    Runtime-checkable protocol definitions for the framework components
