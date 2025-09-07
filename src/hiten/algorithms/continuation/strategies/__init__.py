"""
hiten.algorithms.continuation.strategies
========================================

The `hiten.algorithms.continuation.strategies` package provides the core
algorithmic components for numerical continuation in dynamical systems.
This package implements various continuation strategies, stepping methods,
and protocol definitions that form the foundation of the continuation
framework.

The package is organized into three main components:

1. **Algorithm Strategies**: Abstract base classes that implement different
   continuation approaches (natural parameter, pseudo-arclength).
2. **Stepping Methods**: Concrete implementations of prediction strategies
   used within continuation algorithms.
3. **Interface Protocols**: Protocol definitions that ensure consistent
   interfaces across different stepping strategies.

These components work together to provide a flexible and extensible
framework for tracing families of solutions in parameter space, with
applications to periodic orbits, invariant manifolds, and other dynamical
structures in the Circular Restricted Three-Body Problem (CR3BP).

All numerical computations use nondimensional units appropriate for the
specific dynamical system being studied.

Examples
-------------
The strategies are typically used through higher-level interfaces in the
:mod:`hiten.algorithms.continuation` package, but can be combined directly
for custom continuation scenarios:

>>> from hiten.algorithms.continuation.strategies._algorithms import _NaturalParameter
>>> from hiten.algorithms.continuation.strategies._stepping import _NaturalParameterStep
>>> from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
>>>
>>> # Create a custom continuation algorithm
>>> class CustomOrbitContinuation(_NaturalParameter, _PeriodicOrbitContinuationInterface):
>>>     def __init__(self, predictor_fn, **kwargs):
>>>         stepper = _NaturalParameterStep(predictor_fn)
>>>         super().__init__(stepper, **kwargs)

See Also
--------
:mod:`hiten.algorithms.continuation.base`
    Base continuation engine that coordinates with strategies.
:mod:`hiten.algorithms.continuation.interfaces`
    Domain-specific interfaces for different problem types.
:mod:`hiten.algorithms.continuation.predictors`
    High-level predictor classes for common continuation tasks.
"""

# Algorithm strategy classes
from ._algorithms import (
    _NaturalParameter,
    _SecantArcLength,
)

# Stepping implementation classes  
from ._stepping import (
    _StepStrategy,
    _NaturalParameterStep,
    _SecantStep,
)

# Interface protocol classes
from ._step_interface import (
    _ContinuationStep,
    _PlainStep,
)

# Public API - expose core strategy classes
__all__ = [
    # Algorithm strategies
    "_NaturalParameter",
    "_SecantArcLength",
    
    # Stepping methods
    "_StepStrategy", 
    "_NaturalParameterStep",
    "_SecantStep",
    
    # Interface protocols
    "_ContinuationStep",
    "_PlainStep",
]
