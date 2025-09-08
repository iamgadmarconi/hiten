"""Strategy and stepping components for continuation algorithms.

This package provides abstract strategy classes and concrete stepping
implementations that integrate with the base continuation engine. It also
exposes protocol definitions for implementing custom stepping logic.

The components are grouped as:

- **Algorithm strategies**: natural parameter and pseudo-arclength.
- **Stepping methods**: natural parameter and secant-based steppers.
- **Protocols**: interfaces that stepping implementations should follow.

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
