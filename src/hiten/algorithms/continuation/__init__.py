"""Continuation framework for tracing solution families.

This package implements the predict-correct continuation engine along with
domain-specific interfaces, strategy abstractions, and stepping methods for
numerical continuation in dynamical systems. It supports tracing families of
solutions (e.g., periodic orbits) as parameters vary.

See Also
--------
:mod:`hiten.system.orbits`
    Orbit classes that can be continued using this framework.
:mod:`hiten.system.manifold`
    Manifold classes for continuation of invariant manifolds.
:mod:`hiten.algorithms.corrector`
    Correction algorithms used in the predict-correct framework.
"""

# Core framework components
from .base import _ContinuationEngine
from .interfaces import (_InvariantToriContinuationInterface,
                         _OrbitContinuationConfig,
                         _PeriodicOrbitContinuationInterface)
# High-level user interfaces
from .predictors import _EnergyLevel, _FixedPeriod, _StateParameter
# Algorithm strategies (re-exported for convenience)
from .strategies import (_ContinuationStep, _NaturalParameter,
                         _NaturalParameterStep, _PlainStep, _SecantArcLength,
                         _SecantStep, _StepStrategy)

# Public API - expose main continuation classes
__all__ = [
    # High-level predictors
    "_StateParameter",
    "_FixedPeriod", 
    "_EnergyLevel",
    
    # Core framework
    "_ContinuationEngine",
    "_OrbitContinuationConfig",
    "_PeriodicOrbitContinuationInterface",
    "_InvariantToriContinuationInterface",
    
    # Algorithm strategies (for advanced users)
    "_NaturalParameter",
    "_SecantArcLength",
    "_StepStrategy",
    "_NaturalParameterStep", 
    "_SecantStep",
    "_ContinuationStep",
    "_PlainStep",
]
