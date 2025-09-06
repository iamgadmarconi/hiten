"""
hiten.algorithms.continuation
=============================

The `hiten.algorithms.continuation` package provides a comprehensive framework
for numerical continuation of dynamical systems solutions. This package
implements various continuation algorithms used to trace families of solutions
(such as periodic orbits, invariant manifolds, and fixed points) as system
parameters are varied.

Numerical continuation is a fundamental tool in dynamical systems analysis,
enabling the systematic exploration of solution families, detection of
bifurcations, and understanding of parameter-dependent behavior. This package
provides both high-level user-friendly interfaces and low-level algorithmic
components for advanced customization.

The framework supports multiple continuation strategies:

- **Natural Parameter Continuation**: Simple parameter variation suitable for
  smooth solution branches without bifurcations.
- **Pseudo-Arclength Continuation**: Robust method that can navigate around
  turning points and bifurcations using arc-length parameterization.

All algorithms are designed for the Circular Restricted Three-Body Problem
(CR3BP) and use nondimensional units consistent with the underlying dynamical
system.

Main Components
---------------

High-Level Interfaces
~~~~~~~~~~~~~~~~~~~~~
:class:`~hiten.algorithms.continuation.predictors._StateParameter`
    Continue families by varying components of the initial state vector.
:class:`~hiten.algorithms.continuation.predictors._FixedPeriod`
    Continue families at fixed orbital periods (future implementation).
:class:`~hiten.algorithms.continuation.predictors._EnergyLevel`
    Continue families at fixed energy levels (future implementation).

These classes provide ready-to-use continuation algorithms for common
scenarios in astrodynamics and dynamical systems.

Core Framework
~~~~~~~~~~~~~~
:class:`~hiten.algorithms.continuation.base._ContinuationEngine`
    Abstract base class implementing the predict-correct continuation framework.
:class:`~hiten.algorithms.continuation.interfaces._PeriodicOrbitContinuationInterface`
    Domain-specific interface for periodic orbit continuation.
:class:`~hiten.algorithms.continuation.interfaces._InvariantToriContinuationInterface`
    Domain-specific interface for invariant tori continuation (future implementation).

The core framework provides the foundation for building custom continuation
algorithms tailored to specific problem domains.

Algorithm Strategies
~~~~~~~~~~~~~~~~~~~~
:mod:`~hiten.algorithms.continuation.strategies`
    Low-level algorithmic components including stepping methods, algorithm
    strategies, and interface protocols.

The strategies subpackage contains the building blocks used by higher-level
interfaces, enabling advanced users to create custom continuation algorithms.

Typical Workflow
----------------
The typical workflow for using the continuation framework involves:

1. **Setup**: Create or load an initial solution (e.g., periodic orbit).
2. **Configuration**: Choose continuation parameters and stopping criteria.
3. **Execution**: Run the continuation algorithm to generate a family.
4. **Analysis**: Examine the resulting family for bifurcations or special points.

Example Usage
-------------
Here's a basic example of continuing a family of Halo orbits:

>>> from hiten.system import System
>>> from hiten.algorithms.continuation.predictors import _StateParameter
>>>
>>> # Setup system and initial orbit
>>> system = System.from_bodies("earth", "moon")
>>> l1 = system.get_libration_point(1)
>>> halo = l1.create_orbit('halo', amplitude_z=0.5, zenith='southern')
>>> halo.correct()
>>>
>>> # Create continuation algorithm
>>> continuation = _StateParameter(
>>>     initial_orbit=halo,
>>>     parameter_indices=[2],  # Continue in z-component
>>>     step_size=0.01,
>>>     max_steps=100
>>> )
>>>
>>> # Run continuation
>>> family = continuation.run()
>>> 
>>> # Analyze results
>>> print(f"Generated {len(family)} orbits in family")
>>> for orbit in family:
>>>     print(f"z-amplitude: {orbit.initial_state[2]:.4f}, period: {orbit.period:.4f}")

Advanced Usage
--------------
For advanced users, the framework supports custom continuation algorithms:

>>> from hiten.algorithms.continuation.base import _ContinuationEngine
>>> from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
>>> from hiten.algorithms.continuation.strategies import _NaturalParameter
>>>
>>> class CustomContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
>>>     def __init__(self, initial_orbit, custom_predictor, **kwargs):
>>>         from hiten.algorithms.continuation.strategies import _NaturalParameterStep
>>>         stepper = _NaturalParameterStep(custom_predictor)
>>>         super().__init__(stepper, initial_orbit=initial_orbit, **kwargs)

Mathematical Background
-----------------------
The continuation algorithms implement the predict-correct framework:

1. **Prediction**: Generate a candidate solution using the current solution
   and step size. Different strategies use different prediction methods.
2. **Correction**: Refine the candidate solution to satisfy the governing
   equations using Newton's method or similar techniques.
3. **Step Control**: Adapt the step size based on convergence behavior
   and solution quality.

The framework supports both natural parameter continuation (varying a single
parameter) and pseudo-arclength continuation (parameterizing by arc length
along the solution curve).

Bifurcation Detection
~~~~~~~~~~~~~~~~~~~~~
While not explicitly implemented in this version, the framework provides
the foundation for bifurcation detection through:

- Monitoring solution stability properties
- Tracking parameter-dependent quantities
- Detecting qualitative changes in solution behavior

Performance Considerations
--------------------------
The continuation algorithms are designed for efficiency:

- **Adaptive stepping**: Automatic step size control for optimal performance
- **Efficient correction**: Fast Newton iteration with good initial guesses
- **Memory management**: Minimal memory overhead for long continuations
- **Extensibility**: Clean interfaces for custom optimization

The framework balances computational efficiency with numerical robustness,
making it suitable for both interactive exploration and large-scale
parameter studies.

See Also
--------
:mod:`hiten.system.orbits`
    Orbit classes that can be continued using this framework.
:mod:`hiten.system.manifold`
    Manifold classes for continuation of invariant manifolds.
:mod:`hiten.algorithms.corrector`
    Correction algorithms used in the predict-correct framework.

Notes
-----
This package focuses on continuation of solutions in the CR3BP, but the
framework is designed to be extensible to other dynamical systems with
appropriate interface implementations.

The continuation algorithms assume that solutions can be represented as
objects with well-defined correction procedures and parameter extraction
methods. This design enables continuation of diverse solution types
(periodic orbits, quasi-periodic tori, fixed points, etc.) within a
unified framework.
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
