"""Provide a comprehensive framework for numerical continuation.

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

Examples
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

See Also
--------
:mod:`~hiten.system.orbits`
    Orbit classes that can be continued using this framework.
:mod:`~hiten.system.manifold`
    Manifold classes for continuation of invariant manifolds.
:mod:`~hiten.algorithms.corrector`
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

from .base import _ContinuationEngine
from .config import _OrbitContinuationConfig
from .interfaces import (_InvariantToriContinuationInterface,
                         _PeriodicOrbitContinuationInterface)
from .predictors import _EnergyLevel, _FixedPeriod, _StateParameter

__all__ = [
    "_StateParameter",
    "_FixedPeriod", 
    "_EnergyLevel",
    "_ContinuationEngine",
    "_OrbitContinuationConfig",
    "_PeriodicOrbitContinuationInterface",
    "_InvariantToriContinuationInterface",
]
