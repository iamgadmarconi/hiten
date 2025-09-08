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

Examples
-------------
Here's a basic example of continuing a family of Halo orbits in the
z-amplitude using natural parameter continuation:

>>> from hiten.system import System
>>> from hiten.algorithms.continuation.predictors import _StateParameter
>>> from hiten.algorithms.utils.types import SynodicState as S
>>>
>>> # Setup system and initial orbit
>>> system = System.from_bodies("earth", "moon")
>>> l1 = system.get_libration_point(1)
>>> halo = l1.create_orbit('halo', amplitude_z=0.5, zenith='southern')
>>> halo.correct()
>>>
>>> # Create continuation algorithm in amplitude of z
>>> continuation = _StateParameter(
...     initial_orbit=halo,
...     state=S.Z,
...     amplitude=True,
...     target=(float(halo.amplitude), float(halo.amplitude) * 1.5),
...     step=0.005,
...     max_orbits=50,
...     corrector_kwargs={'tol': 1e-9, 'max_attempts': 50},
... )
>>>
>>> # Run continuation
>>> family = continuation.run()
>>> print(f"Generated {len(family)} orbits in family")

Advanced Usage
--------------
For advanced users, the framework supports custom continuation algorithms:

>>> import numpy as np
>>> from hiten.algorithms.continuation.interfaces import _PeriodicOrbitContinuationInterface
>>> from hiten.algorithms.continuation.strategies import (
...     _NaturalParameter, _NaturalParameterStep,
... )
>>>
>>> class CustomContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
...     def __init__(self, *, initial_orbit, target, step, **kwargs):
...         # Simple predictor that nudges the z-state
...         self._predict_fn = lambda orbit, s: (
...             orbit.initial_state.copy().astype(float)
...             + np.array([0.0, 0.0, float(s[0]), 0.0, 0.0, 0.0])
...         )
...         super().__init__(
...             initial_orbit=initial_orbit,
...             parameter_getter=lambda o: np.asarray([float(o.initial_state[2])]),
...             target=target,
...             step=step,
...             **kwargs,
...         )
...
...     def _make_stepper(self):
...         return _NaturalParameterStep(self._predict_fn)

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
