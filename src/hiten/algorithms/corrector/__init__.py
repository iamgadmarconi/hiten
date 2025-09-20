"""Provide robust iterative correction algorithms for solving nonlinear systems.

The :mod:`~hiten.algorithms.corrector` package provides robust iterative correction
algorithms for solving nonlinear systems arising in dynamical systems analysis.
These algorithms are essential for refining approximate solutions to high
precision, particularly for periodic orbits, invariant manifolds, and other
dynamical structures in the Circular Restricted Three-Body Problem (CR3BP).

The package implements a modular architecture that separates algorithmic
components from domain-specific logic, enabling flexible combinations of
different correction strategies with various problem types.

Examples
-------------
Most users will call `PeriodicOrbit.correct()` which wires a default stepper.
Advanced users can inject a custom stepper factory:

>>> from hiten.algorithms.corrector import _NewtonOrbitCorrector
>>> from hiten.algorithms.corrector.stepping import make_armijo_stepper
>>> from hiten.algorithms.corrector.config import _LineSearchConfig
>>> stepper_factory = make_armijo_stepper(_LineSearchConfig())
>>> corrector = _NewtonOrbitCorrector(stepper_factory=stepper_factory)
>>> corrected_orbit = corrector.correct(orbit)

Advanced users can create custom correctors by combining components:

>>> from hiten.algorithms.corrector.backends.newton import (_NewtonBackend, 
...                                        _PeriodicOrbitCorrectorInterface)
>>> class CustomCorrector(_PeriodicOrbitCorrectorInterface, _NewtonBackend):
...     pass

------------

All algorithms use nondimensional units consistent with the underlying
dynamical system and are designed for high-precision applications in
astrodynamics and mission design.

See Also
--------
:mod:`~hiten.system.orbits`
    Orbit classes that can be corrected using these algorithms.
:mod:`~hiten.algorithms.continuation`
    Continuation algorithms that use correction for family generation.
"""

from .backends.base import _CorrectorBackend
from .backends.newton import _NewtonBackend
from .config import (_BaseCorrectionConfig, _LineSearchConfig,
                     _OrbitCorrectionConfig)
from .correctors import _NewtonOrbitCorrector
from .interfaces import (_InvariantToriCorrectorInterface,
                         _PeriodicOrbitCorrectorInterface)

__all__ = [
    "_NewtonOrbitCorrector",
    
    "_NewtonBackend",
    "_ArmijoLineSearch",
    
    "_BaseCorrectionConfig",
    "_OrbitCorrectionConfig", 
    "_LineSearchConfig",
    
    "_CorrectorBackend",
    "_PeriodicOrbitCorrectorInterface",
    "_InvariantToriCorrectorInterface",
]