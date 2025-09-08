"""
hiten.algorithms.corrector
==========================

The `hiten.algorithms.corrector` package provides robust iterative correction
algorithms for solving nonlinear systems arising in dynamical systems analysis.
These algorithms are essential for refining approximate solutions to high
precision, particularly for periodic orbits, invariant manifolds, and other
dynamical structures in the Circular Restricted Three-Body Problem (CR3BP).

The package implements a modular architecture that separates algorithmic
components from domain-specific logic, enabling flexible combinations of
different correction strategies with various problem types.

Examples
-------------
Most users will work with the ready-to-use correctors:

>>> from hiten.algorithms.corrector import _NewtonOrbitCorrector
>>> corrector = _NewtonOrbitCorrector()
>>> x_corr, t_half = corrector.correct(orbit)

Advanced users can create custom correctors by combining components:

>>> from hiten.algorithms.corrector import (_NewtonCore, 
...                                        _PeriodicOrbitCorrectorInterface)
>>> class CustomCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
...     pass

------------

All algorithms use nondimensional units consistent with the underlying
dynamical system and are designed for high-precision applications in
astrodynamics and mission design.

See Also
--------
:mod:`hiten.system.orbits`
    Orbit classes that can be corrected using these algorithms.
:mod:`hiten.algorithms.continuation`
    Continuation algorithms that use correction for family generation.
"""

# Step control interfaces
from ._step_interface import (_ArmijoStepInterface, _PlainStepInterface,
                              _StepInterface, _Stepper)
# Configuration classes
from .base import _BaseCorrectionConfig, _Corrector
# Ready-to-use correctors
from .correctors import _NewtonOrbitCorrector
from .interfaces import (_InvariantToriCorrectorInterface,
                         _OrbitCorrectionConfig,
                         _PeriodicOrbitCorrectorInterface)
from .line import _ArmijoLineSearch, _LineSearchConfig
# Core algorithms
from .newton import _NewtonCore

# Public API - expose main correction classes
__all__ = [
    # Ready-to-use correctors
    "_NewtonOrbitCorrector",
    
    # Core algorithms
    "_NewtonCore",
    "_ArmijoLineSearch",
    
    # Configuration classes
    "_BaseCorrectionConfig",
    "_OrbitCorrectionConfig", 
    "_LineSearchConfig",
    
    # Domain interfaces
    "_Corrector",
    "_PeriodicOrbitCorrectorInterface",
    "_InvariantToriCorrectorInterface",
    
    # Step control interfaces
    "_Stepper",
    "_StepInterface",
    "_PlainStepInterface",
    "_ArmijoStepInterface",
]