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
Advanced users can compose components explicitly:

>>> from hiten.algorithms.corrector.backends.newton import _NewtonBackend
>>> from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
>>> from hiten.algorithms.corrector.interfaces import _OrbitCorrectionInterface
>>> from hiten.algorithms.corrector.stepping import make_armijo_stepper
>>> backend = _NewtonBackend(stepper_factory=make_armijo_stepper())
>>> interface = _OrbitCorrectionInterface()
>>> engine = _OrbitCorrectionEngine(backend=backend, interface=interface)
>>> problem = interface.create_problem(orbit=orbit, config=orbit._correction_config)
>>> result = engine.solve(problem)

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
from .backends.ms import _MultipleShootingBackend
from .backends.newton import _NewtonBackend
from .config import (CorrectionConfig,
                     MultipleShootingOrbitCorrectionConfig,
                     OrbitCorrectionConfig)
from .engine import _OrbitCorrectionEngine
from .interfaces import (_MultipleShootingOrbitCorrectionInterface,
                         _OrbitCorrectionInterface)
from .options import OrbitCorrectionOptions, MultipleShootingCorrectionOptions

__all__ = [
    # Backends
    "_CorrectorBackend",
    "_NewtonBackend",
    "_MultipleShootingBackend",
    
    # Configs (compile-time structure)
    "CorrectionConfig",
    "OrbitCorrectionConfig",
    "MultipleShootingOrbitCorrectionConfig",
    
    # Options (runtime tuning)
    "OrbitCorrectionOptions",
    "MultipleShootingCorrectionOptions",
    
    # Interfaces & Engines
    "_OrbitCorrectionInterface",
    "_MultipleShootingOrbitCorrectionInterface",
    "_OrbitCorrectionEngine",
]