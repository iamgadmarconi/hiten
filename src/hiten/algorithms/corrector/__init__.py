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

Main Components
---------------

Ready-to-Use Correctors
~~~~~~~~~~~~~~~~~~~~~~~
:class:`_NewtonOrbitCorrector`
    Complete Newton-Raphson corrector for periodic orbits.

Core Algorithms
~~~~~~~~~~~~~~~
:class:`_NewtonCore`
    Newton-Raphson algorithm with robust linear algebra.
:class:`_ArmijoLineSearch`
    Armijo line search with backtracking.

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~
:class:`_BaseCorrectionConfig`
    Base configuration for correction parameters.
:class:`_OrbitCorrectionConfig`
    Orbit-specific correction configuration.
:class:`_LineSearchConfig`
    Line search parameter configuration.

Domain Interfaces
~~~~~~~~~~~~~~~~~
:class:`_PeriodicOrbitCorrectorInterface`
    Interface for periodic orbit correction.
:class:`_InvariantToriCorrectorInterface`
    Interface for invariant tori correction (placeholder).

Step Control Interfaces
~~~~~~~~~~~~~~~~~~~~~~~
:class:`_StepInterface`
    Abstract base for step-size control strategies.
:class:`_PlainStepInterface`
    Simple Newton steps with optional capping.
:class:`_ArmijoStepInterface`
    Armijo line search step control.

Architecture
------------
The package uses a composition pattern where:

1. **Core Algorithms** provide numerical methods (Newton-Raphson, line search)
2. **Domain Interfaces** handle problem-specific logic (orbit constraints, 
   parameter extraction)
3. **Step Control** manages step sizes and convergence robustness
4. **Configuration Classes** encapsulate algorithm parameters

This design enables flexible combinations through multiple inheritance,
allowing users to create specialized correctors by mixing different
components.

Typical Usage
-------------
Most users will work with the ready-to-use correctors:

>>> from hiten.algorithms.corrector import _NewtonOrbitCorrector
>>> corrector = _NewtonOrbitCorrector()
>>> corrected_orbit = corrector.correct(orbit)

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