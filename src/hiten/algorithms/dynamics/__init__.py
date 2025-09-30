"""Provide a dynamical systems framework for astrodynamics and orbital mechanics.

This package provides a comprehensive framework for defining, analyzing, and
integrating dynamical systems with emphasis on applications in astrodynamics,
particularly the Circular Restricted Three-Body Problem (CR3BP).

Examples
--------
Create and integrate a CR3BP system:

>>> from hiten.algorithms.dynamics import rtbp_dynsys
>>> import numpy as np
>>> from scipy.integrate import solve_ivp
>>> 
>>> # Earth-Moon system
>>> system = rtbp_dynsys(mu=0.01215, name="Earth-Moon")
>>> initial_state = np.array([0.8, 0, 0, 0, 0.1, 0])
>>> sol = solve_ivp(system.rhs, [0, 10], initial_state, dense_output=True)

Create a generic dynamical system:

>>> from hiten.algorithms.dynamics import create_rhs_system
>>> 
>>> def harmonic_oscillator(t, y):
...     return np.array([y[1], -y[0]])
>>> 
>>> system = create_rhs_system(harmonic_oscillator, dim=2, name="Harmonic Oscillator")

See Also
--------
:mod:`~hiten.algorithms.integrators` : Numerical integration methods
:mod:`~hiten.algorithms.polynomial` : Polynomial operations for Hamiltonian systems
:mod:`~hiten.system` : High-level system definitions and orbital mechanics
"""

from .base import (_DirectedSystem, _DynamicalSystem, _propagate_dynsys,
                   _validate_initial_state)
from .hamiltonian import create_hamiltonian_system
from .protocols import _DynamicalSystemProtocol, _HamiltonianSystemProtocol
from .rhs import create_rhs_system
from .rtbp import jacobian_dynsys, rtbp_dynsys, variational_dynsys

__all__ = [
    "_DynamicalSystem",
    "_DirectedSystem", 
    "_DynamicalSystemProtocol",
    "_HamiltonianSystemProtocol",
    "_propagate_dynsys",
    "_validate_initial_state",
    "rtbp_dynsys",
    "jacobian_dynsys",
    "variational_dynsys",
    "create_rhs_system",
    "create_hamiltonian_system",
]
