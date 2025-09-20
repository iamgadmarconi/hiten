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

Analyze energy and stability:

>>> from hiten.algorithms.dynamics.utils import crtbp_energy, eigenvalue_decomposition
>>> 
>>> # Compute energy
>>> energy = crtbp_energy(state, mu=0.01215)
>>> 
>>> # Stability analysis
>>> jacobian = compute_jacobian(state, mu)  # User-provided function
>>> stable_vals, unstable_vals, center_vals, Ws, Wu, Wc = eigenvalue_decomposition(jacobian)

See Also
--------
:mod:`~hiten.algorithms.integrators` : Numerical integration methods
:mod:`~hiten.algorithms.polynomial` : Polynomial operations for Hamiltonian systems
:mod:`~hiten.system` : High-level system definitions and orbital mechanics

References
----------
Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*.
Academic Press.

Koon, W. S.; Lo, M. W.; Marsden, J. E.; Ross, S. D. (2011).
*Dynamical Systems, the Three-Body Problem and Space Mission Design*.
Caltech.
"""

from .base import (_DirectedSystem, _DynamicalSystem, _propagate_dynsys,
                   _validate_initial_state)
from .hamiltonian import create_hamiltonian_system
from .protocols import _DynamicalSystemProtocol, _HamiltonianSystemProtocol
from .rhs import create_rhs_system
from .rtbp import jacobian_dynsys, rtbp_dynsys, variational_dynsys
from .utils.energy import (crtbp_energy, effective_potential, energy_to_jacobi,
                           gravitational_potential, hill_region,
                           jacobi_to_energy, kinetic_energy, primary_distance,
                           pseudo_potential_at_point, secondary_distance)
from .utils.linalg import eigenvalue_decomposition

__all__ = [
    # Core framework
    "_DynamicalSystem",
    "_DirectedSystem", 
    "_DynamicalSystemProtocol",
    "_HamiltonianSystemProtocol",
    "_propagate_dynsys",
    "_validate_initial_state",
    
    # CR3BP systems
    "rtbp_dynsys",
    "jacobian_dynsys",
    "variational_dynsys",
    
    # Generic systems
    "create_rhs_system",
    
    # Hamiltonian systems
    "create_hamiltonian_system",
    
    # Energy and potential utilities
    "crtbp_energy",
    "energy_to_jacobi",
    "jacobi_to_energy",
    "kinetic_energy", 
    "effective_potential",
    "pseudo_potential_at_point",
    "gravitational_potential",
    "primary_distance",
    "secondary_distance",
    "hill_region",
    
    # Linear algebra utilities
    "eigenvalue_decomposition",
]
