"""Provide utility functions for dynamical systems analysis.

This submodule provides specialized utility functions for analyzing dynamical
systems, with particular focus on the Circular Restricted Three-Body Problem
(CR3BP) and general stability analysis.

Examples
--------
Energy analysis:

>>> from hiten.algorithms.dynamics.utils import crtbp_energy, hill_region
>>> import numpy as np
>>> 
>>> # Compute system energy
>>> state = np.array([0.8, 0, 0, 0, 0.1, 0])
>>> energy = crtbp_energy(state, mu=0.01215)
>>> 
>>> # Generate Hill region for visualization
>>> X, Y, Z = hill_region(mu=0.01215, C=-1.5, n_grid=200)

Stability analysis:

>>> from hiten.algorithms.dynamics.utils import eigenvalue_decomposition
>>> 
>>> # Classify eigenvalue spectrum
>>> A = compute_system_jacobian()  # User function
>>> stable, unstable, center, Ws, Wu, Wc = eigenvalue_decomposition(A)

See Also
--------
:mod:`~hiten.algorithms.dynamics` : Main dynamical systems framework
:mod:`~hiten.system` : High-level system definitions
"""

from .energy import (crtbp_energy, effective_potential, energy_to_jacobi,
                     gravitational_potential, hill_region, jacobi_to_energy,
                     kinetic_energy, primary_distance,
                     pseudo_potential_at_point, secondary_distance)

__all__ = [
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
]
