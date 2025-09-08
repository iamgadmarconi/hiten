"""Numerical integrators for dynamical systems.

This package provides a collection of numerical integrators for solving
ordinary differential equations that arise in the Circular Restricted
Three-Body Problem (CR3BP). It includes:

- Explicit Runge-Kutta methods (fixed and adaptive step-size)
- High-order symplectic integrators for Hamiltonian systems

The main user-facing classes are the factories:
- :class:`~hiten.algorithms.integrators.rk.RungeKutta` for fixed-step methods
- :class:`~hiten.algorithms.integrators.rk.AdaptiveRK` for adaptive step-size methods
- :class:`~hiten.algorithms.integrators.symplectic.ExtendedSymplectic` for symplectic integration
"""

from .rk import AdaptiveRK, RungeKutta
from .symplectic import ExtendedSymplectic

__all__ = ["RungeKutta", "AdaptiveRK", "ExtendedSymplectic"]
