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

To avoid import-time side effects and circular imports when submodules
such as :mod:`hiten.algorithms.types.configs` are imported, the
exports in this module are loaded lazily on first access (PEP 562).
"""

from importlib import import_module
from typing import Any

__all__ = ["RungeKutta", "AdaptiveRK", "ExtendedSymplectic"]


def __getattr__(name: str) -> Any:
    if name in ("RungeKutta", "AdaptiveRK"):
        module = import_module("hiten.algorithms.integrators.rk")
        return getattr(module, name)
    if name == "ExtendedSymplectic":
        module = import_module("hiten.algorithms.integrators.symplectic")
        return getattr(module, name)
    raise AttributeError(f"module 'hiten.algorithms.integrators' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
