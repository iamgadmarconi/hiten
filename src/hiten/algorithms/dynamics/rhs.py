r"""
dynamics.rhs
============

Lightweight wrappers for arbitrary right-hand side functions.

This module offers a thin adapter class :pyclass:`RHSSystem` that converts a
plain Python callable representing the ODE :math:`\dot y = f(t, y)` into an
object compatible with the internal :pyclass:`~hiten.algorithms.dynamics.base._DynamicalSystem`
interface.  The adapter takes care of JIT-compiling the callable with
:pyfunc:`numba.njit` (unless it is already a Numba :pyclass:`~numba.core.registry.CPUDispatcher`) 
so that it can be invoked from nopython kernels without performance penalties.
"""

from typing import Callable

import numpy as np
from hiten.utils.config import FASTMATH

from hiten.algorithms.dynamics.base import _DynamicalSystem


class RHSSystem(_DynamicalSystem):
    r"""
    Lightweight wrapper around an RHS callable.

    Parameters
    ----------
    rhs_func : Callable[[float, numpy.ndarray], numpy.ndarray]
        Callable implementing the dynamical system :math:`\dot y = f(t, y)`.
        The signature must be ``(t, y)`` where *t* is time (float) and *y* is
        the state vector (one-dimensional :pyclass:`numpy.ndarray`).
    dim : int
        Dimension :math:`n` of the state space.
    name : str, optional
        Human-readable identifier for the system, used only for
        :pyfunc:`__repr__`.  Defaults to ``"Generic RHS"``.

    Attributes
    ----------
    dim : int
        Inherited from :pyclass:`_DynamicalSystem`.  Equal to *dim*.
    name : str
        Identifier provided at construction time.
    rhs : Callable[[float, numpy.ndarray], numpy.ndarray]
        JIT-compiled RHS callable.  Can be used inside :pyfunc:`numba.njit`
        functions without falling back to object mode.

    Notes
    -----
    * If *rhs_func* is already a Numba :pyclass:`CPUDispatcher`, it is reused
      directly.  Otherwise it is compiled with :pyfunc:`numba.njit` using the
      global fast-math setting :pydata:`hiten.utils.config.FASTMATH`.
    * Any error raised by the parent constructor (e.g., if *dim* is not
      positive) propagates unchanged.
    """

    def __init__(self, rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
        r"""
        Wrap an arbitrary RHS into a _DynamicalSystem instance.

        The supplied *rhs_func* is automatically JIT-compiled (if it is not a
        Numba dispatcher already) so that it can be called from inside
        `@njit` kernels without falling back to object mode.
        """

        super().__init__(dim)

        # Detect whether the function is already a Numba dispatcher.  If it is
        # not, compile it with *nopython* mode so that the integrator kernels
        # can invoke it directly.
        try:
            from numba.core.registry import CPUDispatcher  # type: ignore

            is_dispatcher = isinstance(rhs_func, CPUDispatcher)
        except Exception:  # pragma: no cover – very old Numba versions
            is_dispatcher = False

        if is_dispatcher:
            self._rhs_compiled = rhs_func  # Already compiled
        else:
            # Compile with fastmath setting consistent with global config.
            import numba

            self._rhs_compiled = numba.njit(cache=False, fastmath=FASTMATH)(rhs_func)  # type: ignore[arg-type]

        self.name = name
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        r"""
        Return the compiled RHS callable.

        Returns
        -------
        Callable[[float, numpy.ndarray], numpy.ndarray]
            A Numba-compiled function with the same semantics as the original
            *rhs_func* passed at construction.
        """
        return self._rhs_compiled
    
    def __repr__(self) -> str:
        return f"RHSSystem(name='{self.name}', dim={self.dim})"


def create_rhs_system(rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
    r"""
    Factory helper that mirrors :pyfunc:`RHSSystem`.

    This convenience function exists mainly to enable a functional style
    when the object-oriented interface of :pyclass:`RHSSystem` is not needed.

    Parameters
    ----------
    rhs_func, dim, name
        Forwarded verbatim to the :pyfunc:`RHSSystem` constructor.

    Returns
    -------
    RHSSystem
        Instance wrapping *rhs_func*.
    """
    return RHSSystem(rhs_func, dim, name)


