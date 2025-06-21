from typing import Callable

import numpy as np
from config import FASTMATH

from algorithms.dynamics.base import _DynamicalSystem


class RHSSystem(_DynamicalSystem):
    def __init__(self, rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
        """Wrap an arbitrary RHS into a _DynamicalSystem instance.

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

            self._rhs_compiled = numba.njit(cache=True, fastmath=FASTMATH)(rhs_func)  # type: ignore[arg-type]

        self.name = name
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        return self._rhs_compiled
    
    def __repr__(self) -> str:
        return f"RHSSystem(name='{self.name}', dim={self.dim})"


def create_rhs_system(rhs_func: Callable[[float, np.ndarray], np.ndarray], dim: int, name: str = "Generic RHS"):
    return RHSSystem(rhs_func, dim, name)

