from abc import ABC, abstractmethod
from typing import Callable, Literal

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
from hiten.algorithms.poincare.core.events import _SurfaceEvent


class _ReturnMapBackend(ABC):

    def __init__(
        self,
        *,
        dynsys: "_DynamicalSystemProtocol",
        surface: "_SurfaceEvent",
        forward: int = 1,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
        order: int = 8,
        pre_steps: int = 1000,
        refine_steps: int = 3000,
        bracket_dx: float = 1e-10,
        max_expand: int = 500,
    ) -> None:
        self._dynsys = dynsys
        self._surface = surface
        self._forward = 1 if forward >= 0 else -1
        self._method = method
        self._order = int(order)
        self._pre_steps = int(pre_steps)
        self._refine_steps = int(refine_steps)
        self._bracket_dx = float(bracket_dx)
        self._max_expand = int(max_expand)

        self._section_cache = None
        self._grid_cache = None

    # Each backend must implement a *single-step* worker that takes an array
    # of seeds and returns the crossings produced from those seeds. The engine
    # layer is then responsible for looping / caching / parallelism.

    @abstractmethod
    def step_to_section(
        self,
        seeds: "np.ndarray",
        *,
        dt: float = 1e-2,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Propagate *seeds* to the next surface crossing.

        Parameters
        ----------
        seeds
            Array of CM states (shape (m,4)) or full states depending on the
            backend.
        dt
            Integration time-step used by the backend (meaningful for RK).

        Returns
        -------
        points : ndarray, shape (k,2)
            Crossing coordinates in the section plane.
        states : ndarray, shape (k,4) or (k,6)
            Backend-specific state representation at the crossings.
        """

    def _expand_bracket(
        self,
        f: "Callable[[float], float]",
        x0: float,
        *,
        dx0: float,
        grow: float,
        max_expand: int,
        crossing_test: "Callable[[float, float], bool]",
        symmetric: bool = True,
    ) -> tuple[float, float]:
        """Return a tuple ``(a, b)`` bracketing a root of *f*.

        Parameters
        ----------
        f
            Scalar function whose root is being searched for.
        x0
            Reference point around which to start expanding the bracket.
        dx0
            Initial half-width of the trial interval.
        grow
            Multiplicative factor applied to *dx* after every unsuccessful
            iteration.
        max_expand
            Maximum number of expansion attempts before giving up.
        crossing_test
            A 2-argument predicate ``crossing_test(f_prev, f_curr)`` that returns
            *True* when the desired crossing is located inside ``(prev, curr)``.
        symmetric
            If *True* probe both the ``+dx`` and ``-dx`` directions; otherwise
            examine only the positive side.
        """

        f0 = f(x0)

        # If we are already on the section (or very close) return a zero-length
        # bracket so the caller can decide what to do next.
        if abs(f0) < 1e-14:
            return (x0, x0)

        dx = dx0
        for _ in range(max_expand):
            # Probe +dx first (forward propagation).
            xr = x0 + dx
            fr = f(xr)
            if crossing_test(f0, fr):
                return (x0, xr) if x0 < xr else (xr, x0)

            if symmetric:
                xl = x0 - dx
                fl = f(xl)
                if crossing_test(f0, fl):
                    return (xl, x0) if xl < x0 else (x0, xl)

            dx *= grow

        raise RuntimeError("Failed to bracket root.")

    def points2d(self) -> np.ndarray:
        sec = self.compute()
        return sec.points

    def states(self) -> np.ndarray:
        sec = self.compute()
        return getattr(sec, "states", np.empty((0, 0)))

    def __len__(self):
        return self.points2d().shape[0]

