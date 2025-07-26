from typing import Callable, Literal

import numpy as np
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.base import (_DynamicalSystemProtocol,
                                            _propagate_dynsys)
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import (_PlaneEvent, _SectionHit,
                                              _SurfaceEvent)


class _SingleHitBackend(_ReturnMapBackend):
    """Backend that implements the generic surface-of-section crossing search.
    """

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
        super().__init__(
            dynsys=dynsys,
            surface=surface,
            forward=forward,
            method=method,
            order=order,
            pre_steps=pre_steps,
            refine_steps=refine_steps,
            bracket_dx=bracket_dx,
            max_expand=max_expand,
        )

    def step_to_section(
        self,
        seeds: np.ndarray,
        *,
        dt: float = 1e-2,
        t_guess: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find the next crossing for every seed (generic backend).

        Returns plane points & full states arrays.
        """
        pts, states = [], []
        for s in seeds:
            hit = self._cross(s, t_guess=t_guess)
            if hit is not None:
                pts.append(hit.point2d)
                states.append(hit.state.copy())

        if pts:
            return np.asarray(pts, float), np.asarray(states, float)
        return np.empty((0, 2)), np.empty((0, 6))

    def _value_at_time(self, state_ref: np.ndarray, t_ref: float, t_query: float):
        if np.isclose(t_query, t_ref, rtol=3e-10, atol=1e-10):
            return self._surface.value(state_ref)

        sol_seg = _propagate_dynsys(
            self._dynsys,
            state_ref,
            t_ref,
            t_query,
            forward=self._forward,
            steps=self._refine_steps,
            method=self._method,
            order=self._order,
        )
        state_final = sol_seg.states[-1]
        return self._surface.value(state_final)

    def _bracket_root(self, f: Callable[[float], float], x0: float):
        return self._expand_bracket(
            f,
            x0,
            dx0=self._bracket_dx,
            grow=np.sqrt(2),
            max_expand=self._max_expand,
            crossing_test=self._surface.is_crossing,
            symmetric=True,
        )

    def _cross(self, state0: np.ndarray, *, t_guess: float | None = None, t0_offset: float = 0.15):
        t0_z = float(t_guess) if t_guess is not None else (np.pi / 2.0 - t0_offset)

        sol_coarse = _propagate_dynsys(
            self._dynsys,
            state0,
            0.0,
            t0_z,
            forward=self._forward,
            steps=self._pre_steps,
            method=self._method,
            order=self._order,
        )
        state_mid = sol_coarse.states[-1]

        def _g(t: float):
            return self._value_at_time(state_mid, t0_z, t)

        a, b = self._bracket_root(_g, t0_z)

        root_t = root_scalar(_g, bracket=(a, b), method="brentq", xtol=1e-12).root

        sol_final = _propagate_dynsys(
            self._dynsys,
            state_mid,
            t0_z,
            root_t,
            forward=self._forward,
            steps=self._refine_steps,
            method=self._method,
            order=self._order,
        )
        state_cross = sol_final.states[-1].copy()

        # Fallback 2-D projection: first two coordinates
        point2d = state_cross[:2].copy()

        return _SectionHit(time=root_t, state=state_cross, point2d=point2d)


def find_crossing(dynsys, state0, surface, **kwargs):
    be = _SingleHitBackend(dynsys=dynsys, surface=surface, **kwargs)
    return be.step_to_section(np.asarray(state0, float))


def _plane_crossing_factory(coord: str, value: float = 0.0, direction: int | None = None):
    event = _PlaneEvent(coord=coord, value=value, direction=direction)

    def _section_crossing(*, dynsys, x0, forward: int = 1, **kwargs):
        # Ensure the seed state is treated as a full 6-D vector and find a single crossing
        be = _SingleHitBackend(dynsys=dynsys, surface=event, forward=forward)
        hit = be._cross(np.asarray(x0, float))  # compute single crossing
        return hit.time, hit.state

    return _section_crossing

_x_plane_crossing = _plane_crossing_factory("x", 0.0, None)
_y_plane_crossing = _plane_crossing_factory("y", 0.0, None)
_z_plane_crossing = _plane_crossing_factory("z", 0.0, None)
