"""Adapters supporting persistence and numerics for libration points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

import numpy as np

from hiten.algorithms.linalg.base import StabilityProperties
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.interfaces import _LibrationPointInterface
from hiten.algorithms.linalg.types import _ProblemType, _SystemType
from hiten.algorithms.types.adapters.base import (_CachedDynamicsAdapter,
                                                  _PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.utils.io.libration import (load_libration_point,
                                      load_libration_point_inplace,
                                      save_libration_point)

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint


@dataclass
class _LinearData:
    """
    Container with linearised CR3BP invariants.

    Parameters
    ----------
    mu : float
        Mass ratio mu = m2/(m1+m2) of the primaries (dimensionless).
    point : str
        Identifier of the libration point ('L1', 'L2' or 'L3').
    lambda1 : float | None
        Real hyperbolic eigenvalue lambda1 > 0 associated with the
        saddle behaviour along the centre-saddle subspace (nondimensional units).
    omega1 : float
        First elliptic frequency omega1 > 0 of the centre subspace (nondimensional units).
    omega2 : float
        Second elliptic frequency omega2 > 0 of the centre subspace (nondimensional units).
    omega3: float | None
        Vertical frequency omega3 of the centre subspace (nondimensional units).
    C : numpy.ndarray, shape (6, 6)
        Symplectic change-of-basis matrix such that C^(-1)AC is in real
        Jordan canonical form, with A the Jacobian of the vector
        field evaluated at the libration point.
    Cinv : numpy.ndarray, shape (6, 6)
        Precomputed inverse of C.

    Notes
    -----
    The record is immutable thanks to slots=True; all fields are plain
    numpy.ndarray or scalars so the instance can be safely cached
    and shared among different computations.
    """
    mu: float
    point: str        # 'L1', 'L2', 'L3'
    lambda1: float | None
    omega1: float
    omega2: float
    omega3: float | None
    C: np.ndarray     # 6x6 symplectic transform
    Cinv: np.ndarray  # inverse


class _LibrationPersistenceAdapter(_PersistenceAdapterMixin):
    """Encapsulate libration point IO helpers for testability."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda point, path, **kw: save_libration_point(point, Path(path), **kw),
            load_fn=lambda path, **kw: load_libration_point(Path(path), **kw),
            load_inplace_fn=lambda target, path, **kw: load_libration_point_inplace(target, Path(path), **kw),
        )


class _LibrationDynamicsAdapter(_CachedDynamicsAdapter[StabilityProperties]):
    """Provide stability analysis and geometry helpers for libration points."""

    def __init__(self) -> None:
        super().__init__()
        self._geometry_cache: Dict[Tuple[int, str, Tuple], object] = {}

    def compute_stability(
        self,
        point: "LibrationPoint",
        *,
        delta: float = 1e-6,
        tol: float = 1e-8,
    ) -> StabilityProperties:
        cache_key = (id(point), delta, tol)

        def _factory() -> StabilityProperties:
            config = _EigenDecompositionConfig(
                problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
                system_type=_SystemType.CONTINUOUS,
                delta=delta,
                tol=tol,
            )
            props = StabilityProperties.with_default_engine(config=config)
            interface = _LibrationPointInterface(config=config)
            problem = interface.create_problem(point)
            props.compute(
                problem.A,
                system_type=config.system_type,
                problem_type=config.problem_type,
            )
            return props

        return self._get_or_create(cache_key, _factory)

    def collinear_gamma(self, point: "LibrationPoint") -> float:
        return self._cached_point_value(point, "gamma", point._compute_gamma)

    def collinear_cn(self, point: "LibrationPoint", n: int) -> float:
        return self._cached_point_value(point, "cn", lambda: point._compute_cn(n), args=(n,))

    def collinear_linear_modes(self, point: "LibrationPoint") -> Tuple[float, float, float | None]:
        return self._cached_point_value(point, "linear_modes", point._compute_linear_modes)

    def collinear_normal_form(self, point: "LibrationPoint") -> Tuple[np.ndarray, np.ndarray]:
        return self._cached_point_value(point, "normal_form", point._build_normal_form)

    def triangular_linear_modes(self, point: "LibrationPoint") -> Tuple[float, float, float]:
        return self._cached_point_value(point, "triangular_modes", point._compute_linear_modes)

    def triangular_normal_form(self, point: "LibrationPoint") -> Tuple[np.ndarray, np.ndarray]:
        return self._cached_point_value(point, "triangular_normal_form", point._build_normal_form)

    def reset_point(self, point: "LibrationPoint") -> None:
        pid = id(point)
        stability_keys = [key for key in list(self._cache.keys()) if key[0] == pid]
        for key in stability_keys:
            self._cache.pop(key, None)

        geometry_keys = [key for key in list(self._geometry_cache.keys()) if key[0] == pid]
        for key in geometry_keys:
            self._geometry_cache.pop(key, None)

    def _cached_point_value(
        self,
        point: "LibrationPoint",
        label: str,
        factory,
        *,
        args: Tuple = (),
    ):
        key = (id(point), label, args)
        if key not in self._geometry_cache:
            self._geometry_cache[key] = factory()
        return self._geometry_cache[key]


@dataclass
class _LibrationServices(_ServiceBundleBase):
    persistence: _LibrationPersistenceAdapter
    dynamics: _LibrationDynamicsAdapter

    @classmethod
    def default(cls) -> "_LibrationServices":
        return cls(
            persistence=_LibrationPersistenceAdapter(),
            dynamics=_LibrationDynamicsAdapter(),
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _LibrationDynamicsAdapter) -> "_LibrationServices":
        return cls(
            persistence=_LibrationPersistenceAdapter(),
            dynamics=dynamics,
        )
