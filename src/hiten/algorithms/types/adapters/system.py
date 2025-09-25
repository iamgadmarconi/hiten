"""Adapters coordinating dynamics, libration, and persistence services for systems.

This module supplies the concrete adapter implementations used by the
`hiten.system` package to bridge user-facing classes with the algorithms
layer. Each adapter concentrates the knowledge required to instantiate
backends, interfaces, and engines while exposing a slim API back to the
system module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, TYPE_CHECKING

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol
from hiten.algorithms.dynamics.rtbp import (jacobian_dynsys, rtbp_dynsys,
                                            variational_dynsys)
from hiten.algorithms.integrators.base import _Solution
from hiten.algorithms.types.adapters.base import (_CachedDynamicsAdapter,
                                                  _PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.algorithms.utils.coordinates import _get_mass_parameter
from hiten.utils.io.system import load_system, load_system_inplace, save_system


if TYPE_CHECKING:
    from hiten.system.body import Body



class _SystemPersistenceAdapter(_PersistenceAdapterMixin):
    """Thin adapter around system IO helpers for testability and indirection."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda system, path, **kw: save_system(system, Path(path), **kw),
            load_fn=lambda path, **kw: load_system(Path(path), **kw),
            load_inplace_fn=lambda target, path, **kw: load_system_inplace(target, Path(path), **kw),
        )


class _SystemsDynamicsAdapter(_CachedDynamicsAdapter[_DynamicalSystemProtocol]):
    """Lazily construct and cache dynamical system backends for a CR3BP system."""

    def __init__(self, primary: "Body", secondary: "Body", distance: float) -> None:
        super().__init__()
        self._primary = primary
        self._secondary = secondary
        self._distance = distance
        self._mu = _get_mass_parameter(primary.mass, secondary.mass)

    @property
    def dynsys(self) -> _DynamicalSystemProtocol:
        return self._get_or_create(
            "dynsys",
            lambda: rtbp_dynsys(self._mu, name=_make_dynsys_name(self._primary, self._secondary, "rtbp")),
        )

    @property
    def variational(self) -> _DynamicalSystemProtocol:
        return self._get_or_create(
            "variational",
            lambda: variational_dynsys(
                self._mu,
                name=_make_dynsys_name(self._primary, self._secondary, "variational"),
            ),
        )

    @property
    def jacobian(self) -> _DynamicalSystemProtocol:
        return self._get_or_create(
            "jacobian",
            lambda: jacobian_dynsys(
                self._mu,
                name=_make_dynsys_name(self._primary, self._secondary, "jacobian"),
            ),
        )

    def reset(self) -> None:
        self.reset_cache()

    def propagate(
        self,
        state0: Sequence[float],
        *,
        tf: float,
        steps: int,
        method: str,
        order: int,
        forward: int,
        extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> _Solution:
        """Delegate propagation to the shared CR3BP integrator."""

        kwargs = extra_kwargs or {}
        return _propagate_dynsys(
            dynsys=self.dynsys,
            state0=state0,
            t0=0.0,
            tf=tf,
            forward=forward,
            steps=steps,
            method=method,
            order=order,
            **kwargs,
        )


@dataclass
class _SystemServices(_ServiceBundleBase):
    primary: "Body"
    secondary: "Body"
    distance: float
    mu: float
    dynamics: _SystemsDynamicsAdapter
    persistence: _SystemPersistenceAdapter

    @classmethod
    def from_bodies(cls, primary: "Body", secondary: "Body", distance: float) -> "_SystemServices":
        mu = _get_mass_parameter(primary.mass, secondary.mass)
        dynamics = _SystemsDynamicsAdapter(primary, secondary, distance)
        persistence = _SystemPersistenceAdapter()
        return cls(
            primary=primary,
            secondary=secondary,
            distance=distance,
            mu=mu,
            dynamics=dynamics,
            persistence=persistence,
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "_SystemServices":
        system = load_system(file_path)
        return cls.from_bodies(system.primary, system.secondary, system.distance)


def _make_dynsys_name(primary: "Body", secondary: "Body", suffix: str) -> str:
    return f"{primary.name}_{secondary.name}_{suffix}"
