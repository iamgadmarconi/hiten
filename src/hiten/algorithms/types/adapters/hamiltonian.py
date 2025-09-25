"""Adapters for Hamiltonian numerics, conversions, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.hamiltonian.pipeline import _HamiltonianPipeline
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.types.adapters.base import (_CachedDynamicsAdapter,
                                                  _PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.utils.io.hamiltonian import load_hamiltonian, save_hamiltonian


class _HamiltonianPersistenceAdapter(_PersistenceAdapterMixin):
    """Encapsulate save/load helpers for Hamiltonian objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda ham, path, **kw: save_hamiltonian(ham, Path(path), **kw),
            load_fn=lambda path, **kw: load_hamiltonian(Path(path), **kw),
        )


class _HamiltonianDynamicsAdapter(_CachedDynamicsAdapter):
    """Provide helper utilities for Hamiltonian construction."""

    def init_tables(self, degree: int):
        return _init_index_tables(degree)

    def build_encode_dict(self, clmo):
        return _create_encode_dict_from_clmo(clmo)

    def build_hamsys(
        self,
        poly_H,
        degree: int,
        psi,
        clmo,
        encode_dict,
        ndof: int,
        name: str,
    ):
        return create_hamiltonian_system(poly_H, degree, psi, clmo, encode_dict, ndof, name)

    def list_registered_forms(self):
        return set()

    def build_pipeline(self, point, degree: int, conversion: _HamiltonianConversionAdapter) -> _HamiltonianPipeline:
        return _HamiltonianPipeline(point, degree, dynamics=self, conversion=conversion)


class _HamiltonianConversionAdapter:
    """Maintain conversion registry and apply transformations."""

    def __init__(self) -> None:
        self._registry: Dict[Tuple[str, str], Tuple[Callable, list, dict]] = {}

    def items(self) -> Iterable[Tuple[Tuple[str, str], Tuple[Callable, list, dict]]]:
        return self._registry.items()

    def register(
        self,
        src: str,
        dst: str,
        converter: Callable,
        required_context: list,
        default_params: dict,
    ) -> None:
        self._registry[(src, dst)] = (converter, required_context, default_params)

    def get(self, src: str, dst: str):
        return self._registry.get((src, dst))

    def convert(self, ham, target_form, **kwargs):
        if isinstance(target_form, str):
            target_name = target_form
            class _Temp(ham.__class__):
                name = target_name
            target_cls = _Temp
        else:
            target_name = target_form.name
            target_cls = target_form

        entry = self.get(ham.name, target_name)
        if entry is not None:
            converter, required_context, default_params = entry
            missing = [key for key in required_context if key not in kwargs]
            if missing:
                raise ValueError(
                    f"Missing required context for conversion {ham.name} -> {target_name}: {missing}"
                )
            final_kwargs = {**default_params, **kwargs}
            return converter(ham, **final_kwargs)

        if isinstance(target_form, type):
            return target_cls.from_state(ham, **kwargs)

        raise NotImplementedError(f"No conversion path from {ham.name} to {target_name}")

    def available_targets(self, src: str) -> Iterable[str]:
        for (source, dst) in self._registry:
            if source == src:
                yield dst

    def all_forms(self) -> Iterable[str]:
        forms = set()
        for src, dst in self._registry:
            forms.add(src)
            forms.add(dst)
        return forms


class _HamiltonianPipelineAdapter:
    """Construct and cache `_HamiltonianPipeline` instances."""

    def __init__(
        self,
        dynamics: _HamiltonianDynamicsAdapter,
        conversion: _HamiltonianConversionAdapter,
    ) -> None:
        self._dynamics = dynamics
        self._conversion = conversion
        self._pipelines: Dict[int, Dict[int, _HamiltonianPipeline]] = {}

    def _create_pipeline(self, point, degree: int) -> _HamiltonianPipeline:
        return _HamiltonianPipeline(point, degree, dynamics=self._dynamics, conversion=self._conversion)

    def get(self, point, degree: int) -> _HamiltonianPipeline:
        point_key = id(point)
        per_point = self._pipelines.setdefault(point_key, {})
        pipeline = per_point.get(degree)
        if pipeline is None:
            pipeline = self._create_pipeline(point, degree)
            per_point[degree] = pipeline
        return pipeline

    def set(self, point, degree: int) -> _HamiltonianPipeline:
        point_key = id(point)
        pipeline = self._create_pipeline(point, degree)
        self._pipelines.setdefault(point_key, {})[degree] = pipeline
        return pipeline

    def clear(self) -> None:
        self._pipelines.clear()

    def clear_point(self, point) -> None:
        self._pipelines.pop(id(point), None)


@dataclass
class _HamiltonianServices(_ServiceBundleBase):
    dynamics: _HamiltonianDynamicsAdapter
    persistence: _HamiltonianPersistenceAdapter
    conversion: _HamiltonianConversionAdapter
    pipeline: _HamiltonianPipelineAdapter


_DEFAULT_HAMILTONIAN_SERVICES: _HamiltonianServices | None = None


def get_hamiltonian_services() -> _HamiltonianServices:
    global _DEFAULT_HAMILTONIAN_SERVICES
    if _DEFAULT_HAMILTONIAN_SERVICES is None:
        dynamics = _HamiltonianDynamicsAdapter()
        conversion = _HamiltonianConversionAdapter()
        from hiten.algorithms.hamiltonian.pipeline import _CONVERSION_REGISTRY as _GLOBAL_CONVERSIONS

        for (src, dst), (func, ctx, defaults) in _GLOBAL_CONVERSIONS.items():
            if conversion.get(src, dst) is None:
                conversion.register(src, dst, func, ctx, defaults)

        _DEFAULT_HAMILTONIAN_SERVICES = _HamiltonianServices(
            dynamics=dynamics,
            persistence=_HamiltonianPersistenceAdapter(),
            conversion=conversion,
            pipeline=_HamiltonianPipelineAdapter(dynamics, conversion),
        )
    return _DEFAULT_HAMILTONIAN_SERVICES
