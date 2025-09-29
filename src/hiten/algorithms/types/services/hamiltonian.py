"""Adapters for Hamiltonian numerics, conversions, and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Tuple, Union

import numpy as np
from numba.typed import List

from hiten.algorithms.dynamics.hamiltonian import (_HamiltonianSystem,
                                                   create_hamiltonian_system)
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.utils.io.hamiltonian import (load_hamiltonian,
                                        load_lie_generating_function,
                                        save_hamiltonian,
                                        save_lie_generating_function)

if TYPE_CHECKING:
    from hiten.algorithms.hamiltonian.pipeline import HamiltonianPipeline
    from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction

class _HamiltonianPersistenceService(_PersistenceServiceBase):
    """Encapsulate save/load helpers for Hamiltonian objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda ham, path, **kw: save_hamiltonian(ham, Path(path), **kw),
            load_fn=lambda path, **kw: load_hamiltonian(Path(path), **kw),
        )


class _LieGeneratingFunctionPersistenceService(_PersistenceServiceBase):
    """Encapsulate save/load helpers for LieGeneratingFunction objects."""
    
    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda lgf, path, **kw: save_lie_generating_function(lgf, Path(path), **kw),
            load_fn=lambda path, **kw: load_lie_generating_function(Path(path), **kw),
        )


class _HamiltonianDynamicsService(_DynamicsServiceBase):
    """Provide helper utilities for Hamiltonian construction."""

    def __init__(self, domain_obj: "Hamiltonian") -> None:

        self._name = domain_obj.name
        self._degree = domain_obj.degree
        self._ndof = domain_obj.ndof
        self._poly_H = domain_obj.poly_H

        super().__init__(domain_obj)
        self._psi, self._clmo = self.init_tables(self._degree)
        self._encode_dict_list = self.build_encode_dict(self._clmo)
        self._hamsys = self.build_hamsys(self._poly_H, self._degree, self._psi, self._clmo, self._encode_dict_list, self._ndof, self._name)

        self._registry = get_hamiltonian_services()

    @property
    def registry(self) -> _PipelineService:
        return self._registry

    @property
    def ndof(self) -> int:
        return self._ndof
    
    @property
    def degree(self) -> int:
        return self._degree
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def psi(self) -> np.ndarray:
        return self._psi
    
    @property
    def clmo(self) -> List[np.ndarray]:
        return self._clmo

    @property
    def clmoH(self) -> List[np.ndarray]:
        return self.hamsys.clmo

    @property
    def encode_dict_list(self) -> List[dict]:
        return self._encode_dict_list

    @property
    def hamsys(self) -> _HamiltonianSystem:
        return self._hamsys

    @property
    def poly_H(self) -> List[np.ndarray]:
        return self.hamsys.poly_H()
    
    @property
    def jac_H(self) -> List[List[np.ndarray]]:
        return self.hamsys.jac_H

    def evaluate(self, coords: np.ndarray) -> float:
        return _polynomial_evaluate(self.poly_H, coords, self.clmo)

    def init_tables(self, degree: int):
        return _init_index_tables(degree)

    def build_encode_dict(self, clmo):
        return _create_encode_dict_from_clmo(clmo)

    def build_hamsys(self, poly_H, degree: int, psi, clmo, encode_dict, ndof: int, name: str):
        return create_hamiltonian_system(poly_H, degree, psi, clmo, encode_dict, ndof, name)

    def list_registered_forms(self):
        from hiten.algorithms.hamiltonian.pipeline import \
            _CONVERSION_REGISTRY as _GLOBAL_CONVERSIONS
        forms = set()
        for src, dst in _GLOBAL_CONVERSIONS:
            forms.add(src)
            forms.add(dst)
        return forms

    def build_pipeline(self, point, degree: int, conversion: _HamiltonianConversionService) -> "HamiltonianPipeline":
        from hiten.algorithms.hamiltonian.pipeline import HamiltonianPipeline
        return HamiltonianPipeline(point, degree, dynamics=self, conversion=conversion)

    def from_state(self, other: "Hamiltonian", target_cls: type["Hamiltonian"], **kwargs) -> "Hamiltonian":
        """Convert another Hamiltonian to the target class using shared conversion service."""
        conversion = self.registry.conversion
        result = conversion.convert(other, target_cls, **kwargs)
        return target_cls(result.poly_H, result.degree, result.ndof, result.name)

    def to_state(self, target_form: Union[type["Hamiltonian"], str], **kwargs) -> "Hamiltonian":
        """Convert this Hamiltonian to another form using shared conversion service."""
        conversion = self.registry.conversion
        return conversion.convert(self.domain_obj, target_form, **kwargs)

    def register_conversion(self, src: str, dst: str, converter: Callable, required_context: list, default_params: dict) -> None:
        """Register a conversion function in the shared conversion service."""
        self.registry.register_conversion(src, dst, converter, required_context, default_params)


class _LieGeneratingFunctionDynamicsService(_DynamicsServiceBase):
    """Provide helper utilities for Lie generating function construction."""

    def __init__(self, domain_obj: "LieGeneratingFunction") -> None:
        self._name = domain_obj.name
        self._degree = domain_obj.degree
        self._ndof = domain_obj.ndof
        self._poly_G = domain_obj.poly_G
        self._poly_elim = domain_obj.poly_elim

        super().__init__(domain_obj)
        self._psi, self._clmo = self.init_tables(self._degree)
        self._encode_dict_list = self.build_encode_dict(self._clmo)

    @property
    def ndof(self) -> int:
        return self._ndof
    
    @property
    def degree(self) -> int:
        return self._degree
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def psi(self) -> np.ndarray:
        return self._psi
    
    @property
    def clmo(self) -> List[np.ndarray]:
        return self._clmo

    @property
    def encode_dict_list(self) -> List[dict]:
        return self._encode_dict_list

    @property
    def poly_G(self) -> List[np.ndarray]:
        return self._poly_G
    
    @property
    def poly_elim(self) -> List[np.ndarray]:
        return self._poly_elim

    def evaluate(self, coords: np.ndarray) -> float:
        return _polynomial_evaluate(self.poly_G, coords, self.clmo)

    def init_tables(self, degree: int):
        return _init_index_tables(degree)

    def build_encode_dict(self, clmo):
        return _create_encode_dict_from_clmo(clmo)


class _HamiltonianConversionService:
    """Maintain conversion registry and apply transformations."""

    def __init__(self) -> None:
        self._registry: Dict[Tuple[str, str], Tuple[Callable, list, dict]] = {}

    def items(self) -> Iterable[Tuple[Tuple[str, str], Tuple[Callable, list, dict]]]:
        return self._registry.items()

    def register(self, src: str, dst: str, converter: Callable, required_context: list, default_params: dict) -> None:
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


class _HamiltonianPipelineService:
    """Construct and cache `HamiltonianPipeline` instances."""

    def __init__(self, conversion: _HamiltonianConversionService) -> None:
        self._conversion = conversion
        self._pipelines: Dict[int, Dict[int, "HamiltonianPipeline"]] = {}

    def _create_pipeline(self, point, degree: int) -> "HamiltonianPipeline":
        from hiten.algorithms.hamiltonian.pipeline import HamiltonianPipeline
        return HamiltonianPipeline(point, degree)

    def get(self, point, degree: int) -> "HamiltonianPipeline":
        point_key = id(point)
        per_point = self._pipelines.setdefault(point_key, {})
        pipeline = per_point.get(degree)
        if pipeline is None:
            pipeline = self._create_pipeline(point, degree)
            per_point[degree] = pipeline
        return pipeline

    def set(self, point, degree: int) -> "HamiltonianPipeline":
        point_key = id(point)
        pipeline = self._create_pipeline(point, degree)
        self._pipelines.setdefault(point_key, {})[degree] = pipeline
        return pipeline

    def clear(self) -> None:
        self._pipelines.clear()

    def clear_point(self, point) -> None:
        self._pipelines.pop(id(point), None)

class _PipelineService:
    """Registry for shared Hamiltonian services that don't depend on specific instances."""
    
    # Define the conversion registry locally to avoid circular imports
    _CONVERSION_REGISTRY: Dict[Tuple[str, str], Tuple[Callable[..., "Hamiltonian | tuple[Hamiltonian, LieGeneratingFunction]"], list, dict]] = {}
    
    def __init__(self):
        self._conversion = None
        self._pipeline = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of shared services."""
        if not self._initialized:
            self._conversion = _HamiltonianConversionService()
            self._pipeline = _HamiltonianPipelineService(self._conversion)
            
            # Register conversions from the local registry
            for (src, dst), (func, ctx, defaults) in self._CONVERSION_REGISTRY.items():
                if self._conversion.get(src, dst) is None:
                    self._conversion.register(src, dst, func, ctx, defaults)
            
            self._initialized = True

    @property
    def conversion(self) -> _HamiltonianConversionService:
        self._ensure_initialized()
        return self._conversion
    
    @property
    def pipeline(self) -> _HamiltonianPipelineService:
        self._ensure_initialized()
        return self._pipeline
    
    def register_conversion(self, src: str, dst: str, converter: Callable, required_context: list, default_params: dict) -> None:
        """Register a conversion function in the local registry."""
        self._CONVERSION_REGISTRY[(src, dst)] = (converter, required_context, default_params)
        # Also register in the conversion service if it's already initialized
        if self._initialized:
            self._conversion.register(src, dst, converter, required_context, default_params)


class _HamiltonianServices(_ServiceBundleBase):

    def __init__(self, domain_obj: "Hamiltonian", dynamics: _HamiltonianDynamicsService, persistence: _HamiltonianPersistenceService, conversion: _HamiltonianConversionService, pipeline: _HamiltonianPipelineService) -> None:
        super().__init__(domain_obj)
        self.dynamics = dynamics
        self.persistence = persistence
        self.conversion = conversion
        self.pipeline = pipeline

    @classmethod
    def default(cls, domain_obj: "Hamiltonian") -> "_HamiltonianServices":
        dynamics = _HamiltonianDynamicsService(domain_obj)
        
        registry = get_hamiltonian_services()
        shared_conversion = registry.conversion
        shared_pipeline = registry.pipeline
        
        return cls(
            domain_obj=domain_obj,
            dynamics=dynamics,
            persistence=_HamiltonianPersistenceService(),
            conversion=shared_conversion,
            pipeline=shared_pipeline
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _HamiltonianDynamicsService) -> "_HamiltonianServices":
        registry = get_hamiltonian_services()
        shared_conversion = registry.conversion
        shared_pipeline = registry.pipeline
        
        return cls(
            domain_obj=dynamics.domain_obj,
            dynamics=dynamics,
            persistence=_HamiltonianPersistenceService(),
            conversion=shared_conversion,
            pipeline=shared_pipeline
        )


class _LieGeneratingFunctionServices(_ServiceBundleBase):

    def __init__(self, domain_obj: "LieGeneratingFunction", dynamics: _LieGeneratingFunctionDynamicsService, persistence: _LieGeneratingFunctionPersistenceService) -> None:
        super().__init__(domain_obj)
        self.dynamics = dynamics
        self.persistence = persistence

    @classmethod
    def default(cls, domain_obj: "LieGeneratingFunction") -> "_LieGeneratingFunctionServices":
        dynamics = _LieGeneratingFunctionDynamicsService(domain_obj)
        return cls(
            domain_obj=domain_obj,
            dynamics=dynamics,
            persistence=_LieGeneratingFunctionPersistenceService()
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _LieGeneratingFunctionDynamicsService) -> "_LieGeneratingFunctionServices":
        return cls(
            domain_obj=dynamics.domain_obj,
            dynamics=dynamics,
            persistence=_LieGeneratingFunctionPersistenceService(),
        )

_SHARED_REGISTRY = _PipelineService()

def get_hamiltonian_services() -> _PipelineService:
    """Get the global shared services (for backward compatibility)."""
    return _SHARED_REGISTRY
