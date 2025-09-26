"""Adapters supporting center manifold numerics and persistence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal

import numpy as np

from hiten.algorithms.hamiltonian.center._lie import _evaluate_transform
from hiten.algorithms.hamiltonian.transforms import (_coordlocal2realmodal,
                                                     _coordrealmodal2local,
                                                     _local2synodic_collinear,
                                                     _local2synodic_triangular,
                                                     _solve_complex,
                                                     _solve_real,
                                                     _synodic2local_collinear,
                                                     _synodic2local_triangular)
from hiten.algorithms.poincare.centermanifold.backend import \
    _CenterManifoldBackend
from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
from hiten.algorithms.poincare.centermanifold.config import \
    _CenterManifoldMapConfig
from hiten.algorithms.poincare.centermanifold.interfaces import (
    _CenterManifoldInterface, _get_section_interface)
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.types.services.hamiltonian import \
    get_hamiltonian_services
from hiten.utils.io.center import load_center_manifold, save_center_manifold
from hiten.utils.log_config import logger
from hiten.utils.printing import _format_poly_table

if TYPE_CHECKING:
    from hiten.system.center import CenterManifold
    from hiten.system.hamiltonian import Hamiltonian
    from hiten.system.libration.base import LibrationPoint
    from hiten.algorithms.hamiltonian.pipeline import _HamiltonianPipeline


class _CenterManifoldPersistenceService(_PersistenceServiceBase):
    """Handle persistence for center manifold objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda cm, path, **kw: save_center_manifold(cm, Path(path), **kw),
            load_fn=lambda path, **kw: load_center_manifold(Path(path), **kw),
        )


class _CenterManifoldDynamicsService(_DynamicsServiceBase):
    """Provide numerical operations for center manifold computations."""

    def __init__(self, point: "LibrationPoint", degree: int) -> None:
        super().__init__(point)
        self._degree = degree
        self._services = get_hamiltonian_services()
        self._ham_dynamics = self._services.dynamics
        self._ham_conversion = self._services.conversion
        self._ham_pipeline = self._services.pipeline
        self._hamsys = None
        self._configure_point()

    @property
    def point(self) -> "LibrationPoint":
        return self._domain_obj

    @property
    def degree(self) -> int:
        return self._degree

    @degree.setter
    def degree(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("degree must be a positive integer.")
        if value != self._degree:
            self.reset(self.make_key("pipeline", self._degree))
            self.reset(self.make_key("pipeline", value))
            self._degree = value
            self._hamsys = None

    @property
    def pipeline(self) -> _HamiltonianPipeline:
        """Get or create the pipeline for the current point and degree."""
        cache_key = self.make_key("pipeline", self._degree)
        
        def _factory():
            return self._ham_pipeline.get(self._domain_obj, self._degree)
        
        return self.get_or_create(cache_key, _factory)

    def pipeline_for_degree(self, degree: int) -> _HamiltonianPipeline:
        """Get pipeline for a specific degree, changing current degree if needed."""
        if degree != self._degree:
            self.degree = degree
        return self.pipeline

    def get_backend(self, energy: float, section_coord: str, *, forward: int = 1, max_steps: int = 2000, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8, pre_steps: int = 1000, refine_steps: int = 3000, bracket_dx: float = 1e-10, max_expand: int = 500, c_omega_heuristic: float = 20.0) -> _CenterManifoldBackend:
        key = self.make_key(id(self._domain_obj), self._degree, energy, section_coord, forward, max_steps, method, order, pre_steps, refine_steps, bracket_dx, max_expand, c_omega_heuristic)
        
        def _factory() -> _CenterManifoldBackend:

            return _CenterManifoldBackend(
                forward=forward,
                max_steps=max_steps,
                method=method,
                order=order,
                pre_steps=pre_steps,
                refine_steps=refine_steps,
                bracket_dx=bracket_dx,
                max_expand=max_expand,
                c_omega_heuristic=c_omega_heuristic,
            )
        
        return self.get_or_create(key, _factory)

    def clear_caches(self) -> None:
        self.reset()
        try:
            pipeline = self._ham_pipeline.get(self._domain_obj, self._degree)
            if hasattr(pipeline, 'cache_clear'):
                pipeline.cache_clear()
        except Exception:
            pass
        self._hamsys = None

    def format_coefficients(self, ham: "Hamiltonian", degree: int) -> str:
        return _format_poly_table(ham.poly_H, ham._clmo, degree)

    def get_map(self, cm: "CenterManifold", energy: float, **kwargs):

        config_fields = set(_CenterManifoldMapConfig.__dataclass_fields__.keys())
        config_kwargs = {}
        for key, value in kwargs.items():
            if key in config_fields:
                config_kwargs[key] = value
            else:
                raise TypeError(f"'{key}' is not a valid keyword argument for PoincareMap configuration.")

        cfg = _CenterManifoldMapConfig(**config_kwargs)
        config_tuple = tuple(sorted(asdict(cfg).items()))
        cache_key = self.make_key(id(self._domain_obj), self._degree, energy, config_tuple)

        def _factory():
            return CenterManifoldMap.with_default_engine(cm, energy, cfg)

        return self.get_or_create(cache_key, _factory)

    @property
    def hamsys(self):
        if self._hamsys is None:
            self._hamsys = self.pipeline.get_hamiltonian("center_manifold_real").hamsys
        return self._hamsys

    def cm_point_to_synodic(self, cm_point: np.ndarray, *, energy: float | None, section_coord: str = "q3", tol: float = 1e-14) -> np.ndarray:
        cm_point = np.asarray(cm_point)
        if cm_point.size == 2:
            if energy is None:
                raise ValueError(
                    "energy must be specified when converting a 2-D Poincaré point to initial conditions."
                )
            return self._cm_point_to_synodic_from_section(cm_point, float(energy), section_coord, tol)
        if cm_point.size == 4:
            return self._cm_point_to_synodic_4d(cm_point, tol)
        raise ValueError(
            "cm_point must be either a 2- or 4-element vector representing a Poincaré-section point or full"
            " center-manifold coordinates, respectively."
        )

    def synodic_to_cm(self, synodic_6d: np.ndarray, tol: float = 1e-14) -> np.ndarray:
        synodic_6d = np.asarray(synodic_6d, dtype=np.float64).reshape(6)
        local_6d = self._synodic2local(self._domain_obj, synodic_6d, tol)
        real_modal_6d = _coordlocal2realmodal(self._domain_obj, local_6d, tol)
        complex_modal_6d = _solve_complex(real_modal_6d, tol=tol, mix_pairs=self._mix_pairs)

        expansions = self.pipeline.get_lie_expansions(inverse=True, tol=tol)
        complex_pnf_6d = _evaluate_transform(expansions, complex_modal_6d, self.hamsys.clmo)
        real_pnf_6d = _solve_real(complex_pnf_6d, tol=tol, mix_pairs=self._mix_pairs)
        restricted = self._restrict_to_center_manifold(real_pnf_6d)

        return np.array([
            restricted[1],
            restricted[4],
            restricted[2],
            restricted[5],
        ], dtype=np.float64)

    def _cm_point_to_synodic_from_section(
        self,
        poincare_point: np.ndarray,
        energy: float,
        section_coord: str,
        tol: float,
    ) -> np.ndarray:
        sec_if = _get_section_interface(section_coord)

        known_vars: Dict[str, float] = {sec_if.section_coord: 0.0}
        known_vars[sec_if.plane_coords[0]] = float(poincare_point[0])
        known_vars[sec_if.plane_coords[1]] = float(poincare_point[1])

        var_to_solve = {"q3": "p3", "p3": "q3", "q2": "p2", "p2": "q2"}[sec_if.section_coord]

        solved_val = _CenterManifoldInterface.solve_missing_coord(
            var_to_solve,
            known_vars,
            h0=float(energy),
            H_blocks=self.hamsys.poly_H(),
            clmo_table=self.hamsys.clmo_table,
        )

        full_cm_coords = known_vars.copy()
        full_cm_coords[var_to_solve] = solved_val

        if any(v is None for v in full_cm_coords.values()):
            raise RuntimeError("Failed to reconstruct full CM coordinates - root finding did not converge.")

        real_4d_cm = np.array([
            full_cm_coords["q2"],
            full_cm_coords["p2"],
            full_cm_coords["q3"],
            full_cm_coords["p3"],
        ], dtype=np.float64)

        return self._cm_point_to_synodic_4d(real_4d_cm, tol)

    def _cm_point_to_synodic_4d(self, cm_coords_4d: np.ndarray, tol: float) -> np.ndarray:
        cm_coords_4d = np.asarray(cm_coords_4d, dtype=np.float64).reshape(4)

        real_6d_cm = np.zeros(6, dtype=np.complex128)
        real_6d_cm[1] = cm_coords_4d[0]
        real_6d_cm[4] = cm_coords_4d[1]
        real_6d_cm[2] = cm_coords_4d[2]
        real_6d_cm[5] = cm_coords_4d[3]

        complex_6d_cm = _solve_complex(real_6d_cm, tol=tol, mix_pairs=self._mix_pairs)
        expansions = self.pipeline.get_lie_expansions(inverse=False, tol=tol)
        complex_6d = _evaluate_transform(expansions, complex_6d_cm, self.hamsys.clmo_H)

        real_6d = _solve_real(complex_6d, tol=tol, mix_pairs=self._mix_pairs)
        local_6d = _coordrealmodal2local(self._domain_obj, real_6d, tol)
        return self._local2synodic(self._domain_obj, local_6d, tol)

    def _restrict_to_center_manifold(self, coords: np.ndarray) -> np.ndarray:
        coords = coords.copy()
        if coords.ndim == 1:
            coords = coords.astype(np.complex128, copy=False)
        tol = 1e-30
        if coords.ndim == 1:
            arrs = [coords]
        else:
            arrs = coords
        for arr in arrs:
            if arr.size == 0:
                continue
            for idx, value in enumerate(arr):
                if abs(value) <= tol:
                    arr[idx] = 0.0
                    continue
        return coords

    def _configure_point(self) -> None:
        from hiten.system.libration.collinear import CollinearPoint, L3Point
        from hiten.system.libration.triangular import TriangularPoint

        if isinstance(self._domain_obj, CollinearPoint):
            self._local2synodic = _local2synodic_collinear
            self._synodic2local = _synodic2local_collinear
            self._mix_pairs = (1, 2)
            if isinstance(self._domain_obj, L3Point):
                logger.warning(
                    "L3 point has not been verified for centre manifold / normal form computations!"
                )
                raise NotImplementedError("L3 points are not supported yet.")
        elif isinstance(self._domain_obj, TriangularPoint):
            self._local2synodic = _local2synodic_triangular
            self._synodic2local = _synodic2local_triangular
            self._mix_pairs = (0, 1, 2)
            logger.warning(
                "Triangular points have not been verified for centre manifold / normal form computations!"
            )
            raise NotImplementedError("Triangular points are not supported yet.")
        else:
            raise ValueError(f"Unsupported libration point type: {type(self._domain_obj)}")


@dataclass
class _CenterManifoldServices(_ServiceBundleBase):

    domain_obj: "LibrationPoint"
    degree: int
    persistence: _CenterManifoldPersistenceService
    dynamics: _CenterManifoldDynamicsService

    @classmethod
    def default(cls, point: "LibrationPoint", degree: int) -> "_CenterManifoldServices":
        return cls(
            domain_obj=point,
            degree=degree,
            persistence=_CenterManifoldPersistenceService(),
            dynamics=_CenterManifoldDynamicsService(point, degree)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _CenterManifoldDynamicsService) -> "_CenterManifoldServices":
        return cls(
            domain_obj=dynamics._domain_obj,
            degree=dynamics._degree,
            persistence=_CenterManifoldPersistenceService(),
            dynamics=dynamics
        )
