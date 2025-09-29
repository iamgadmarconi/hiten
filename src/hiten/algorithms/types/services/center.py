"""Adapters supporting center manifold numerics and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
from hiten.algorithms.types.services import get_hamiltonian_services
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.system.maps.center import CenterManifoldMap
from hiten.utils.io.center import load_center_manifold, save_center_manifold
from hiten.utils.log_config import logger
from hiten.utils.printing import _format_poly_table

if TYPE_CHECKING:
    from hiten.algorithms.hamiltonian.pipeline import HamiltonianPipeline
    from hiten.system.center import CenterManifold
    from hiten.system.hamiltonian import Hamiltonian
    from hiten.system.libration.base import LibrationPoint


class _CenterManifoldPersistenceService(_PersistenceServiceBase):
    """Handle persistence for center manifold objects."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda cm, path, **kw: save_center_manifold(cm, Path(path), **kw),
            load_fn=lambda path, **kw: load_center_manifold(Path(path), **kw),
        )


class _CenterManifoldDynamicsService(_DynamicsServiceBase):
    """Provide numerical operations for center manifold computations."""

    def __init__(self, domain_obj: "CenterManifold") -> None:
        self._point = domain_obj._point
        self._degree = domain_obj._max_degree

        super().__init__(domain_obj)
        self._services = get_hamiltonian_services()
        self._ham_conversion = self._services.conversion
        self._ham_pipeline = self._services.pipeline
        self._hamsys = None
        self._configure_point()

    @property
    def point(self) -> "LibrationPoint":
        return self._point

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
    def pipeline(self) -> HamiltonianPipeline:
        """Get or create the pipeline for the current point and degree."""
        cache_key = self.make_key("pipeline", self.degree)
        
        def _factory():
            return self._ham_pipeline.get(self.point, self.degree)
        
        return self.get_or_create(cache_key, _factory)

    def hamiltonian(self, degree: int) -> "Hamiltonian":

        cache_key = self.make_key("hamiltonian", degree)

        def _factory():
            return self.pipeline_for_degree(degree).get_hamiltonian("center_manifold_real")
        
        return self.get_or_create(cache_key, _factory)

    def pipeline_for_degree(self, degree: int) -> HamiltonianPipeline:
        """Get pipeline for a specific degree, changing current degree if needed."""
        if degree != self._degree:
            self.degree = degree
        return self.pipeline

    def clear_caches(self) -> None:
        self.reset()
        try:
            pipeline = self._ham_pipeline.get(self.domain_obj, self._degree)
            if hasattr(pipeline, 'cache_clear'):
                pipeline.cache_clear()
        except Exception:
            pass
        self._hamsys = None

    def format_coefficients(self, ham: "Hamiltonian", degree: int) -> str:
        return _format_poly_table(ham.poly_H, ham._clmo, degree)

    def get_map(self, energy: float):

        cache_key = self.make_key(id(self.domain_obj), self._degree, energy)

        def _factory():
            return CenterManifoldMap(self.domain_obj, energy)

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
            return self._cm_point_to_synodic_from_section(float(energy), cm_point, section_coord, tol)
        if cm_point.size == 4:
            return self._cm_point_to_synodic_4d(cm_point, tol)
        raise ValueError(
            "cm_point must be either a 2- or 4-element vector representing a Poincaré-section point or full"
            " center-manifold coordinates, respectively."
        )

    def synodic_to_cm(self, synodic_6d: np.ndarray, tol: float = 1e-14) -> np.ndarray:
        synodic_6d = np.asarray(synodic_6d, dtype=np.float64).reshape(6)
        local_6d = self._synodic2local(self._point, synodic_6d, tol)
        real_modal_6d = _coordlocal2realmodal(self._point, local_6d, tol)
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

    def _cm_point_to_synodic_from_section(self, energy: float, poincare_point: np.ndarray, section_coord: str, tol: float) -> np.ndarray:
        real_4d_cm = self.get_map(energy).dynamics._to_real_4d_cm(poincare_point, section_coord)
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
        local_6d = _coordrealmodal2local(self._point, real_6d, tol)
        return self._local2synodic(self._point, local_6d, tol)

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

        if isinstance(self._point, CollinearPoint):
            self._local2synodic = _local2synodic_collinear
            self._synodic2local = _synodic2local_collinear
            self._mix_pairs = (1, 2)
            if isinstance(self._point, L3Point):
                logger.warning(
                    "L3 point has not been verified for centre manifold / normal form computations!"
                )
                raise NotImplementedError("L3 points are not supported yet.")
        elif isinstance(self._point, TriangularPoint):
            self._local2synodic = _local2synodic_triangular
            self._synodic2local = _synodic2local_triangular
            self._mix_pairs = (0, 1, 2)
            logger.warning(
                "Triangular points have not been verified for centre manifold / normal form computations!"
            )
            raise NotImplementedError("Triangular points are not supported yet.")
        else:
            raise ValueError(f"Unsupported libration point type: {type(self._point)}")


class _CenterManifoldServices(_ServiceBundleBase):

    def __init__(self, domain_obj: "CenterManifold", persistence: _CenterManifoldPersistenceService, dynamics: _CenterManifoldDynamicsService) -> None:
        super().__init__(domain_obj)
        self.degree = domain_obj._max_degree
        self.persistence = persistence
        self.dynamics = dynamics

    @classmethod
    def default(cls, domain_obj: "CenterManifold") -> "_CenterManifoldServices":
        return cls(
            domain_obj=domain_obj,
            persistence=_CenterManifoldPersistenceService(),
            dynamics=_CenterManifoldDynamicsService(domain_obj)
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _CenterManifoldDynamicsService) -> "_CenterManifoldServices":
        return cls(
            domain_obj=dynamics.domain_obj,
            persistence=_CenterManifoldPersistenceService(),
            dynamics=dynamics
        )
