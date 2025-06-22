from typing import Any, TYPE_CHECKING

import numpy as np

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_evaluate_transform, _lie_expansion,
                                   _lie_transform)
from algorithms.center.transforms import (_local2realmodal,
                                          _local2synodic_collinear,
                                          _local2synodic_triangular,
                                          _realmodal2local, solve_complex,
                                          solve_real, substitute_complex,
                                          substitute_real)
from algorithms.poincare.map import _solve_missing_coord
from algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                        decode_multiindex, init_index_tables)
from system.libration.collinear import CollinearPoint
from utils.log_config import logger


if TYPE_CHECKING:
    from algorithms.poincare.base import PoincareMap, poincareMapConfig


class CenterManifold:
    def __init__(self, point: CollinearPoint, max_degree: int):
        self.point = point
        self.max_degree = max_degree

        self.psi, self.clmo = init_index_tables(self.max_degree)
        self.encode_dict_list = _create_encode_dict_from_clmo(self.clmo)

        self._local2synodic = _local2synodic_collinear if isinstance(self.point, CollinearPoint) else _local2synodic_triangular

        self._cache = {}

        self._poincare_map: "PoincareMap" = None

    def __str__(self):
        return f"CenterManifold(point={self.point}, max_degree={self.max_degree})"
    
    def __repr__(self):
        return f"CenterManifold(point={self.point}, max_degree={self.max_degree})"
    
    def cache_get(self, key: tuple) -> Any:
        """
        Get a value from the cache.
        """
        return self._cache.get(key)
    
    def cache_set(self, key: tuple, value: Any):
        """
        Set a value in the cache.
        """
        self._cache[key] = value
    
    def cache_clear(self):
        """
        Clear the cache.
        """
        self._cache.clear()
    
    def compute(self):
        """
        Computes the center manifold in realified coordinates.
        
        Returns
        -------
        List[numpy.ndarray]
            Polynomial representation of the Hamiltonian restricted to the center manifold
            in realified coordinates
        """
        # First check if realified center manifold is already cached
        cm_real = self.cache_get(('hamiltonian', self.max_degree, 'center_manifold_real'))
        if cm_real is not None:
            return [h.copy() for h in cm_real]

        # If not, check if complex center manifold is cached
        cm_complex = self.cache_get(('hamiltonian', self.max_degree, 'center_manifold_complex'))
        if cm_complex is not None:
            # Convert complex to real
            cm_real = substitute_real(cm_complex, self.max_degree, self.psi, self.clmo)
            self.cache_set(('hamiltonian', self.max_degree, 'center_manifold_real'), [h.copy() for h in cm_real])
            return cm_real

        # If complex center manifold is not cached, compute it from scratch
        logger.info(f"Computing center manifold for {type(self.point).__name__}, max_deg={self.max_degree}")
        
        # Build physical Hamiltonian
        poly_phys = self.cache_get(('hamiltonian', self.max_degree, 'physical'))
        if poly_phys is None:
            poly_phys = build_physical_hamiltonian(self.point, self.max_degree)
            self.cache_set(('hamiltonian', self.max_degree, 'physical'), [h.copy() for h in poly_phys])
        else:
            poly_phys = [h.copy() for h in poly_phys]

        # Transform to real normal form
        poly_rn = self.cache_get(('hamiltonian', self.max_degree, 'real_normal'))
        if poly_rn is None:
            poly_rn = _local2realmodal(self.point, poly_phys, self.max_degree, self.psi, self.clmo)
            self.cache_set(('hamiltonian', self.max_degree, 'real_normal'), [h.copy() for h in poly_rn])
        else:
            poly_rn = [h.copy() for h in poly_rn]

        # Transform to complex normal form
        poly_cn = self.cache_get(('hamiltonian', self.max_degree, 'complex_normal'))
        if poly_cn is None:
            poly_cn = substitute_complex(poly_rn, self.max_degree, self.psi, self.clmo)
            self.cache_set(('hamiltonian', self.max_degree, 'complex_normal'), [h.copy() for h in poly_cn])
        else:
            poly_cn = [h.copy() for h in poly_cn]

        # Perform Lie transformation
        poly_trans = self.cache_get(('hamiltonian', self.max_degree, 'normalized'))
        poly_G_total = self.cache_get(('generating_functions', self.max_degree))
        poly_elim_total = self.cache_get(('terms_to_eliminate', self.max_degree))
        
        if poly_trans is None or poly_G_total is None or poly_elim_total is None:
            poly_trans, poly_G_total, poly_elim_total = _lie_transform(self.point, poly_cn, self.psi, self.clmo, self.max_degree)
            self.cache_set(('hamiltonian', self.max_degree, 'normalized'), [h.copy() for h in poly_trans])
            self.cache_set(('generating_functions', self.max_degree), [g.copy() for g in poly_G_total])
            self.cache_set(('terms_to_eliminate', self.max_degree), [e.copy() for e in poly_elim_total])
        else:
            if poly_trans is not None:
                poly_trans = [h.copy() for h in poly_trans]
            if poly_G_total is not None:
                poly_G_total = [g.copy() for g in poly_G_total]
            if poly_elim_total is not None:
                poly_elim_total = [e.copy() for e in poly_elim_total]

        # Restrict to center manifold
        poly_cm_complex = self._restrict_to_center_manifold(poly_trans, tol=1e-14)
        self.cache_set(('hamiltonian', self.max_degree, 'center_manifold_complex'), [h.copy() for h in poly_cm_complex])

        # Convert to real coordinates
        poly_cm_real = substitute_real(poly_cm_complex, self.max_degree, self.psi, self.clmo)
        self.cache_set(('hamiltonian', self.max_degree, 'center_manifold_real'), [h.copy() for h in poly_cm_real])

        logger.info(f"Center manifold computation complete for {type(self.point).__name__}")
        return poly_cm_real

    def _restrict_to_center_manifold(self, poly_H, tol=1e-14):
        """
        Restrict a Hamiltonian to the center manifold by eliminating hyperbolic variables.
        
        Parameters
        ----------
        poly_H : List[numpy.ndarray]
            Polynomial representation of the Hamiltonian in normal form
        tol : float, optional
            Tolerance for considering coefficients as zero, default is 1e-14
            
        Returns
        -------
        List[numpy.ndarray]
            Polynomial representation of the Hamiltonian restricted to the center manifold
            
        Notes
        -----
        The center manifold is obtained by setting the hyperbolic variables (q1, p1)
        to zero. This function filters out all monomials that contain non-zero
        powers of q1 or p1.
        
        In the packed multi-index format, q1 corresponds to k[0] and p1 corresponds to k[3].
        Any term with non-zero exponents for these variables is eliminated.
        
        Additionally, terms with coefficients smaller than the tolerance are set to zero.
        """
        poly_cm = [h.copy() for h in poly_H]
        for deg, coeff_vec in enumerate(poly_cm):
            if coeff_vec.size == 0:
                continue
            for pos, c in enumerate(coeff_vec):
                if abs(c) <= tol:
                    coeff_vec[pos] = 0.0
                    continue
                k = decode_multiindex(pos, deg, self.clmo)
                if k[0] != 0 or k[3] != 0:       # q1 or p1 exponent non-zero
                    coeff_vec[pos] = 0.0
        return poly_cm
    
    def poincare_map(self, energy: float, **kwargs) -> "PoincareMap":
        from algorithms.poincare.base import PoincareMap, poincareMapConfig

        if self._poincare_map is None:

            default_cfg = dict(
                dt=1e-2,
                method="rk",
                integrator_order=4,
                c_omega_heuristic=20.0,
                n_seeds=20,
                n_iter=40,
                seed_axis="q2",
                section_coord="q3",

                compute_on_init=True,
                use_gpu=False,
            )

            default_cfg.update(kwargs)

            cfg = poincareMapConfig(**default_cfg)
            self._poincare_map = PoincareMap(self, energy, cfg)
        return self._poincare_map

    def ic(self, poincare_point: np.ndarray, energy: float, section_coord: str = "q3") -> np.ndarray:
        """
        Convert a single Poincaré section point (in centre-manifold coordinates)
        into full synodic initial conditions.

        The meaning of *poincare_point* depends on which section was used:

        ┌─────────────┬────────────────────────────────────────────┐
        │ section     │ contents of *poincare_point*               │
        ├─────────────┼────────────────────────────────────────────┤
        │ q3 (=0)     │ (q2, p2)                                   │
        │ p3 (=0)     │ (q2, p2)                                   │
        │ q2 (=0)     │ (q3, p3)                                   │
        │ p2 (=0)     │ (q3, p3)                                   │
        └─────────────┴────────────────────────────────────────────┘

        The missing coordinate is solved for on-the-fly using
        ``_solve_missing_coord`` so no information is lost regardless of the
        chosen section.
        """
        logger.info(
            "Converting Poincaré point %s (section=%s) to initial conditions", 
            poincare_point, section_coord,
        )

        # Ensure we have the centre-manifold Hamiltonian and Lie generators.
        poly_cm_real = self.cache_get(("hamiltonian", self.max_degree, "center_manifold_real"))
        if poly_cm_real is None:
            self.compute()
            poly_cm_real = self.cache_get(("hamiltonian", self.max_degree, "center_manifold_real"))

        poly_G_total = self.cache_get(("generating_functions", self.max_degree))
        if poly_G_total is None:
            err = "Generating functions not cached - centre-manifold computation incomplete."
            logger.error(err)
            raise RuntimeError(err)

        # Alias for brevity.
        h0 = float(energy)
        q2 = p2 = q3 = p3 = None  # type: ignore

        if section_coord == "q3":
            # q3 = 0 section → need p3
            q2, p2 = map(float, poincare_point)
            q3 = 0.0
            p3 = _solve_missing_coord(
                "p3", {"q2": q2, "p2": p2}, h0, poly_cm_real, self.clmo
            )
        elif section_coord == "p3":
            # p3 = 0 section → need q3
            q2, p2 = map(float, poincare_point)
            p3 = 0.0
            q3 = _solve_missing_coord(
                "q3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, poly_cm_real, self.clmo
            )
        elif section_coord == "q2":
            # q2 = 0 section → need p2
            q3, p3 = map(float, poincare_point)
            q2 = 0.0
            p2 = _solve_missing_coord(
                "p2", {"q2": 0.0, "q3": q3, "p3": p3}, h0, poly_cm_real, self.clmo
            )
        elif section_coord == "p2":
            # p2 = 0 section → need q2
            q3, p3 = map(float, poincare_point)
            p2 = 0.0
            q2 = _solve_missing_coord(
                "q2", {"p2": 0.0, "q3": q3, "p3": p3}, h0, poly_cm_real, self.clmo
            )
        else:
            raise ValueError(f"Unsupported section_coord '{section_coord}'.")

        # Validate solutions.
        if None in (q2, p2, q3, p3):
            err = "Failed to reconstruct full CM coordinates - root finding did not converge."
            logger.error(err)
            raise RuntimeError(err)

        q2, p2, q3, p3 = float(q2), float(p2), float(q3), float(p3)  # type: ignore

        real_4d_cm = np.array([q2, p2, q3, p3], dtype=np.complex128)

        real_6d_cm = np.zeros(6, dtype=np.complex128)
        real_6d_cm[1] = real_4d_cm[0]  # q2
        real_6d_cm[2] = real_4d_cm[2]  # q3
        real_6d_cm[4] = real_4d_cm[1]  # p2
        real_6d_cm[5] = real_4d_cm[3]  # p3

        complex_6d_cm = solve_complex(real_6d_cm)
        expansions = _lie_expansion(
            poly_G_total, self.max_degree, self.psi, self.clmo, 1e-30,
            inverse=False, sign=1, restrict=False,
        )
        complex_6d = _evaluate_transform(expansions, complex_6d_cm, self.clmo)
        real_6d = solve_real(complex_6d)
        local_6d = _realmodal2local(self.point, real_6d)
        synodic_6d = self._local2synodic(self.point, local_6d)

        logger.info("CM → synodic transformation complete")
        return synodic_6d

    def ic2cm(self) -> np.ndarray:
        """
        TODO: Implement initial conditions to center manifold transformation.
        """
        pass
