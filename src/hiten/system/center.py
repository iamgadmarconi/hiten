r"""
center.base
===========

High-level utilities for computing a polynomial normal form of the centre
manifold around a collinear libration point of the spatial circular
restricted three body problem (CRTBP).

All heavy algebra is performed symbolically on packed coefficient arrays.
Only NumPy is used so the implementation is portable and fast.

References
----------
Jorba, À. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".

Zhang, H. Q., Li, S. (2001). "Improved semi-analytical computation of center
manifolds near collinear libration points".
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from hiten.algorithms.center.hamiltonian import _build_physical_hamiltonian
from hiten.algorithms.center.lie import (_evaluate_transform, _lie_expansion,
                                         _lie_transform)
from hiten.algorithms.center.transforms import (_local2realmodal,
                                                _local2synodic_collinear,
                                                _local2synodic_triangular,
                                                _realmodal2local,
                                                _solve_complex, _solve_real,
                                                _substitute_complex,
                                                _substitute_real)
from hiten.algorithms.poincare.map import _solve_missing_coord
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _decode_multiindex,
                                              _init_index_tables)
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import CollinearPoint, L3Point
from hiten.system.libration.triangular import TriangularPoint
from hiten.utils.log_config import logger
from hiten.utils.printing import _format_cm_table

if TYPE_CHECKING:
    from hiten.algorithms.poincare.base import _PoincareMap


class CenterManifold:
    r"""
    Centre manifold normal-form builder.

    Parameters
    ----------
    point : hiten.system.libration.collinear.CollinearPoint
        Collinear libration point about which the normal form is computed.
    max_degree : int
        Maximum total degree :math:`N` of the polynomial truncation.

    Attributes
    ----------
    point : hiten.system.libration.collinear.CollinearPoint
        Same as the constructor argument.
    max_degree : int
        Same as the constructor argument.
    psi, clmo : numpy.ndarray
        Index tables used to pack and unpack multivariate monomials.
    encode_dict_list : list of dict
        Helper structures for encoding multi-indices.
    _cache : dict
        Stores intermediate polynomial objects keyed by tuples to avoid
        recomputation.
    _poincare_maps : Dict[Tuple[float, tuple], hiten.algorithms.poincare.base._PoincareMap]
        Lazy cached instances of the Poincaré return maps.

    Notes
    -----
    All heavy computations are cached. Calling :py:meth:`compute` more than once
    with the same *max_degree* is inexpensive because it reuses cached results.
    """
    def __init__(self, point: LibrationPoint, max_degree: int):
        self.point = point
        self.max_degree = max_degree

        if isinstance(self.point, CollinearPoint):
            self._local2synodic = _local2synodic_collinear

            if isinstance(self.point, L3Point):
                logger.warning("L3 point is not has not been verified for centre manifold computation!")

        elif isinstance(self.point, TriangularPoint):
            self._local2synodic = _local2synodic_triangular
            err = "Triangular points not implemented for centre manifold computation!"
            logger.error(err)
            raise NotImplementedError(err)

        else:
            raise ValueError(f"Unsupported libration point type: {type(self.point)}")

        self._psi, self._clmo = _init_index_tables(self.max_degree)
        self._encode_dict_list = _create_encode_dict_from_clmo(self._clmo)
        self._cache = {}
        self._poincare_maps: Dict[Tuple[float, tuple], "_PoincareMap"] = {}

    def __str__(self):
        r"""
        Return a nicely formatted table of centre-manifold coefficients.

        The coefficients are taken from the cache if available; otherwise the
        centre-manifold Hamiltonian is computed on the fly (which implicitly
        stores the result in the cache).  The helper function
        :pyfunc:`hiten.utils.printing._format_cm_table` is then used to create
        the textual representation.
        """
        # Retrieve cached coefficients if present; otherwise compute them.
        poly_cm = self.cache_get(("hamiltonian", self.max_degree, "center_manifold_real"))

        if poly_cm is None:
            poly_cm = self.compute()

        return _format_cm_table(poly_cm, self._clmo)
    
    def __repr__(self):
        return f"CenterManifold(point={self.point}, max_degree={self.max_degree})"
    
    def cache_get(self, key: tuple) -> Any:
        r"""
        Get a value from the cache.
        """
        return self._cache.get(key)
    
    def cache_set(self, key: tuple, value: Any):
        r"""
        Set a value in the cache.
        """
        self._cache[key] = value
    
    def cache_clear(self):
        r"""
        Clear the cache.
        """
        self._cache.clear()
    
    def _get_physical_hamiltonian(self) -> List[np.ndarray]:
        key = ('hamiltonian', self.max_degree, 'physical')
        if (poly_phys := self.cache_get(key)) is None:
            poly_phys = _build_physical_hamiltonian(self.point, self.max_degree)
            self.cache_set(key, [h.copy() for h in poly_phys])
        return [h.copy() for h in poly_phys]

    def _get_real_normal_form(self) -> List[np.ndarray]:
        key = ('hamiltonian', self.max_degree, 'real_normal')
        if (poly_rn := self.cache_get(key)) is None:
            poly_phys = self._get_physical_hamiltonian()
            poly_rn = _local2realmodal(self.point, poly_phys, self.max_degree, self._psi, self._clmo)
            self.cache_set(key, [h.copy() for h in poly_rn])
        return [h.copy() for h in poly_rn]

    def _get_complex_normal_form(self) -> List[np.ndarray]:
        key = ('hamiltonian', self.max_degree, 'complex_normal')
        if (poly_cn := self.cache_get(key)) is None:
            poly_rn = self._get_real_normal_form()
            poly_cn = _substitute_complex(poly_rn, self.max_degree, self._psi, self._clmo)
            self.cache_set(key, [h.copy() for h in poly_cn])
        return [h.copy() for h in poly_cn]

    def _get_lie_transform_results(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        key_trans = ('hamiltonian', self.max_degree, 'normalized')
        key_G = ('generating_functions', self.max_degree)
        key_elim = ('terms_to_eliminate', self.max_degree)

        poly_trans = self.cache_get(key_trans)
        poly_G_total = self.cache_get(key_G)
        poly_elim_total = self.cache_get(key_elim)

        if any(p is None for p in [poly_trans, poly_G_total, poly_elim_total]):
            logger.info("Performing Lie transformation...")
            poly_cn = self._get_complex_normal_form()
            poly_trans, poly_G_total, poly_elim_total = _lie_transform(
                self.point, poly_cn, self._psi, self._clmo, self.max_degree
            )
            
            self.cache_set(key_trans, [h.copy() for h in poly_trans])
            self.cache_set(key_G, [g.copy() for g in poly_G_total])
            self.cache_set(key_elim, [e.copy() for e in poly_elim_total])
        
        return ([h.copy() for h in poly_trans], [g.copy() for g in poly_G_total], [e.copy() for e in poly_elim_total])

    def _get_center_manifold_complex(self) -> List[np.ndarray]:
        key = ('hamiltonian', self.max_degree, 'center_manifold_complex')
        if (poly_cm_complex := self.cache_get(key)) is None:
            poly_trans, _, _ = self._get_lie_transform_results()
            poly_cm_complex = self._restrict_to_center_manifold(poly_trans)
            self.cache_set(key, [h.copy() for h in poly_cm_complex])
        return [h.copy() for h in poly_cm_complex]
    
    def compute(self) -> List[np.ndarray]:
        r"""
        Compute the polynomial Hamiltonian restricted to the centre manifold.

        The returned list lives in *real modal* coordinates
        :math:`(q_2, p_2, q_3, p_3)`. This method serves as the main entry
        point for the centre manifold computation pipeline, triggering lazy
        computation and caching of all intermediate steps.

        Returns
        -------
        list of numpy.ndarray
            Sequence :math:`[H_0, H_2, \dots, H_N]` where each entry contains the
            packed coefficients of the homogeneous polynomial of that degree.

        Raises
        ------
        RuntimeError
            If any underlying computation step fails.
        
        Notes
        -----
        This routine chains together the full normal-form pipeline and may be
        computationally expensive on the first call. Intermediate objects are
        cached so that subsequent calls are fast.
        """
        key = ('hamiltonian', self.max_degree, 'center_manifold_real')
        if (poly_cm_real := self.cache_get(key)) is None:
            logger.info(f"Computing center manifold for {type(self.point).__name__}, max_deg={self.max_degree}")
            poly_cm_complex = self._get_center_manifold_complex()
            poly_cm_real = _substitute_real(poly_cm_complex, self.max_degree, self._psi, self._clmo)
            self.cache_set(key, [h.copy() for h in poly_cm_real])
            logger.info(f"Center manifold computation complete for {type(self.point).__name__}")
        
        return [h.copy() for h in poly_cm_real]

    def _restrict_to_center_manifold(self, poly_H, tol=1e-14):
        r"""
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
                k = _decode_multiindex(pos, deg, self._clmo)
                if k[0] != 0 or k[3] != 0:       # q1 or p1 exponent non-zero
                    coeff_vec[pos] = 0.0
        return poly_cm
    
    def poincare_map(self, energy: float, **kwargs) -> "_PoincareMap":
        r"""
        Return a cached (or newly built) Poincaré return map.

        Parameters
        ----------
        energy : float
            Hamiltonian energy :math:`h_0` corresponding to the desired Jacobi
            constant.
        **kwargs
            Optional keyword arguments forwarded to
            :pyclass:`hiten.algorithms.poincare.base._PoincareMapConfig`.

        Returns
        -------
        hiten.algorithms.poincare.base._PoincareMap
            Configured Poincaré map instance.

        Notes
        -----
        A map is constructed for each unique combination of energy and
        configuration, and stored internally. Subsequent calls with the same
        parameters return the cached object.
        """
        # Note: moved here from top level to avoid circular import.
        from dataclasses import asdict

        from hiten.algorithms.poincare.base import (_PoincareMap,
                                                    _PoincareMapConfig)

        # Create a config object from kwargs, using dataclass defaults for any
        # that are not provided.
        config_fields = set(_PoincareMapConfig.__dataclass_fields__.keys())
        valid_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        cfg = _PoincareMapConfig(**valid_kwargs)

        # Create a hashable key from the configuration.
        config_tuple = tuple(sorted(asdict(cfg).items()))
        cache_key = (energy, config_tuple)

        if cache_key not in self._poincare_maps:
            self._poincare_maps[cache_key] = _PoincareMap(self, energy, cfg)
        
        return self._poincare_maps[cache_key]

    def ic(self, poincare_point: np.ndarray, energy: float, section_coord: str = "q3") -> np.ndarray:
        r"""
        Convert a point on a 2-dimensional centre-manifold section to full ICs.

        Parameters
        ----------
        poincare_point : numpy.ndarray, shape (2,)
            Coordinates on the chosen Poincaré section.
        energy : float
            Hamiltonian energy :math:`h_0` used to solve for the missing coordinate.
        section_coord : {'q3', 'p3', 'q2', 'p2'}, default 'q3'
            Coordinate fixed to zero on the section.

        Returns
        -------
        numpy.ndarray, shape (6,)
            Synodic initial conditions
            :math:`(q_1, q_2, q_3, p_1, p_2, p_3)`.

        Raises
        ------
        RuntimeError
            If root finding fails or if required Lie generators are missing.

        Examples
        --------
        >>> cm = CenterManifold(L1, 8)
        >>> ic_synodic = cm.ic(np.array([0.01, 0.0]), energy=-1.5, section_coord='q3')
        """
        logger.info(
            "Converting Poincaré point %s (section=%s) to initial conditions", 
            poincare_point, section_coord,
        )

        # Ensure we have the centre-manifold Hamiltonian and Lie generators.
        poly_cm_real = self.compute()
        _, poly_G_total, _ = self._get_lie_transform_results()

        # Alias for brevity.
        h0 = float(energy)
        q2 = p2 = q3 = p3 = None  # type: ignore

        if section_coord == "q3":
            # q3 = 0 section → need p3
            q2, p2 = map(float, poincare_point)
            q3 = 0.0
            p3 = _solve_missing_coord(
                "p3", {"q2": q2, "p2": p2}, h0, poly_cm_real, self._clmo
            )
        elif section_coord == "p3":
            # p3 = 0 section → need q3
            q2, p2 = map(float, poincare_point)
            p3 = 0.0
            q3 = _solve_missing_coord(
                "q3", {"q2": q2, "p2": p2, "p3": 0.0}, h0, poly_cm_real, self._clmo
            )
        elif section_coord == "q2":
            # q2 = 0 section → need p2
            q3, p3 = map(float, poincare_point)
            q2 = 0.0
            p2 = _solve_missing_coord(
                "p2", {"q2": 0.0, "q3": q3, "p3": p3}, h0, poly_cm_real, self._clmo
            )
        elif section_coord == "p2":
            # p2 = 0 section → need q2
            q3, p3 = map(float, poincare_point)
            p2 = 0.0
            q2 = _solve_missing_coord(
                "q2", {"p2": 0.0, "q3": q3, "p3": p3}, h0, poly_cm_real, self._clmo
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

        complex_6d_cm = _solve_complex(real_6d_cm)
        expansions = _lie_expansion(
            poly_G_total, self.max_degree, self._psi, self._clmo, 1e-30,
            inverse=False, sign=1, restrict=False,
        )
        complex_6d = _evaluate_transform(expansions, complex_6d_cm, self._clmo)
        real_6d = _solve_real(complex_6d)
        local_6d = _realmodal2local(self.point, real_6d)
        synodic_6d = self._local2synodic(self.point, local_6d)

        logger.info("CM → synodic transformation complete")
        return synodic_6d

    def ic2cm(self) -> np.ndarray:
        r"""
        TODO: Implement initial conditions to center manifold transformation.
        """
        pass
