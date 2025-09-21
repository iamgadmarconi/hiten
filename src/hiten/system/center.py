"""High-level utilities for computing a polynomial normal form of the centre
manifold around a collinear libration point of the spatial circular
restricted three body problem (CRTBP).

All heavy algebra is performed symbolically on packed coefficient arrays.
Only NumPy is used so the implementation is portable and fast.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Jorba, A. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".

Zhang, H. Q., Li, S. (2001). "Improved semi-analytical computation of center
manifolds near collinear libration points".
"""

from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

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
from hiten.algorithms.poincare.centermanifold.backend import _CenterManifoldBackend
from hiten.algorithms.poincare.centermanifold.interfaces import _CenterManifoldInterface
from hiten.algorithms.poincare.centermanifold.config import _get_section_config
from hiten.algorithms.poincare.core.events import _PlaneEvent
from hiten.system.hamiltonians.pipeline import HamiltonianPipeline
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import CollinearPoint, L3Point
from hiten.system.libration.triangular import TriangularPoint
from hiten.utils.io.center import load_center_manifold, save_center_manifold
from hiten.utils.log_config import logger
from hiten.utils.printing import _format_poly_table

if TYPE_CHECKING:
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap


class CenterManifold:
    """
    Centre manifold normal-form builder.

    This class provides high-level operations for working with center manifolds
    around libration points. It uses the HamiltonianPipeline internally for
    all Hamiltonian computations and focuses on center manifold specific
    operations like coordinate transformations and Poincare maps.

    Parameters
    ----------
    point : :class:`~hiten.system.libration.base.LibrationPoint`
        Libration point about which the center manifold is computed.
    degree : int
        Maximum total degree N of the polynomial truncation.

    Attributes
    ----------
    point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point about which the center manifold is computed.
    degree : int
        The maximum total degree of the polynomial truncation.
    pipeline : :class:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline`
        Internal pipeline for managing Hamiltonian computations and caching.

    Notes
    -----
    All heavy computations are cached by the internal HamiltonianPipeline.
    Calling methods more than once with the same parameters is inexpensive
    because cached results are reused.
    """
    
    def __init__(self, point: LibrationPoint, degree: int):
        self._point = point
        self._max_degree = degree
        
        # Initialize the Hamiltonian pipeline
        self._pipeline = HamiltonianPipeline(point, degree)
        self._hamsys = self._pipeline.get_hamiltonian("center_manifold_real").hamsys
        
        # Set up coordinate transformation functions based on point type
        if isinstance(self._point, CollinearPoint):
            self._local2synodic = _local2synodic_collinear
            self._synodic2local = _synodic2local_collinear
            self._mix_pairs = (1, 2)

            if isinstance(self._point, L3Point):
                logger.warning("L3 point has not been verified for centre manifold / normal form computations!")

        elif isinstance(self._point, TriangularPoint):
            logger.warning("Triangular points have not been verified for centre manifold / normal form computations!")
            self._local2synodic = _local2synodic_triangular
            self._synodic2local = _synodic2local_triangular
            self._mix_pairs = (0, 1, 2)
            raise NotImplementedError("Triangular points are not supported yet.")

        else:
            raise ValueError(f"Unsupported libration point type: {type(self._point)}")

        # Cache for Poincare maps
        self._poincare_maps: Dict[Tuple[float, tuple], "CenterManifoldMap"] = {}
        self._backends: Dict[Tuple[float, str], _CenterManifoldBackend] = {} # energy, section_coord

    @property
    def point(self) -> LibrationPoint:
        """The libration point about which the center manifold is computed.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point about which the center manifold is computed.
        """
        return self._point

    @property
    def degree(self) -> int:
        """The maximum total degree of the polynomial truncation.
        
        Returns
        -------
        int
            The maximum total degree of the polynomial truncation.
        """
        return self._max_degree

    @degree.setter
    def degree(self, value: int):
        """Set a new maximum degree, which invalidates all cached data.
        
        Parameters
        ----------
        value : int
            New maximum degree for the polynomial truncation.
            
        Raises
        ------
        ValueError
            If value is not a positive integer.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("degree must be a positive integer.")
            
        if value != self._max_degree:
            logger.info(
                f"Maximum degree changed from {self._max_degree} to {value}. "
                "Invalidating all cached data."
            )
            self._max_degree = value
            # Recreate pipeline with new degree
            self._pipeline = HamiltonianPipeline(self._point, value)
            # Refresh internal Hamiltonian system and dependent caches
            self._hamsys = self._pipeline.get_hamiltonian("center_manifold_real").hamsys
            self._poincare_maps.clear()
            self._backends.clear()

    @property
    def pipeline(self) -> HamiltonianPipeline:
        """Access to the internal Hamiltonian pipeline.
        
        Returns
        -------
        :class:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline`
            The internal Hamiltonian pipeline.
        """
        return self._pipeline

    def __str__(self):
        return f"CenterManifold(point={self._point}, degree={self._max_degree})" 
    
    def __repr__(self):
        return f"CenterManifold(point={self._point}, degree={self._max_degree})"
    
    def __getstate__(self):
        """Get the state for pickling.
        
        Returns
        -------
        dict
            Dictionary containing the serializable state.
        """
        return {
            "_point": self._point,
            "_max_degree": self._max_degree,
        }

    def __setstate__(self, state):
        """Restore the state after unpickling.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state.
        """
        self._point = state["_point"]
        self._max_degree = state["_max_degree"]
        self._pipeline = HamiltonianPipeline(self._point, self._max_degree)
        self._hamsys = self._pipeline.get_hamiltonian("center_manifold_real").hamsys
        self._poincare_maps = {}
        self._backends = {}
        
        # Re-setup coordinate transformation functions
        if isinstance(self._point, CollinearPoint):
            self._local2synodic = _local2synodic_collinear
            self._synodic2local = _synodic2local_collinear
            self._mix_pairs = (1, 2)
        elif isinstance(self._point, TriangularPoint):
            self._local2synodic = _local2synodic_triangular
            self._synodic2local = _synodic2local_triangular
            self._mix_pairs = (0, 1, 2)

    def compute(self, form: str = "center_manifold_real") -> List[np.ndarray]:
        """
        Compute and return a specific polynomial representation of the Hamiltonian.

        This method delegates to the internal :class:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline`
        for all Hamiltonian computations and caching.

        Parameters
        ----------
        form : str, optional
            Identifier of the desired polynomial representation. Defaults to
            "center_manifold_real" for backward compatibility.

        Returns
        -------
        list of numpy.ndarray
            Sequence [H_0, H_2, ..., H_N] where each entry contains the packed
            coefficients of the homogeneous polynomial of that degree.
        """
        return self._pipeline.get_hamiltonian(form).poly_H

    def coefficients(self, form: str = "center_manifold_real", 
                    degree: Union[int, Iterable[int], str, None] = None) -> str:
        """
        Return a formatted string representation of the Hamiltonian coefficients.

        Parameters
        ----------
        form : str, optional
            Identifier of the desired polynomial representation.
        degree : int, Iterable[int], str, or None, optional
            Degree filter for the coefficient table.

        Returns
        -------
        str
            Formatted coefficient table.
        """
        ham = self._pipeline.get_hamiltonian(form)
        table = _format_poly_table(ham.poly_H, ham._clmo, degree)
        logger.info(f'{form} coefficients:\n\n{table}\n\n')
        return table

    def cache_clear(self):
        """Clear all caches (Hamiltonian pipeline and Poincare maps).
        
        This method clears all internal caches including the Hamiltonian pipeline
        cache and all Poincare map caches.
        """
        logger.debug("Clearing all caches.")
        self._pipeline.cache_clear()
        self._poincare_maps.clear()
        self._backends.clear()

    def _restrict_coord_to_center_manifold(self, coord_6d):
        """Project a 6-D Phase-space coordinate onto the centre manifold.

        For collinear points the hyperbolic pair (q1, p1) is removed. For
        triangular points all six variables belong to the centre manifold so
        the original coordinates are returned unchanged (apart from casting to
        real dtype and ensuring contiguity).
        
        Parameters
        ----------
        coord_6d : numpy.ndarray
            6-dimensional phase space coordinate.
            
        Returns
        -------
        numpy.ndarray
            Projected coordinate on the centre manifold.
        """
        # Always work with real numbers once we reach this stage.
        if np.iscomplexobj(coord_6d):
            coord_6d = np.real(coord_6d)

        if isinstance(self._point, TriangularPoint):
            # Nothing to eliminate, return full 6-vector.
            return np.ascontiguousarray(coord_6d, dtype=np.float64)

        # Collinear case: zero out the hyperbolic coordinates.
        return np.array([0.0, coord_6d[1], coord_6d[2], 0.0, coord_6d[4], coord_6d[5]], dtype=np.float64)
    
    def _get_or_create_backend(self, energy: float, section_coord: str, **kwargs) -> _CenterManifoldBackend:
        """Get or create a backend for the given section coordinate and energy.
        
        Parameters
        ----------
        energy : float
            Energy level for the backend.
        section_coord : str
            Section coordinate identifier.
        **kwargs
            Additional keyword arguments for backend configuration.
            
        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.backend._CenterManifoldBackend`
            The backend instance.
        """
        # Include integration-related kwargs in the cache key to allow
        # multiple backends with different runtime settings to coexist.
        method = kwargs.get("method")
        order = kwargs.get("order")
        c_omega_heuristic = kwargs.get("c_omega_heuristic")
        cache_key = (energy, section_coord, method, order, c_omega_heuristic)
        if cache_key not in self._backends:
            cm_hamsys = self._hamsys
            self._backends[cache_key] = _CenterManifoldBackend(
                                    dynsys=cm_hamsys,
                                    surface=_PlaneEvent(coord=section_coord, value=0.0, direction=None),
                                    section_coord=section_coord,
                                    h0=energy,
                                    **kwargs
                                )
        return self._backends[cache_key]

    def _4d_cm_to_ic(self, cm_coords_4d: np.ndarray, tol: float = 1e-14) -> np.ndarray:
        """Convert 4-D centre-manifold coordinates to 6-D synodic initial conditions.

        This helper assumes cm_coords_4d is an array-like object of shape (4,)
        containing the real centre-manifold variables [q2, p2, q3, p3]. No
        root-finding or Hamiltonian energy information is required - the
        supplied coordinates are taken to lie on the centre manifold already.

        The transformation follows exactly the second half of the original
        :meth:`~hiten.system.center.CenterManifold.ic` pipeline:

            CM (real) -> CM (complex) -> Lie transform -> real modal -> local -> synodic
            
        Parameters
        ----------
        cm_coords_4d : numpy.ndarray, shape (4,)
            Centre-manifold coordinates [q2, p2, q3, p3] in nondimensional units.
        tol : float, default 1e-14
            Numerical tolerance for the transformation.
            
        Returns
        -------
        numpy.ndarray, shape (6,)
            Synodic initial conditions [x, y, z, vx, vy, vz] in nondimensional units.
        """
        # Construct a 6-D centre-manifold phase-space vector
        real_4d_cm = np.asarray(cm_coords_4d, dtype=np.float64).reshape(4)

        real_6d_cm = np.zeros(6, dtype=np.complex128)
        real_6d_cm[1] = real_4d_cm[0]  # q2
        real_6d_cm[4] = real_4d_cm[1]  # p2
        real_6d_cm[2] = real_4d_cm[2]  # q3
        real_6d_cm[5] = real_4d_cm[3]  # p3

        # Modal (real -> complex) representation
        complex_6d_cm = _solve_complex(real_6d_cm, tol=tol, mix_pairs=self._mix_pairs)

        # Apply the forward Lie transform (centre-manifold -> physical variables)
        expansions = self._pipeline.get_lie_expansions(inverse=False, tol=tol)
        complex_6d = _evaluate_transform(expansions, complex_6d_cm, self._hamsys.clmo_H)

        # Back to real modal variables
        real_6d = _solve_real(complex_6d, tol=tol, mix_pairs=self._mix_pairs)

        # Modal (real) -> local -> synodic coordinate chain
        local_6d = _coordrealmodal2local(self._point, real_6d, tol)
        synodic_6d = self._local2synodic(self._point, local_6d, tol)

        logger.info("CM->synodic transformation (4-D input) complete")
        return synodic_6d

    def _2d_cm_to_ic(self, poincare_point: np.ndarray, energy: float,
                     section_coord: str = "q3", tol: float = 1e-14, **kwargs) -> np.ndarray:
        """Original ic behaviour - convert a 2-D Poincare-section point.

        This routine reproduces verbatim the legacy implementation that:

        1. Uses the Hamiltonian energy constraint to solve for the missing
           coordinate on the chosen section;
        2. Embeds the resulting 4-D CM coordinates into 6-D phase-space;
        3. Applies the Lie transform and coordinate conversions to obtain the
           synodic initial conditions.
           
        Parameters
        ----------
        poincare_point : numpy.ndarray, shape (2,)
            Point on the Poincare section in nondimensional units.
        energy : float
            Hamiltonian energy level in nondimensional units.
        section_coord : str, default "q3"
            Coordinate fixed to zero on the Poincare section.
        tol : float, default 1e-14
            Numerical tolerance for the transformation.
        **kwargs
            Additional keyword arguments for backend configuration.
            
        Returns
        -------
        numpy.ndarray, shape (6,)
            Synodic initial conditions [x, y, z, vx, vy, vz] in nondimensional units.
        """
        # Section configuration specifies which coordinate is fixed to zero and
        # which one must be solved for
        config = _get_section_config(section_coord)

        # Known variables on the section
        known_vars: Dict[str, float] = {config.section_coord: 0.0}
        known_vars[config.plane_coords[0]] = float(poincare_point[0])
        known_vars[config.plane_coords[1]] = float(poincare_point[1])

        var_to_solve = config.missing_coord

        # Use the stateless Interface to solve the missing coordinate
        solved_val = _CenterManifoldInterface.solve_missing_coord(
            var_to_solve,
            known_vars,
            h0=float(energy),
            H_blocks=self._hamsys.poly_H(),
            clmo_table=self._hamsys.clmo_table,
        )

        # Combine into a full CM coordinate dictionary
        full_cm_coords = known_vars.copy()
        full_cm_coords[var_to_solve] = solved_val

        # Sanity check
        if any(v is None for v in full_cm_coords.values()):
            err = "Failed to reconstruct full CM coordinates - root finding did not converge."
            logger.error(err)
            raise RuntimeError(err)

        real_4d_cm = np.array([
            full_cm_coords["q2"],
            full_cm_coords["p2"],
            full_cm_coords["q3"],
            full_cm_coords["p3"],
        ], dtype=np.float64)

        # Delegate the second half of the pipeline to the 4-D helper
        return self._4d_cm_to_ic(real_4d_cm, tol)

    def ic(self, cm_point: np.ndarray, energy: Optional[float] = None,
           section_coord: str = "q3", tol: float = 1e-14) -> np.ndarray:
        """Convert centre-manifold coordinates to full synodic ICs.

        The method now supports two input formats:

        1. 2-D Poincare-section coordinates (legacy behaviour). In this case
           energy must be provided and section_coord specifies which CM
           coordinate is fixed to zero on the section.
        2. 4-D centre-manifold coordinates [q2, p2, q3, p3]. Here the
           coordinates are assumed to satisfy the Hamiltonian energy
           constraint already, so energy and section_coord are ignored.

        Parameters
        ----------
        cm_point : numpy.ndarray, shape (2,) or (4,)
            Point on the Poincare section (length-2) or full centre-manifold
            coordinates (length-4) in nondimensional units.
        energy : float or None, optional
            Hamiltonian energy level h0 in nondimensional units. Required only when cm_point is a
            2-vector.
        section_coord : {'q3', 'p3', 'q2', 'p2'}, default 'q3'
            Coordinate fixed to zero on the Poincare section. Ignored for
            4-D inputs.
        tol : float, default 1e-14
            Numerical tolerance used by the various helper routines.

        Returns
        -------
        numpy.ndarray, shape (6,)
            Synodic-frame initial conditions (x, y, z, vx, vy, vz) in nondimensional units.
            
        Raises
        ------
        ValueError
            If energy is not provided for 2-D input or if cm_point has invalid shape.
        """
        cm_point = np.asarray(cm_point)

        if cm_point.size == 2:
            if energy is None:
                raise ValueError(
                    "energy must be specified when converting a 2-D Poincare "
                    "point to initial conditions."
                )
            logger.info(
                "Converting 2-D Poincare point %s (section=%s) to synodic ICs",
                cm_point, section_coord,
            )
            return self._2d_cm_to_ic(cm_point, float(energy), section_coord, tol)

        elif cm_point.size == 4:
            logger.info("Converting 4-D CM point %s to synodic ICs", cm_point)
            return self._4d_cm_to_ic(cm_point, tol)

        else:
            raise ValueError(
                "cm_point must be either a 2- or 4-element vector representing "
                "a Poincare-section point or full CM coordinates, respectively."
            )

    def cm(self, synodic_6d: np.ndarray, tol=1e-14) -> np.ndarray:
        """Return 4-D centre-manifold coordinates (q2, p2, q3, p3) from 6-D synodic ICs.

        This is the exact inverse of :meth:`~hiten.system.center.CenterManifold.ic` and therefore performs the
        following steps in reverse order:

            synodic -> local -> real modal -> complex modal -> Lie-inverse -> CM.

        Parameters
        ----------
        synodic_6d : numpy.ndarray, shape (6,)
            Synodic coordinates (x, y, z, vx, vy, vz) in nondimensional units.
        tol : float, default 1e-14
            Numerical tolerance for the transformation.

        Returns
        -------
        numpy.ndarray, shape (4,)
            Centre-manifold real coordinates [q2, p2, q3, p3] in nondimensional units.
        """
        local_6d = self._synodic2local(self._point, synodic_6d, tol)
        real_modal_6d = _coordlocal2realmodal(self._point, local_6d, tol)
        complex_modal_6d = _solve_complex(real_modal_6d, tol=tol, mix_pairs=self._mix_pairs)

        expansions = self._pipeline.get_lie_expansions(inverse=True, tol=tol)
        complex_pnf_6d = _evaluate_transform(expansions, complex_modal_6d, 
                                           self._hamsys.clmo)
        real_pnf_6d = _solve_real(complex_pnf_6d, tol=tol, mix_pairs=self._mix_pairs)
        real_cm_6d = self._restrict_coord_to_center_manifold(real_pnf_6d)

        real_cm_4d = np.array([
            real_cm_6d[1], # q2
            real_cm_6d[4], # p2
            real_cm_6d[2], # q3
            real_cm_6d[5], # p3
        ], dtype=np.float64)

        return real_cm_4d

    def poincare_map(self, energy: float, **kwargs) -> "CenterManifoldMap":
        """
        Create a Poincare map at the specified energy level.

        Parameters
        ----------
        energy : float
            Hamiltonian energy h0 in nondimensional units.
        **kwargs
            Configuration parameters for the Poincare map.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
            A Poincare map object for the given energy and configuration.

        Notes
        -----
        A map is constructed for each unique combination of energy and
        configuration, and stored internally. Subsequent calls with the same
        parameters return the cached object.
        """
        from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
        from hiten.algorithms.poincare.centermanifold.config import \
            _CenterManifoldMapConfig

        # Separate config kwargs from runtime kwargs
        config_fields = set(_CenterManifoldMapConfig.__dataclass_fields__.keys())
        
        config_kwargs = {}
        for key, value in kwargs.items():
            if key in config_fields:
                config_kwargs[key] = value
            else:
                raise TypeError(f"'{key}' is not a valid keyword argument for PoincareMap configuration.")
        
        cfg = _CenterManifoldMapConfig(**config_kwargs)

        # Create a hashable key from the configuration
        config_tuple = tuple(sorted(asdict(cfg).items()))
        cache_key = (energy, config_tuple)

        if cache_key not in self._poincare_maps:
            # Prefer DI-friendly constructor with a default engine wired
            self._poincare_maps[cache_key] = CenterManifoldMap.with_default_engine(self, energy, cfg)
        
        return self._poincare_maps[cache_key]

    def save(self, dir_path: str, **kwargs):
        """
        Save the :class:`~hiten.system.center.CenterManifold` instance to a directory.

        This method serializes the main object to 'manifold.pkl' and saves
        each associated Poincare map to a separate file within a 'poincare_maps'
        subdirectory.

        Parameters
        ----------
        dir_path : str or path-like object
            The path to the directory where the data will be saved.
        **kwargs
            Additional keyword arguments for the save operation.
        """
        save_center_manifold(self, dir_path, **kwargs)

    @classmethod
    def load(cls, dir_path: str, **kwargs) -> "CenterManifold":
        """
        Load a :class:`~hiten.system.center.CenterManifold` instance from a directory.

        This class method deserializes a CenterManifold object and its
        associated Poincare maps that were saved with the save method.

        Parameters
        ----------
        dir_path : str or path-like object
            The path to the directory from which to load the data.
        **kwargs
            Additional keyword arguments for the load operation.

        Returns
        -------
        :class:`~hiten.system.center.CenterManifold`
            The loaded CenterManifold instance with its Poincare maps.
        """
        return load_center_manifold(dir_path, **kwargs)



