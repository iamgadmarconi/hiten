r"""
hiten.algorithms.poincare.base
=========================

Poincaré return map utilities on the centre manifold of the spatial circular
restricted three body problem.

The module exposes a high level interface :pyclass:`_PoincareMap` that wraps
specialised CPU/GPU kernels to generate, query, and visualise discrete
Poincaré sections arising from the reduced Hamiltonian flow. Numerical
parameters are grouped in the lightweight dataclass
:pyclass:`_PoincareMapConfig`.
"""

import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.cuda.map import _generate_map_gpu
from hiten.algorithms.poincare.map import _generate_grid
from hiten.algorithms.poincare.map import _generate_map as _generate_map_cpu
from hiten.algorithms.poincare.map import _PoincareSection
from hiten.system.center import CenterManifold
from hiten.system.libration.triangular import TriangularPoint
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io import (_ensure_dir, _load_poincare_map,
                            _load_poincare_map_inplace, _save_poincare_map)
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive


@dataclass
class _PoincareMapConfig:
    dt: float = 1e-2
    method: str = "rk"  # "symplectic" or "rk"
    integrator_order: int = 4
    c_omega_heuristic: float = 20.0  # Only used by the extended-phase symplectic scheme

    n_seeds: int = 20
    n_iter: int = 40
    seed_strategy: Literal["single", "axis_aligned", "level_sets", "radial", "random"] = "axis_aligned"
    seed_axis: Optional[Literal["q2", "p2", "q3", "p3"]] = None
    section_coord: Literal["q2", "p2", "q3", "p3"] = "q3"

    compute_on_init: bool = False
    use_gpu: bool = False

    def __post_init__(self):
        if self.seed_strategy == "single" and self.seed_axis is None:
            raise ValueError("seed_axis must be specified when seed_strategy is 'single'")
        
        elif self.seed_strategy != 'single':
            if self.seed_axis is not None:
                logger.warning("seed_axis is ignored when seed_strategy is not 'single'")

        if self.use_gpu:
            raise NotImplementedError("GPU backend is not implemented yet")


class _PoincareMap:
    r"""
    High-level object representing a Poincaré map on the centre manifold.

    Parameters
    ----------
    cm : CenterManifold
        The centre-manifold object to operate on.  Its polynomial representation is
        used for the reduced Hamiltonian flow.
    energy : float
        Energy level (same convention as :pyfunc:`_solve_missing_coord`, *not* the Jacobi constant).
    config : _PoincareMapConfig, optional
        Numerical parameters controlling the map generation.  A sensible default
        configuration is used if none is supplied.
    """

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[_PoincareMapConfig] = None,
    ) -> None:
        self.cm: CenterManifold = cm
        if isinstance(self.cm.point, TriangularPoint):
            raise ValueError("Poincaré map is not supported for triangular points.")
        self.energy: float = float(energy)
        self.config: _PoincareMapConfig = config or _PoincareMapConfig()

        # Derived flags
        self._use_symplectic: bool = self.config.method.lower() == "symplectic"

        # Storage for computed sections/grids - allow multiple section planes
        self._section: Optional[_PoincareSection] = None  # Most-recent section (back-compat)
        self._grid: Optional[_PoincareSection] = None      # Most-recent grid  (back-compat)

        # Dictionaries keyed by section coordinate ("q2", "p2", "q3", "p3")
        self._sections: dict[str, _PoincareSection] = {}
        self._grids: dict[tuple, _PoincareSection] = {}

        self._backend: str = "cpu" if not self.config.use_gpu else "gpu"

        if self.config.compute_on_init:
            self.compute()

    def __repr__(self) -> str:
        return (
            f"_PoincareMap(cm={self.cm!r}, energy={self.energy:.3e}, "
            f"points={len(self) if self._section is not None else '∅'})"
        )

    def __str__(self) -> str:
        return (
            f"Poincaré map at h0={self.energy:.3e} with {len(self)} points"
            if self._section is not None
            else f"Poincaré map (uncomputed) at h0={self.energy:.3e}"
        )

    def __len__(self) -> int:  # Convenient len() support
        return 0 if self._section is None else self._section.points.shape[0]

    @property
    def points(self) -> np.ndarray:
        r"""
        Return the computed Poincaré-map points (backward compatibility).
        """
        if self._section is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._section.points
    
    @property
    def grid(self) -> np.ndarray:
        r"""
        Return the computed Poincaré-map grid (backward compatibility).
        """
        if self._grid is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._grid.points

    @property
    def sections(self) -> dict[str, _PoincareSection]:
        r"""
        Return the computed Poincaré sections.
        """
        return self._sections

    @property
    def section(self) -> _PoincareSection:
        r"""
        Return the computed Poincaré section with labels.
        """
        if self._section is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._section

    def _propagate_from_point(self, cm_point, energy, steps=1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order=6):
        r"""
        Convert a Poincaré map point to initial conditions, create a GenericOrbit, propagate, and return the orbit.
        """
        ic = self.cm.ic(cm_point, energy, section_coord=self.config.section_coord)
        logger.info(f"Initial conditions: {ic}")
        orbit = GenericOrbit(self.cm.point, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    def compute(self, section_coord: Optional[str] = None, **kwargs) -> np.ndarray:
        r"""
        Compute the discrete Poincaré return map.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate to compute. If None, uses config default.
        **kwargs
            Reserved for future extensions.

        Returns
        -------
        numpy.ndarray
            Array of shape (:math:`n`, 2) containing the intersection points.

        Raises
        ------
        RuntimeError
            If the underlying centre manifold computation fails.
        TypeError
            If invalid kwargs are provided.

        Notes
        -----
        The resulting section is cached in :pyattr:`_sections`; subsequent calls
        with the same section coordinate reuse the stored data. Parallel processing 
        is enabled automatically for CPU computations.
        """
        # Determine which section coordinate to use (explicit arg overrides config)
        section_key: str = section_coord or self.config.section_coord

        # Check if already computed
        if section_key in self._sections:
            logger.info(f"Using cached Poincaré section for {section_key}")
            self._section = self._sections[section_key]  # Update current reference
            return self._section.points

        logger.info(
            "Generating Poincaré map at energy h0=%.6e (section=%s, method=%s, cpu_parallel=%s)",
            self.energy,
            section_key,
            self.config.method,
            self._backend == "cpu",
        )

        poly_cm_real = self.cm.compute()

        kernel = _generate_map_gpu if self._backend == "gpu" else _generate_map_cpu

        self._section = kernel(
            h0=self.energy,
            H_blocks=poly_cm_real,
            max_degree=self.cm.max_degree,
            psi_table=self.cm._psi,
            clmo_table=self.cm._clmo,
            encode_dict_list=self.cm._encode_dict_list,
            n_seeds=self.config.n_seeds,
            n_iter=self.config.n_iter,
            dt=self.config.dt,
            use_symplectic=self._use_symplectic,
            integrator_order=self.config.integrator_order,
            c_omega_heuristic=self.config.c_omega_heuristic,
            seed_strategy=self.config.seed_strategy,
            seed_axis=self.config.seed_axis,
            section_coord=section_key)

        # Cache by section coordinate
        self._sections[section_key] = self._section

        logger.info("Poincaré map computation complete: %d points (section=%s)", len(self), section_key)
        return self._section.points

    def compute_grid(self, Nq: int = 201, Np: int = 201, max_steps: int = 20_000, section_coord: Optional[str] = None, **kwargs) -> np.ndarray:
        r"""
        Generate a dense rectangular grid of the Poincaré map.

        Parameters
        ----------
        Nq, Np : int, default 201
            Number of nodes along the :math:`q` and :math:`p` axes.
        max_steps : int, default 20000
            Maximum number of integration steps for each seed.
        section_coord : str, optional
            Section coordinate to compute. If None, uses config default.
        **kwargs
            Reserved for future extensions.

        Returns
        -------
        numpy.ndarray
            Array containing the grid points with the same layout as
            :pyattr:`section.points`.

        Raises
        ------
        ValueError
            If an unsupported backend is selected.
        TypeError
            If invalid kwargs are provided.

        Notes
        -----
        The resulting grid is cached in :pyattr:`_grids`; subsequent calls
        with the same parameters reuse the stored data. Parallel processing 
        is enabled automatically for CPU computations.
        """
        if self._backend == "gpu":
            raise ValueError("GPU backend does not support CPU parallel processing.")
            
        section_key: str = section_coord or self.config.section_coord
        
        # Create a cache key that includes grid parameters
        grid_key = (section_key, Nq, Np, max_steps)
        
        # Check if already computed
        if grid_key in self._grids:
            logger.info(f"Using cached Poincaré grid for {section_key} ({Nq}x{Np})")
            self._grid = self._grids[grid_key]  # Update current reference
            return self._grid.points

        logger.info(
            "Generating *dense-grid* Poincaré map at energy h0=%.6e (section=%s, Nq=%d, Np=%d)",
            self.energy,
            section_key,
            Nq,
            Np,
        )

        # Ensure that the centre manifold polynomial is current.
        poly_cm_real = self.cm.compute()

        self._grid = _generate_grid(
            h0=self.energy,
            H_blocks=poly_cm_real,
            max_degree=self.cm.max_degree,
            psi_table=self.cm._psi,
            clmo_table=self.cm._clmo,
            encode_dict_list=self.cm._encode_dict_list,
            dt=self.config.dt,
            max_steps=max_steps,
            Nq=Nq,
            Np=Np,
            integrator_order=self.config.integrator_order,
            use_symplectic=self._use_symplectic,
            section_coord=section_key,
            )

        # Cache by section coordinate and grid parameters
        self._grids[grid_key] = self._grid

        logger.info("Dense-grid Poincaré map computation complete: %d points (section=%s)", len(self), section_key)
        return self._grid.points

    def get_section(self, section_coord: str) -> _PoincareSection:
        r"""
        Retrieve a specific Poincaré section by coordinate.

        Parameters
        ----------
        section_coord : {"q2", "p2", "q3", "p3"}
            The section coordinate to retrieve.

        Returns
        -------
        _PoincareSection
            The cached section object.

        Raises
        ------
        KeyError
            If the requested section has not been computed.
        """
        if section_coord not in self._sections:
            raise KeyError(f"Section '{section_coord}' has not been computed. Available: {list(self._sections.keys())}")
        return self._sections[section_coord]

    def get_grid(self, section_coord: str, Nq: int = 201, Np: int = 201, max_steps: int = 20_000) -> _PoincareSection:
        r"""
        Retrieve a specific Poincaré grid by parameters.

        Parameters
        ----------
        section_coord : {"q2", "p2", "q3", "p3"}
            The section coordinate.
        Nq, Np : int, default 201
            Grid dimensions.
        max_steps : int, default 20000
            Maximum integration steps.

        Returns
        -------
        _PoincareSection
            The cached grid object.

        Raises
        ------
        KeyError
            If the requested grid has not been computed.
        """
        grid_key = (section_coord, Nq, Np, max_steps)
        if grid_key not in self._grids:
            raise KeyError(f"Grid with parameters {grid_key} has not been computed. Available: {list(self._grids.keys())}")
        return self._grids[grid_key]

    def list_sections(self) -> list[str]:
        r"""
        List all computed section coordinates.

        Returns
        -------
        list[str]
            List of section coordinate names that have been computed.
        """
        return list(self._sections.keys())

    def list_grids(self) -> list[tuple]:
        r"""
        List all computed grid parameter combinations.

        Returns
        -------
        list[tuple]
            List of (section_coord, Nq, Np, max_steps) tuples for computed grids.
        """
        return list(self._grids.keys())

    def has_section(self, section_coord: str) -> bool:
        r"""
        Check if a section has been computed.

        Parameters
        ----------
        section_coord : str
            The section coordinate to check.

        Returns
        -------
        bool
            True if the section exists in cache.
        """
        return section_coord in self._sections

    def has_grid(self, section_coord: str, Nq: int = 201, Np: int = 201, max_steps: int = 20_000) -> bool:
        r"""
        Check if a grid has been computed.

        Parameters
        ----------
        section_coord : str
            The section coordinate.
        Nq, Np : int, default 201
            Grid dimensions.
        max_steps : int, default 20000
            Maximum integration steps.

        Returns
        -------
        bool
            True if the grid exists in cache.
        """
        grid_key = (section_coord, Nq, Np, max_steps)
        return grid_key in self._grids

    def clear_cache(self) -> None:
        r"""
        Clear all cached sections and grids.
        """
        logger.info("Clearing Poincaré map cache")
        self._sections.clear()
        self._grids.clear()
        self._section = None
        self._grid = None

    def get_cm_states(self, section_coord: Optional[str] = None) -> np.ndarray:
        r"""
        Get the centre-manifold states for a specific section.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate to retrieve states from. If None, uses the most recent section.

        Returns
        -------
        numpy.ndarray, shape (n, 4)
            Centre-manifold coordinates (q2, p2, q3, p3) for each map point.

        Raises
        ------
        RuntimeError
            If no section has been computed.
        KeyError
            If the specified section has not been computed.
        """
        if section_coord is None:
            if self._section is None:
                raise RuntimeError("No Poincaré section has been computed yet.")
            return self._section.cm_states
        else:
            section = self.get_section(section_coord)
            return section.cm_states
    
    def get_points(self, axes: Sequence[str] | None = None, section_coord: Optional[str] = None) -> np.ndarray:
        """Return the Poincaré-map points projected onto arbitrary coordinate axes.

        Parameters
        ----------
        axes : Sequence[str] | None, optional
            Pair of coordinate names to project the section onto (e.g. ("q3", "p2")).
            If *None* (default) the axes associated with ``section.labels`` are used,
            reproducing the legacy behaviour.

        Notes
        -----
        The underlying map stores only the two coordinates that were chosen when
        the section was computed.  When a different projection is requested we
        reconstruct the full 4-D center manifold coordinates for every stored point
        and extract the desired components.  This is done on-demand and therefore 
        incurs a modest overhead which is negligible for interactive exploration/plotting workflows.
        """
        # Determine target section
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self.compute(section_coord=section_coord)
            target_section = self.get_section(section_coord)
            prev_sec = self._section
            self._section = target_section
            restore_section = True
        else:
            restore_section = False
            if self._section is None:
                logger.debug("No cached Poincaré-map points found - computing now...")
                self.compute()

        # Default - legacy - behaviour
        if axes is None:
            result = self._section.points
            if restore_section:
                self._section = prev_sec
            return result

        if len(axes) != 2:
            raise ValueError("Exactly two axis names must be provided (e.g. ('q3', 'p2')).")

        # Map variable name -> index in 4-D center manifold coordinates (q2, p2, q3, p3)
        idx_map = {
            "q2": 0, "p2": 1, "q3": 2, "p3": 3,
        }

        try:
            i0, i1 = idx_map[axes[0]], idx_map[axes[1]]
        except KeyError as exc:
            raise ValueError(f"Unknown axis name: {exc.args[0]}. Must be one of q2, p2, q3, p3.") from exc

        # Get section configuration
        from hiten.algorithms.poincare.config import _get_section_config
        from hiten.algorithms.poincare.map import _solve_missing_coord
        config = _get_section_config(self.config.section_coord)
        
        # Get center manifold Hamiltonian for solving missing coordinate
        poly_cm_real = self.cm.compute()

        # Reconstruct the requested coordinates for every section point
        pts_proj = np.empty((len(self), 2), dtype=np.float64)
        for k, pt in enumerate(self._section.points):
            # Build known variables from the stored section point
            known_vars = {config.section_coord: 0.0}  # Section coordinate is zero
            known_vars[config.plane_coords[0]] = float(pt[0])
            known_vars[config.plane_coords[1]] = float(pt[1])
            
            # Solve for the missing coordinate
            solved_val = _solve_missing_coord(
                config.missing_coord, known_vars, self.energy, poly_cm_real, self.cm._clmo
            )
            
            # Build full 4D center manifold coordinates
            full_cm_coords = known_vars.copy()
            full_cm_coords[config.missing_coord] = solved_val
            
            # Extract the requested coordinates (in order q2, p2, q3, p3)
            cm_4d = np.array([
                full_cm_coords["q2"],
                full_cm_coords["p2"], 
                full_cm_coords["q3"],
                full_cm_coords["p3"]
            ])
            
            pts_proj[k, 0] = cm_4d[i0]
            pts_proj[k, 1] = cm_4d[i1]

        if restore_section:
            self._section = prev_sec
        return pts_proj

    def ic(self, pt: np.ndarray, section_coord: Optional[str] = None) -> np.ndarray:
        r"""
        Map a Poincaré point to six dimensional initial conditions.

        Parameters
        ----------
        pt : numpy.ndarray, shape (2,)
            Poincaré section coordinates.
        section_coord : str, optional
            Section coordinate that defines which plane the point lies on.
            If None, uses the config default.

        Returns
        -------
        numpy.ndarray
            Synodic frame state vector of length 6.
        """
        section_key = section_coord or self.config.section_coord
        return self.cm.ic(pt, self.energy, section_coord=section_key)

    def map2ic(self, indices: Optional[Sequence[int]] = None, section_coord: Optional[str] = None) -> np.ndarray:
        r"""
        Convert stored map points to full six dimensional initial conditions.

        Parameters
        ----------
        indices : Sequence[int] or None, optional
            Indices of the points to convert. If *None* all points are used.
        section_coord : str, optional
            Section coordinate to use. If None, uses the most recently computed section.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (:math:`m`, 6) with synodic frame coordinates.

        Raises
        ------
        RuntimeError
            If the map has not been computed yet.
        KeyError
            If the specified section has not been computed.
        """
        # Determine which section to use
        if section_coord is None:
            if self._section is None:
                raise RuntimeError("Poincaré map has not been computed yet - cannot convert.")
            section = self._section
            section_key = self.config.section_coord  # Best guess for the current section
        else:
            section = self.get_section(section_coord)
            section_key = section_coord

        if indices is None:
            sel_pts = section.points
        else:
            sel_pts = section.points[np.asarray(indices, dtype=int)]

        ic_list: List[np.ndarray] = []
        for pt in sel_pts:
            ic = self.cm.ic(pt, self.energy, section_coord=section_key)
            ic_list.append(ic)

        return np.stack(ic_list, axis=0)

    def plot(self, section_coord: Optional[str] = None, dark_mode: bool = True, save: bool = False, filepath: str = 'poincare_map.svg', axes: Optional[Sequence[str]] = None, **kwargs):
        r"""
        Render the 2-D Poincaré map on a selectable pair of axes.

        Parameters
        ----------
        dark_mode : bool, default True
            Use a dark background colour scheme.
        save : bool, default False
            Whether to save the plot to a file.
        filepath : str, default 'poincare_map.svg'
            Path to save the plot to.
        axes : Sequence[str] | None, optional
            Names of the coordinates to visualise (e.g. ("q3", "p2")).  If *None*
            the default pair associated with the section (``self.section.labels``)
            is used.
        **kwargs
            Additional keyword arguments forwarded to
            :pyfunc:`hiten.utils.plots.plot_poincare_map`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure handle.
        matplotlib.axes.Axes
            Axes handle.
        """
        # Select which section to use
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self.compute(section_coord=section_coord)
            section = self.get_section(section_coord)
        else:
            # Default behaviour - use most recent or compute default
            if self._section is None:
                logger.debug("No cached Poincaré-map points found - computing now...")
                self.compute()
            section = self._section

        # Select the requested projection
        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            # get_points still refers to most recent section; temporarily switch
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(tuple(axes), section_coord=section_coord)
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs
        )

    def plot_interactive(self, steps=1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order=6, frame="rotating", dark_mode: bool = True, axes: Optional[Sequence[str]] = None, section_coord: Optional[str] = None):
        r"""
        Interactively select map points and propagate the corresponding orbits.

        Parameters
        ----------
        steps : int, default 1000
            Number of propagation steps for the generated orbit.
        method : {'rk', 'scipy', 'symplectic', 'adaptive'}, default 'scipy'
            _Integrator backend.
        order : int, default 6
            _Integrator order when applicable.
        frame : str, default 'rotating'
            Reference frame used by :pyfunc:`GenericOrbit.plot`.
        dark_mode : bool, default True
            Use dark background colours.
        axes : Sequence[str] | None, optional
            Names of the coordinates to visualise (e.g. ("q3", "p2")).  If *None*
            the default pair associated with the section (``self.section.labels``)
            is used.

        Returns
        -------
        hiten.system.orbits.base.GenericOrbit or None
            The last orbit generated by the selector (None if no point was selected).
        """
        # Ensure desired section exists
        if section_coord is not None:
            if not self.has_section(section_coord):
                logger.debug("Section %s not cached - computing now...", section_coord)
                self.compute(section_coord=section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self.compute()
            section = self._section

        def _on_select(pt_np: np.ndarray):
            """Generate and display an orbit for the selected map point."""
            # Convert the selected point back to the original section coordinates
            if axes is None:
                section_pt = pt_np  # already in section coords
            else:
                prev_sec = self._section
                self._section = section
                try:
                    proj_pts = self.get_points(tuple(axes), section_coord=section_coord)
                finally:
                    self._section = prev_sec
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                closest_idx = np.argmin(distances)
                section_pt = section.points[closest_idx]
            
            orbit = self._propagate_from_point(
                section_pt,
                self.energy,
                steps=steps,
                method=method,
                order=order,
            )

            orbit.plot(
                frame=frame,
                dark_mode=dark_mode,
                block=False,
                close_after=False,
            )

            return orbit

        # Select the requested projection
        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(tuple(axes), section_coord=section_coord)
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        # Launch interactive viewer and return the last selected orbit.
        return plot_poincare_map_interactive(
            points=pts,
            labels=lbls,
            on_select=_on_select,
            dark_mode=dark_mode,
        )

    def save(self, filepath: str, **kwargs) -> None:
        """Serialise the map to *filepath* (HDF5 only)."""
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        _save_poincare_map(self, filepath)

    def load_inplace(self, filepath: str, **kwargs) -> None:
        _load_poincare_map_inplace(self, filepath)

    @classmethod
    def load(cls, filepath: str, cm: "CenterManifold", **kwargs) -> "_PoincareMap":
        return _load_poincare_map(filepath, cm)
