r"""
hiten.algorithms.poincare.base
=========================

Poincaré return map utilities on the centre manifold of the spatial circular
restricted three body problem.

The module exposes a high level interface :pyclass:`_PoincareMap` that wraps
specialised CPU/GPU kernels to generate, query, and visualise discrete
Poincaré sections arising from the reduced Hamiltonian flow. Numerical
parameters are grouped in the lightweight dataclass
:pyclass:`_CenterManifoldMapConfig`.
"""

import os
from typing import Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare import _build_seeding_strategy
from hiten.algorithms.poincare.cm.backend import \
    _CenterManifoldBackend
from hiten.algorithms.poincare.cm.config import (_CenterManifoldMapConfig,
                                              _get_section_config)
from hiten.algorithms.poincare.core.base import _Section
from hiten.algorithms.poincare.cm.engine import _CenterManifoldEngine
from hiten.algorithms.poincare.events import _PlaneEvent
from hiten.system.center import CenterManifold
from hiten.system.libration.triangular import TriangularPoint
from hiten.system.orbits.base import GenericOrbit
from hiten.utils.io import (_ensure_dir, _load_poincare_map,
                            _load_poincare_map_inplace, _save_poincare_map)
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive


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
    config : _CenterManifoldMapConfig, optional
        Numerical parameters controlling the map generation.  A sensible default
        configuration is used if none is supplied.
    """

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[_CenterManifoldMapConfig] = None,
    ) -> None:
        self.cm: CenterManifold = cm
        if isinstance(self.cm.point, TriangularPoint):
            raise ValueError("Poincaré map is not supported for triangular points.")
        self._energy: float = float(energy)

        # Configuration
        self.config: _CenterManifoldMapConfig = config or _CenterManifoldMapConfig()

        # Dictionaries keyed by section coordinate ("q2", "p2", "q3", "p3")
        self._sections: dict[str, _Section] = {}
        self._engines: dict[str, _CenterManifoldEngine] = {}

        # Keep a pointer to the *most recently computed* section for legacy
        self._section: Optional[_Section] = None

        if self.config.compute_on_init:
            self.compute()

    def __repr__(self) -> str:
        return (
            f"_PoincareMap(cm={self.cm!r}, energy={self.energy:.3e}, "
            f"points={len(self) if self._section is not None else '0'})"
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
    def energy(self) -> float:
        return self._energy

    @property
    def sections(self) -> dict[str, _Section]:
        r"""
        Return the computed Poincaré sections.
        """
        return self._sections

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

    def compute(self, section_coord: Optional[str] = None, *args, **kwargs) -> np.ndarray:
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
        section_key: str = section_coord or self.config.section_coord

        if section_key in self._sections:
            logger.info("Using cached Poincaré section for %s", section_key)
            self._section = self._sections[section_key]
            return self._section.points

        logger.info(
            "Generating Poincaré map via engine backend (h0=%.6e, section=%s, method=%s)",
            self.energy,
            section_key,
            self.config.method,
        )

        if section_key not in self._engines:
            dynsys = self.cm._get_hamsys("center_manifold_real")

            surface = _PlaneEvent(coord=section_key, value=0.0, direction=None)

            backend = _CenterManifoldBackend(
                dynsys=dynsys,
                surface=surface,
                section_coord=section_key,
                h0=self.energy,
                method=self.config.method,
                order=self.config.order,
                c_omega_heuristic=self.config.c_omega_heuristic,
                *args
            )

            sec_cfg = _get_section_config(section_key)
            seed_strategy = _build_seeding_strategy(sec_cfg, self.config)

            engine = _CenterManifoldEngine(
                backend=backend,
                seed_strategy=seed_strategy,
                n_iter=self.config.n_iter,
                dt=self.config.dt,
                **kwargs
            )

            self._engines[section_key] = engine

        engine = self._engines[section_key]

        self._section = engine.compute_section()

        self._sections[section_key] = self._section

        logger.info(
            "Poincaré map computation complete: %d points (section=%s)",
            len(self),
            section_key,
        )
        return self._section.points

    def get_section(self, section_coord: str) -> _Section:
        r"""
        Retrieve a specific Poincaré section by coordinate.

        Parameters
        ----------
        section_coord : {"q2", "p2", "q3", "p3"}
            The section coordinate to retrieve.

        Returns
        -------
        _Section
            The cached section object.

        Raises
        ------
        KeyError
            If the requested section has not been computed.
        """
        if section_coord not in self._sections:
            raise KeyError(f"Section '{section_coord}' has not been computed. Available: {list(self._sections.keys())}")
        return self._sections[section_coord]

    def list_sections(self) -> list[str]:
        r"""
        List all computed section coordinates.

        Returns
        -------
        list[str]
            List of section coordinate names that have been computed.
        """
        return list(self._sections.keys())

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

    def clear_cache(self) -> None:
        r"""
        Clear all cached sections and grids.
        """
        logger.info("Clearing Poincaré map cache")
        self._sections.clear()
        self._engines.clear()
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
        # Ensure the requested section is available; compute on-demand.
        sec_key = section_coord or self.config.section_coord
        if sec_key not in self._sections:
            logger.debug("Section %s not cached. Computing now...", sec_key)
            self.compute(section_coord=sec_key)

        section = self._sections[sec_key]

        # Update legacy pointer
        self._section = section

        if axes is None:
            return section.points

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

        config = _get_section_config(sec_key)
        
        # Get center manifold Hamiltonian for solving missing coordinate
        poly_cm_real = self.cm.compute()

        # Reconstruct the requested coordinates for every section point
        pts_proj = np.empty((len(self), 2), dtype=np.float64)
        for k, pt in enumerate(self._section.points):
            # Build known variables from the stored section point
            known_vars = {config.section_coord: 0.0}  # Section coordinate is zero
            known_vars[config.plane_coords[0]] = float(pt[0])
            known_vars[config.plane_coords[1]] = float(pt[1])
            
            # Use the backend helper for the *same section* currently processed.
            engine = self._engines.get(sec_key)
            if engine is None:
                self.compute(section_coord=sec_key)
                engine = self._engines[sec_key]

            solved_val = engine._backend._solve_missing_coord(  # type: ignore[attr-defined]
                config.missing_coord,
                known_vars,
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
