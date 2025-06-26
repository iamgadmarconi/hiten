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
    seed_strategy: Literal["single", "axis_aligned", "level_sets", "radial", "random"] = "single"
    seed_axis: Optional[Literal["q2", "p2", "q3", "p3"]] = "q2"
    section_coord: Literal["q2", "p2", "q3", "p3"] = "q3"

    compute_on_init: bool = False
    use_gpu: bool = False

    def __post_init__(self):
        if self.seed_strategy == "single" and self.seed_axis is None:
            raise ValueError("seed_axis must be specified when seed_strategy is 'single'")
        
        elif self.seed_strategy != 'single':
            if self.seed_axis is not None:
                logger.warning("seed_axis is ignored when seed_strategy is not 'single'")


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
        self.energy: float = float(energy)
        self.config: _PoincareMapConfig = config or _PoincareMapConfig()

        # Derived flags
        self._use_symplectic: bool = self.config.method.lower() == "symplectic"

        # Storage for computed points
        self._section: Optional[_PoincareSection] = None
        self._grid: Optional[np.ndarray] = None
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
        return self._grid

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

    def compute(self, **kwargs) -> np.ndarray:
        r"""
        Compute the discrete Poincaré return map.

        Parameters
        ----------
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
        The resulting section is cached in :pyattr:`_section`; subsequent calls
        reuse the stored data. Parallel processing is enabled automatically for
        CPU computations.
        """
        logger.info(
            "Generating Poincaré map at energy h0=%.6e (method=%s, cpu_parallel=%s)",
            self.energy,
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
            section_coord=self.config.section_coord)

        logger.info("Poincaré map computation complete: %d points", len(self))
        return self._section.points

    def compute_grid(self, Nq: int = 201, Np: int = 201, max_steps: int = 20_000, **kwargs) -> np.ndarray:
        r"""
        Generate a dense rectangular grid of the Poincaré map.

        Parameters
        ----------
        Nq, Np : int, default 201
            Number of nodes along the :math:`q` and :math:`p` axes.
        max_steps : int, default 20000
            Maximum number of integration steps for each seed.
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
        Parallel processing is enabled automatically for CPU computations.
        """
        if self._backend == "gpu":
            raise ValueError("GPU backend does not support CPU parallel processing.")
            
        logger.info(
            "Generating *dense-grid* Poincaré map at energy h0=%.6e (Nq=%d, Np=%d)",
            self.energy,
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
            section_coord=self.config.section_coord,
            )

        logger.info("Dense-grid Poincaré map computation complete: %d points", len(self))
        return self._grid

    def ic(self, pt: np.ndarray) -> np.ndarray:
        r"""
        Map a Poincaré point to six dimensional initial conditions.

        Parameters
        ----------
        pt : numpy.ndarray, shape (2,)
            Poincaré section coordinates.

        Returns
        -------
        numpy.ndarray
            Synodic frame state vector of length 6.
        """
        return self.cm.ic(pt, self.energy, section_coord=self.config.section_coord)
    
    def map2ic(self, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        r"""
        Convert stored map points to full six dimensional initial conditions.

        Parameters
        ----------
        indices : Sequence[int] or None, optional
            Indices of the points to convert. If *None* all points are used.

        Returns
        -------
        numpy.ndarray
            Matrix of shape (:math:`m`, 6) with synodic frame coordinates.

        Raises
        ------
        RuntimeError
            If the map has not been computed yet.
        """
        if self._section is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet - cannot convert.")

        if indices is None:
            sel_pts = self._section.points
        else:
            sel_pts = self._section.points[np.asarray(indices, dtype=int)]

        ic_list: List[np.ndarray] = []
        for pt in sel_pts:
            ic = self.cm.ic(pt, self.energy, section_coord=self.config.section_coord)
            ic_list.append(ic)

        return np.stack(ic_list, axis=0)

    def get_points(self, axes: Sequence[str] | None = None) -> np.ndarray:
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
        if self._section is None:
            # Automatically compute if the map has not been generated yet
            logger.debug("No cached Poincaré-map points found - computing now...")
            self.compute()

        # Default - legacy - behaviour
        if axes is None:
            return self._section.points

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

        return pts_proj

    def plot(self, dark_mode: bool = True, save: bool = False, filepath: str = 'poincare_map.svg', axes: Optional[Sequence[str]] = None, **kwargs):
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
        if self._section is None:
            logger.debug("No cached Poincaré-map points found - computing now...")
            self.compute()

        # Select the requested projection
        if axes is None:
            pts = self._section.points
            lbls = self._section.labels
        else:
            pts = self.get_points(tuple(axes))
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs
        )

    def plot_interactive(self, steps=1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order=6, frame="rotating", dark_mode: bool = True, axes: Optional[Sequence[str]] = None):
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
        # Ensure Poincaré-map points are available.
        if self._section is None:
            self.compute()

        def _on_select(pt_np: np.ndarray):
            """Generate and display an orbit for the selected map point."""
            # Convert the selected point back to the original section coordinates
            if axes is None:
                # Direct use of stored section point
                section_pt = pt_np
            else:
                # Need to find the corresponding section point
                # This is a bit tricky - we need to reverse the projection
                # For now, we'll use the first point that matches closely
                proj_pts = self.get_points(tuple(axes))
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                closest_idx = np.argmin(distances)
                section_pt = self._section.points[closest_idx]
            
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
            pts = self._section.points
            lbls = self._section.labels
        else:
            pts = self.get_points(tuple(axes))
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
