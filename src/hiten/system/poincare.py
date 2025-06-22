r"""
hiten.algorithms.poincare.base
=========================

Poincaré return map utilities on the centre manifold of the spatial circular
restricted three body problem.

The module exposes a high level interface :pyclass:`PoincareMap` that wraps
specialised CPU/GPU kernels to generate, query, and visualise discrete
Poincaré sections arising from the reduced Hamiltonian flow. Numerical
parameters are grouped in the lightweight dataclass
:pyclass:`poincareMapConfig`.
"""

import os
import pickle
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from hiten.algorithms.poincare.cuda.map import _generate_map_gpu
from hiten.algorithms.poincare.map import PoincareSection, _generate_grid
from hiten.algorithms.poincare.map import _generate_map as _generate_map_cpu
from hiten.system.center import CenterManifold
from hiten.system.orbits.base import GenericOrbit, orbitConfig
from hiten.utils.files import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import _set_dark_mode


@dataclass
class poincareMapConfig:
    r"""
    Configuration parameters for the Poincaré map generation.
    """

    # Numerical / integration
    dt: float = 1e-2
    method: str = "rk"  # "symplectic" or "rk"
    integrator_order: int = 4
    c_omega_heuristic: float = 20.0  # Only used by the extended-phase symplectic scheme

    n_seeds: int = 20
    n_iter: int = 40
    seed_axis: str = "q2"  # "q2" or "p2"
    section_coord: Literal["q2", "p2", "q3", "p3"] = "q3"  # Default keeps existing behavior

    # Misc
    compute_on_init: bool = False
    use_gpu: bool = False


class PoincareMap:
    r"""
    High-level object representing a Poincaré map on the centre manifold.

    Parameters
    ----------
    cm : CenterManifold
        The centre-manifold object to operate on.  Its polynomial representation is
        used for the reduced Hamiltonian flow.
    energy : float
        Energy level (same convention as :pyfunc:`_solve_missing_coord`, *not* the Jacobi constant).
    config : poincareMapConfig, optional
        Numerical parameters controlling the map generation.  A sensible default
        configuration is used if none is supplied.
    """

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[poincareMapConfig] = None,
    ) -> None:
        self.cm: CenterManifold = cm
        self.energy: float = float(energy)
        self.config: poincareMapConfig = config or poincareMapConfig()

        # Derived flags
        self._use_symplectic: bool = self.config.method.lower() == "symplectic"

        # Storage for computed points
        self._section: Optional[PoincareSection] = None
        self._backend: str = "cpu" if not self.config.use_gpu else "gpu"

        if self.config.compute_on_init:
            self.compute()

    def __repr__(self) -> str:
        return (
            f"PoincareMap(cm={self.cm!r}, energy={self.energy:.3e}, "
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
    def section(self) -> PoincareSection:
        r"""
        Return the computed Poincaré section with labels.
        """
        if self._section is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._section

    def compute(self) -> np.ndarray:
        r"""
        Compute the discrete Poincaré return map.

        Returns
        -------
        numpy.ndarray
            Array of shape (:math:`n`, 2) containing the intersection points.

        Raises
        ------
        RuntimeError
            If the underlying centre manifold computation fails.

        Notes
        -----
        The resulting section is cached in :pyattr:`_section`; subsequent calls
        reuse the stored data.
        """
        logger.info(
            "Generating Poincaré map at energy h0=%.6e (method=%s)",
            self.energy,
            self.config.method,
        )

        poly_cm_real = self.cm.compute()

        kernel = _generate_map_gpu if self._backend == "gpu" else _generate_map_cpu

        section = kernel(
            h0=self.energy,
            H_blocks=poly_cm_real,
            max_degree=self.cm.max_degree,
            psi_table=self.cm.psi,
            clmo_table=self.cm.clmo,
            encode_dict_list=self.cm.encode_dict_list,
            n_seeds=self.config.n_seeds,
            n_iter=self.config.n_iter,
            dt=self.config.dt,
            use_symplectic=self._use_symplectic,
            integrator_order=self.config.integrator_order,
            c_omega_heuristic=self.config.c_omega_heuristic,
            seed_axis=self.config.seed_axis,
            section_coord=self.config.section_coord,
        )

        self._section = section
        logger.info("Poincaré map computation complete: %d points", len(self))
        return section.points  # Return raw points for backward compatibility

    def pm2ic(self, indices: Optional[Sequence[int]] = None) -> np.ndarray:
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

    def grid(self, Nq: int = 201, Np: int = 201, max_steps: int = 20_000) -> np.ndarray:
        r"""
        Generate a dense rectangular grid of the Poincaré map.

        Parameters
        ----------
        Nq, Np : int, default 201
            Number of nodes along the :math:`q` and :math:`p` axes.
        max_steps : int, default 20000
            Maximum number of integration steps for each seed.

        Returns
        -------
        numpy.ndarray
            Array containing the grid points with the same layout as
            :pyattr:`section.points`.

        Raises
        ------
        ValueError
            If an unsupported backend is selected.
        """
        logger.info(
            "Generating *dense-grid* Poincaré map at energy h0=%.6e (Nq=%d, Np=%d)",
            self.energy,
            Nq,
            Np,
        )

        # Ensure that the centre manifold polynomial is current.
        poly_cm_real = self.cm.compute()

        if self._backend == "cpu":
            pts = _generate_grid(
                h0=self.energy,
                H_blocks=poly_cm_real,
                max_degree=self.cm.max_degree,
                psi_table=self.cm.psi,
                clmo_table=self.cm.clmo,
                encode_dict_list=self.cm.encode_dict_list,
                dt=self.config.dt,
                max_steps=max_steps,
                Nq=Nq,
                Np=Np,
                integrator_order=self.config.integrator_order,
                use_symplectic=self._use_symplectic,
                section_coord=self.config.section_coord,
                )
        else:
            raise ValueError(f"Unsupported backend: {self._backend}")

        self._section = pts  # _generate_grid will be updated to return PoincareSection
        logger.info("Dense-grid Poincaré map computation complete: %d points", len(self))
        return pts.points if hasattr(pts, 'points') else pts

    def _propagate_from_point(self, cm_point, energy, steps=1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order=6):
        r"""
        Convert a Poincaré map point to initial conditions, create a GenericOrbit, propagate, and return the orbit.
        """
        ic = self.cm.ic(cm_point, energy, section_coord=self.config.section_coord)
        logger.info(f"Initial conditions: {ic}")
        cfg = orbitConfig(orbit_family="generic", libration_point=self.cm.point)
        orbit = GenericOrbit(cfg, ic)
        if orbit.period is None:
            orbit.period = 2 * np.pi
        orbit.propagate(steps=steps, method=method, order=order)
        return orbit

    def save(self, filepath: str, **kwargs) -> None:
        r"""
        Serialize the current map to disk.

        Parameters
        ----------
        filepath : str
            Destination pickle file.
        **kwargs
            Reserved for future extensions.
        """

        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

        data = {
            "map_type": self.__class__.__name__,
            "energy": self.energy,
            "config": asdict(self.config),
        }

        if self._section is not None:
            data["points"] = self._section.points.tolist()
            data["labels"] = self._section.labels

        # Ensure directory exists.
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)

        logger.info("Poincaré map saved to %s", filepath)

    def load(self, filepath: str, **kwargs) -> None:
        r"""
        Load a map previously stored with :pyfunc:`save`.

        Parameters
        ----------
        filepath : str
            Path to the pickle file.
        **kwargs
            Reserved for future extensions.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Poincaré-map file not found: {filepath}")

        with open(filepath, "rb") as fh:
            data = pickle.load(fh)

        if data.get("map_type") != self.__class__.__name__:
            logger.warning(
                "Loading %s data into %s instance",
                data.get("map_type", "<unknown>"),
                self.__class__.__name__,
            )

        # Update simple attributes.
        self.energy = data["energy"]

        # Reconstruct config dataclass (fall back to defaults if missing).
        cfg_dict = data.get("config", {})
        try:
            self.config = poincareMapConfig(**cfg_dict)
        except TypeError:
            logger.error("Saved configuration is incompatible with current poincareMapConfig schema; using defaults.")
            self.config = poincareMapConfig()

        # Refresh derived flags dependent on config.
        self._use_symplectic = self.config.method.lower() == "symplectic"

        # Load points (if present).
        if "points" in data and data["points"] is not None:
            points_array = np.array(data["points"])
            # Try to load labels, fall back to default q3 section labels for backward compatibility
            labels = data.get("labels", ("q2", "p2"))
            self._section = PoincareSection(points_array, labels)
        else:
            self._section = None
        logger.info("Poincaré map loaded from %s", filepath)

    def plot(self, dark_mode: bool = True, output_dir: Optional[str] = None, filename: Optional[str] = None, **kwargs):
        r"""
        Render the 2-D Poincaré map.

        Parameters
        ----------
        dark_mode : bool, default True
            Use a dark background colour scheme.
        output_dir : str or None, optional
            Folder where the figure is saved. If *None* the plot is only
            displayed.
        filename : str or None, optional
            Name of the output image inside *output_dir*.
        **kwargs
            Additional arguments forwarded to :pyfunc:`matplotlib.pyplot.scatter`.

        Returns
        -------
        matplotlib.figure.Figure
            Figure handle.
        matplotlib.axes.Axes
            Axes handle.
        """
        if self._section is None:
            logger.debug("No cached Poincaré-map points found - computing now …")
            self.compute()

        # Create a single plot for this energy level
        fig, ax = plt.subplots(figsize=(6, 6))
        
        points = self._section.points
        labels = self._section.labels
        
        if points.shape[0] == 0:
            logger.info(f"No points to plot for h0={self.energy:.6e}.")
            ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', 
                    color='red' if not dark_mode else 'white', fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            title_text = "Poincaré Map (No Data)"
        else:
            n_pts = points.shape[0]
            point_size = max(0.2, min(4.0, 4000.0 / max(n_pts, 1)))

            ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.7)
            
            max_val_0 = max(abs(points[:, 0].max()), abs(points[:, 0].min()))
            max_val_1 = max(abs(points[:, 1].max()), abs(points[:, 1].min()))
            max_abs_val = max(max_val_0, max_val_1, 1e-9)
            
            ax.set_xlim(-max_abs_val * 1.1, max_abs_val * 1.1)
            ax.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
            
            ax.set_xlabel(f"${labels[0]}'$")
            ax.set_ylabel(f"${labels[1]}'$")
            title_text = f"Poincaré Map (h={self.energy:.6e})"

        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        
        if dark_mode:
            _set_dark_mode(fig, ax, title=title_text)
        else:
            ax.set_title(title_text)

        plt.tight_layout()

        if output_dir and filename:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            filepath = os.path.join(output_dir, filename)
            try:
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Poincaré map saved to {filepath}")
            except Exception as e:
                logger.error(f"Error saving Poincaré map to {filepath}: {e}")

        plt.show()
        return fig, ax

    def plot_interactive(self, steps=1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order=6, frame="rotating", dark_mode: bool = True):
        r"""
        Interactively select map points and propagate the corresponding orbits.

        Parameters
        ----------
        steps : int, default 1000
            Number of propagation steps for the generated orbit.
        method : {'rk', 'scipy', 'symplectic', 'adaptive'}, default 'scipy'
            Integrator backend.
        order : int, default 6
            Integrator order when applicable.
        frame : str, default 'rotating'
            Reference frame used by :pyfunc:`GenericOrbit.plot`.
        dark_mode : bool, default True
            Use dark background colours.

        Returns
        -------
        hiten.system.orbits.base.GenericOrbit
            The last orbit generated by the selector.
        """
        if self._section is None:
            self.compute()
        fig, ax = plt.subplots(figsize=(6, 6))
        pts = self._section.points
        labels = self._section.labels
        # Adaptive point size: scale inversely with number of points
        n_pts_int = pts.shape[0]
        adaptive_ps = max(0.2, min(4.0, 4000.0 / max(n_pts_int, 1)))

        scatter = ax.scatter(pts[:, 0], pts[:, 1], s=adaptive_ps, alpha=0.7)
        ax.set_xlabel(f"${labels[0]}'$")
        ax.set_ylabel(f"${labels[1]}'$")
        ax.set_title(f"Select a point on the Poincaré Map (h={self.energy:.6e})\n(Press 'q' to quit)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        
        # Apply dark-mode styling if requested (match other plotting routines)
        if dark_mode:
            _set_dark_mode(fig, ax, title=ax.get_title())
        
        selected_marker = ax.scatter([], [], s=60, c='red', marker='x')
        selected_orbit = {'orbit': None}
        def onclick(event):
            if event.inaxes != ax:
                return
            x, y = event.xdata, event.ydata
            dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
            idx = np.argmin(dists)
            pt = pts[idx]
            ax.scatter([pt[0]], [pt[1]], s=60, c='red', marker='x')
            fig.canvas.draw()
            print(f"Selected Poincaré point: {pt}")
            orbit = self._propagate_from_point(pt, self.energy, steps=steps, method=method, order=order)
            orbit.plot(frame=frame, show=True, dark_mode=dark_mode)
            selected_orbit['orbit'] = orbit
            print("Orbit propagation and plot complete.")
        def onkey(event):
            if event.key == 'q':
                print("Quitting Poincaré map selection window.")
                plt.close(fig)
        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', onkey)
        plt.show()
        return selected_orbit['orbit']

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