from dataclasses import dataclass
from dataclasses import asdict
from typing import List, Optional, Sequence

import numpy as np

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.map import (compute_poincare_map_for_energy,
                                            generate_iterated_poincare_map)
from utils.log_config import logger
from plots.plots import plot_poincare_map

# Standard-library helpers for serialisation
import os
import pickle


@dataclass
class PoincareMapConfig:
    """Configuration parameters for the Poincaré map generation."""

    # Numerical / integration
    dt: float = 1e-3
    method: str = "symplectic"  # "symplectic" or "rk4"
    integrator_order: int = 6
    c_omega_heuristic: float = 20.0  # Only used by the extended-phase symplectic scheme

    n_seeds: int = 20
    n_iter: int = 1500
    seed_axis: str = "q2"  # "q2" or "p2"

    # Misc
    compute_on_init: bool = True


class PoincareMap:
    """High-level object representing a Poincaré map on the centre manifold.

    Parameters
    ----------
    cm : CenterManifold
        The centre-manifold object to operate on.  Its polynomial representation is
        used for the reduced Hamiltonian flow.
    energy : float
        Energy level (same convention as `solve_p3`, *not* the Jacobi constant).
    config : PoincareMapConfig, optional
        Numerical parameters controlling the map generation.  A sensible default
        configuration is used if none is supplied.
    """

    def __init__(
        self,
        cm: CenterManifold,
        energy: float,
        config: Optional[PoincareMapConfig] = None,
    ) -> None:
        self.cm: CenterManifold = cm
        self.energy: float = float(energy)
        self.config: PoincareMapConfig = config or PoincareMapConfig()

        # Derived flags
        self._use_symplectic: bool = self.config.method.lower() == "symplectic"

        # Storage for computed points
        self._points: Optional[np.ndarray] = None  # shape (M,2)

        if self.config.compute_on_init:
            self.compute()

    def __repr__(self) -> str:
        return (
            f"PoincareMap(cm={self.cm!r}, energy={self.energy:.3e}, "
            f"points={len(self) if self._points is not None else '∅'})"
        )

    def __str__(self) -> str:
        return (
            f"Poincaré map at h0={self.energy:.3e} with {len(self)} points"
            if self._points is not None
            else f"Poincaré map (uncomputed) at h0={self.energy:.3e}"
        )

    def __len__(self) -> int:  # Convenient len() support
        return 0 if self._points is None else self._points.shape[0]

    @property
    def points(self) -> np.ndarray:
        """Return the computed Poincaré-map points (q2, p2)."""
        if self._points is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._points

    def compute(self) -> np.ndarray:
        logger.info(
            "Generating Poincaré map at energy h0=%.6e (method=%s)",
            self.energy,
            self.config.method,
        )

        poly_cm_real = self.cm.compute()

        pts = generate_iterated_poincare_map(
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
        )

        self._points = pts
        logger.info("Poincaré map computation complete: %d points", len(self))
        return pts

    def pm2ic(self, indices: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._points is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet - cannot convert.")

        if indices is None:
            sel_pts = self._points
        else:
            sel_pts = self._points[np.asarray(indices, dtype=int)]

        ic_list: List[np.ndarray] = []
        for pt in sel_pts:
            ic = self.cm.cm2ic(pt, self.energy)
            ic_list.append(ic)

        return np.stack(ic_list, axis=0)

    def grid(self, Nq: int = 201, Np: int = 201, max_steps: int = 20_000) -> np.ndarray:
        logger.info(
            "Generating *dense-grid* Poincaré map at energy h0=%.6e (Nq=%d, Np=%d)",
            self.energy,
            Nq,
            Np,
        )

        # Ensure that the centre manifold polynomial is current.
        poly_cm_real = self.cm.compute()

        pts = compute_poincare_map_for_energy(
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
        )

        self._points = pts
        logger.info("Dense-grid Poincaré map computation complete: %d points", len(self))
        return pts

    def plot(self, dark_mode: bool = True, output_dir: Optional[str] = None, filename: Optional[str] = None, **kwargs):

        if self._points is None:
            logger.debug("No cached Poincaré-map points found - computing now …")
            self.compute()

        # Call the shared plotting utility.  The helper expects *lists* of
        # point arrays/levels, so we wrap our single dataset accordingly.
        fig, axs = plot_poincare_map(
            pts_list=[self._points],
            h0_levels=[self.energy],
            dark_mode=dark_mode,
            output_dir=output_dir,
            filename=filename,
            **kwargs,
        )

        ax = axs[0] if isinstance(axs, list) and len(axs) > 0 else axs

        return fig, ax

    def save(self, filepath: str, **kwargs) -> None:

        data = {
            "map_type": self.__class__.__name__,
            "energy": self.energy,
            "config": asdict(self.config),
        }

        if self._points is not None:
            data["points"] = self._points.tolist()

        # Ensure directory exists.
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)

        logger.info("Poincaré map saved to %s", filepath)

    def load(self, filepath: str, **kwargs) -> None:
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
            self.config = PoincareMapConfig(**cfg_dict)
        except TypeError:
            logger.error("Saved configuration is incompatible with current PoincareMapConfig schema; using defaults.")
            self.config = PoincareMapConfig()

        # Refresh derived flags dependent on config.
        self._use_symplectic = self.config.method.lower() == "symplectic"

        # Load points (if present).
        if "points" in data and data["points"] is not None:
            self._points = np.array(data["points"])
        else:
            self._points = None
        logger.info("Poincaré map loaded from %s", filepath)
