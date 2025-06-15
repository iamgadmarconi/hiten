from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.map import (compute_poincare_map_for_energy,
                                            generate_iterated_poincare_map)
from utils.log_config import logger


@dataclass
class PoincareMapConfig:
    """Configuration parameters for the Poincaré map generation."""

    # Numerical / integration
    dt: float = 1e-3
    method: str = "symplectic"  # "symplectic" or "rk4"
    integrator_order: int = 6
    c_omega_heuristic: float = 20.0  # Only used by the extended-phase symplectic scheme

    # Seed / iteration control
    use_iterated: bool = False  # True → iterate seeds many crossings, False → dense grid scan

    # Dense grid parameters (use_iterated == False)
    Nq: int = 201
    Np: int = 201
    max_steps: int = 20_000

    # Iterated-seed parameters (use_iterated == True)
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

    @property
    def points(self) -> np.ndarray:
        """Return the computed Poincaré-map points (q2, p2)."""
        if self._points is None:
            raise RuntimeError(
                "Poincaré map has not been computed yet.  Call compute() first."
            )
        return self._points

    def __len__(self) -> int:  # Convenient len() support
        return 0 if self._points is None else self._points.shape[0]

    def compute(self) -> np.ndarray:
        """(Re-)compute the Poincaré map and store the resulting points.

        Returns
        -------
        numpy.ndarray
            Array of shape (M,2) containing the (q2, p2) coordinates of
            successful section crossings.
        """
        logger.info(
            "Generating Poincaré map at energy h0=%.6e (method=%s)",
            self.energy,
            self.config.method,
        )

        # Ensure that the centre manifold is up-to-date (builds & caches poly).
        poly_cm_real = self.cm.compute()

        # Choose which back-end to call.
        if self.config.use_iterated:
            logger.info(
                "Using seed-iteration algorithm: %d seeds, %d crossings",
                self.config.n_seeds,
                self.config.n_iter,
            )
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
        else:
            logger.info(
                "Using dense grid algorithm: Nq=%d, Np=%d",
                self.config.Nq,
                self.config.Np,
            )
            pts = compute_poincare_map_for_energy(
                h0=self.energy,
                H_blocks=poly_cm_real,
                max_degree=self.cm.max_degree,
                psi_table=self.cm.psi,
                clmo_table=self.cm.clmo,
                encode_dict_list=self.cm.encode_dict_list,
                dt=self.config.dt,
                max_steps=self.config.max_steps,
                Nq=self.config.Nq,
                Np=self.config.Np,
                integrator_order=self.config.integrator_order,
                use_symplectic=self._use_symplectic,
            )

        self._points = pts
        logger.info("Poincaré map computation complete: %d points", len(self))
        return pts

    def pm2ic(
        self,
        indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Convert selected Poincaré points to 6-D synodic initial conditions.

        Parameters
        ----------
        indices : sequence of int or None, optional
            Indices of the map points to convert.  *None* (default) converts all
            points.

        Returns
        -------
        numpy.ndarray, shape (K,6)
            The 6-dimensional synodic initial conditions corresponding to the
            chosen Poincaré-section points.
        """
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
