"""Adapter utilities backing invariant torus computations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Tuple

import numpy as np

from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.types.adapters.base import (_CachedDynamicsAdapter,
                                                  _PersistenceAdapterMixin,
                                                  _ServiceBundleBase)
from hiten.utils.log_config import logger
from hiten.utils.io.torus import load_torus, load_torus_inplace, save_torus

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit
    from hiten.system.torus import InvariantTori, Torus


class _TorusPersistenceAdapter(_PersistenceAdapterMixin):
    """Persistence helpers for invariant tori."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda torus, path, **kw: save_torus(torus, Path(path), **kw),
            load_fn=lambda path, **kw: load_torus(Path(path), **kw),
            load_inplace_fn=lambda torus, path, **kw: load_torus_inplace(torus, Path(path), **kw),
        )


class _TorusDynamicsAdapter(_CachedDynamicsAdapter[tuple]):
    """Provide STM preparation and torus construction helpers."""

    def __init__(self, torus: "InvariantTori") -> None:
        super().__init__()
        self._torus = torus
        self._orbit = self._torus.orbit
        self._latest_grid: np.ndarray | None = None
        self._latest_params: dict | None = None

    def eigen_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = self._make_cache_key(id(self._orbit))
        
        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            monodromy = self._orbit.monodromy
            evals, evecs = np.linalg.eig(monodromy)
            return monodromy, evals, evecs
        
        return self._get_or_create(key, _factory)

    def prepare(
        self,
        *,
        n_theta1: int,
        method: str,
        order: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare STM and eigen data for invariant torus construction.
        
        Returns
        -------
        theta1 : np.ndarray
            Theta1 values for the torus parameterization.
        ubar : np.ndarray
            State series along the periodic orbit.
        y_series : np.ndarray
            Complex eigenvector series.
        phi_mats : np.ndarray
            State transition matrices.
        monodromy : np.ndarray
            Monodromy matrix.
        eigenvalues : np.ndarray
            Eigenvalues of the monodromy matrix.
        eigenvectors : np.ndarray
            Eigenvectors of the monodromy matrix.
        """
        cache_key = self._make_cache_key(id(self._orbit), n_theta1, method, order)

        def _factory() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            monodromy, evals, evecs = self.eigen_data()

            logger.info(
                "Pre-computing STM samples for invariant-torus initialisation (n_theta1=%d)",
                n_theta1,
            )

            x_series, times, _, phi_flat = _compute_stm(
                self._orbit.system.var_dynsys,
                self._orbit.initial_state,
                self._orbit.period,
                steps=n_theta1,
                forward=1,
                method=method,
                order=order,
            )

            phi_mats = phi_flat[:, :36].reshape((n_theta1, 6, 6))
            theta1 = 2.0 * np.pi * times / self._orbit.period

            idx, lam_c = self._select_unit_circle_eigenvalue(evals)
            y0 = evecs[:, idx]
            y0 = y0 / np.linalg.norm(y0)

            alpha = np.angle(lam_c)
            phase = np.exp(-1j * alpha * theta1 / (2.0 * np.pi))
            y_series = np.empty((n_theta1, 6), dtype=np.complex128)
            for k in range(n_theta1):
                y_series[k] = phase[k] * phi_mats[k] @ y0

            logger.info("Cached STM and eigen-vector field for torus initialisation.")

            return (
                theta1.copy(),
                x_series.copy(),
                y_series,
                phi_mats,
                monodromy,
                evals,
                evecs,
            )

        return self._get_or_create(cache_key, _factory)

    def state(
        self,
        theta1: float,
        theta2: float,
        *,
        epsilon: float,
        n_theta1: int,
        method: str,
        order: int,
    ) -> np.ndarray:
        theta1_vals, ubar, y_series, _, _, _, _ = self.prepare(n_theta1=n_theta1, method=method, order=order)
        th1 = np.mod(theta1, 2.0 * np.pi)
        th2 = np.mod(theta2, 2.0 * np.pi)

        idx = np.searchsorted(theta1_vals, th1, side="left")
        idx0 = (idx - 1) % len(theta1_vals)
        idx1 = idx % len(theta1_vals)
        t0, t1 = theta1_vals[idx0], theta1_vals[idx1]
        if t1 < t0:
            t1 += 2.0 * np.pi
            if th1 < t0:
                th1 += 2.0 * np.pi
        w = 0.0 if t1 == t0 else (th1 - t0) / (t1 - t0)

        ubar_interp = (1.0 - w) * ubar[idx0] + w * ubar[idx1]
        yvec = (1.0 - w) * y_series[idx0] + w * y_series[idx1]
        yr = np.real(yvec)
        yi = np.imag(yvec)
        uhat = np.cos(th2) * yr - np.sin(th2) * yi
        return ubar_interp + float(epsilon) * uhat

    def compute_grid(
        self,
        *,
        epsilon: float,
        n_theta1: int,
        n_theta2: int,
        method: str,
        order: int,
    ) -> np.ndarray:
        cache_key = self._make_cache_key(epsilon, n_theta1, n_theta2, method, order)
        
        def _factory() -> np.ndarray:
            _, ubar, y_series, _, _, _, _ = self.prepare(n_theta1=n_theta1, method=method, order=order)
            theta2_vals = np.linspace(0.0, 2.0 * np.pi, num=n_theta2, endpoint=False)
            cos_t2 = np.cos(theta2_vals)
            sin_t2 = np.sin(theta2_vals)

            yr = np.real(y_series)
            yi = np.imag(y_series)

            return (
                ubar[:, None, :]
                + epsilon
                * (
                    cos_t2[None, :, None] * yr[:, None, :]
                    - sin_t2[None, :, None] * yi[:, None, :]
                )
            )
        
        grid = self._get_or_create(cache_key, _factory)
        
        # Store the latest grid and parameters
        self._latest_grid = grid
        self._latest_params = {
            'epsilon': epsilon,
            'n_theta1': n_theta1,
            'n_theta2': n_theta2,
            'method': method,
            'order': order,
        }
        
        return grid

    @property
    def grid(self) -> np.ndarray:
        """Return the latest computed grid.
        
        Returns
        -------
        np.ndarray
            The most recently computed torus grid.
            
        Raises
        ------
        ValueError
            If no grid has been computed yet.
        """
        if self._latest_grid is None:
            raise ValueError("No grid has been computed yet. Call compute_grid() first.")
        return self._latest_grid

    def initial_section_curve(
        self,
        *,
        epsilon: float,
        n_theta2: int,
        phi_idx: int,
        n_theta1: int,
        method: str,
        order: int,
    ) -> Tuple[np.ndarray, float]:
        _, ubar, y_series, _, _, _, _ = self.prepare(n_theta1=n_theta1, method=method, order=order)
        theta2_vals = np.linspace(0.0, 2.0 * np.pi, num=n_theta2, endpoint=False)
        cos_t2 = np.cos(theta2_vals)
        sin_t2 = np.sin(theta2_vals)

        ubar_phi = ubar[phi_idx]
        yvec_phi = y_series[phi_idx]
        yr = np.real(yvec_phi)
        yi = np.imag(yvec_phi)

        v_curve = (
            ubar_phi[None, :]
            + epsilon * (cos_t2[:, None] * yr[None, :] - sin_t2[:, None] * yi[None, :])
        )

        _, evals, _ = self.eigen_data()
        idx, lam_c = self._select_unit_circle_eigenvalue(evals)
        _ = idx  # unused but retained for clarity
        rho = np.angle(lam_c)
        return v_curve.astype(float), float(rho)


    def as_torus(
        self,
        *,
        epsilon: float,
        n_theta1: int,
        n_theta2: int,
        method: str,
        order: int,
    ) -> "Torus":
        """
        Return an immutable :class:`~hiten.system.torus.Torus` view of the current grid.

        The fundamental frequencies are derived from the generating periodic
        orbit: omega_1 = 2 * pi / T (longitudinal) and 
        omega_2 = arg(lambda) / T where lambda is the
        complex unit-circle eigenvalue of the monodromy matrix.

        Parameters
        ----------
        epsilon : float
            Amplitude parameter for the torus.
        n_theta1 : int
            Number of points in the theta1 direction.
        n_theta2 : int
            Number of points in the theta2 direction.
        method : str
            Integration method.
        order : int
            Integration order.

        Returns
        -------
        :class:`~hiten.system.torus.Torus`
            Immutable torus representation with computed fundamental frequencies.

        Raises
        ------
        RuntimeError
            If no suitable complex eigenvalue is found in the monodromy matrix.
        """
        # Get the cached grid
        grid = self.compute_grid(
            epsilon=epsilon,
            n_theta1=n_theta1,
            n_theta2=n_theta2,
            method=method,
            order=order,
        )

        # Get eigenvalues for frequency computation
        _, _, _, _, _, evals, _ = self.prepare(n_theta1=n_theta1, method=method, order=order)

        omega_long = 2.0 * np.pi / self._orbit.period

        tol_mag = 1e-6
        cand_idx = [
            i for i, lam in enumerate(evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError(
                "No complex eigenvalue of modulus one found in monodromy matrix - cannot determine Omega_2."
            )

        idx = max(cand_idx, key=lambda i: np.imag(evals[i]))
        lam_c = evals[idx]
        omega_lat = np.angle(lam_c) / self._orbit.period

        omega = np.array([omega_long, omega_lat], dtype=float)

        C0 = self._orbit.jacobi

        return Torus(grid=grid.copy(), omega=omega, C0=C0, system=self._orbit.system)

    @staticmethod
    def _select_unit_circle_eigenvalue(evals: np.ndarray) -> Tuple[int, complex]:
        tol_mag = 1e-6
        candidates: Iterable[int] = (
            idx
            for idx, lam in enumerate(evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        )
        idx_list = list(candidates)
        if not idx_list:
            raise RuntimeError(
                "No complex eigenvalue of modulus one found in monodromy matrix - cannot construct torus."
            )
        idx = max(idx_list, key=lambda i: np.imag(evals[i]))
        return idx, evals[idx]


@dataclass
class _TorusServices(_ServiceBundleBase):
    dynamics: _TorusDynamicsAdapter
    persistence: _TorusPersistenceAdapter

    @classmethod
    def from_torus(cls, torus: "InvariantTori") -> "_TorusServices":
        return cls(
            dynamics=_TorusDynamicsAdapter(torus),
            persistence=_TorusPersistenceAdapter(),
        )

