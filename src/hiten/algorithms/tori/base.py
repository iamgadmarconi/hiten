from dataclasses import dataclass
from typing import Callable, Literal, NamedTuple, Optional, Tuple

import numpy as np

from hiten.algorithms.corrector.newton import _NewtonCorrector
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.dynamics.utils.energy import (crtbp_energy,
                                                    energy_to_jacobi)
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_invariant_torus


class _ToriCorrectionConfig(NamedTuple):
    """Default numerical parameters for invariant-torus Newton solves."""

    max_iter: int = 100  # Maximum Newton iterations
    tol: float = 1e-8  # Convergence tolerance on the residual
    method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy"
    order: int = 4

    # Line-search / step-control parameters
    line_search: bool = False
    max_delta: float = None
    alpha_reduction: float = None
    min_alpha: float = None
    armijo_c: float = None

    def __post_init__(self):
        if self.line_search and (self.max_delta is None or self.alpha_reduction is None or self.min_alpha is None or self.armijo_c is None):
            raise ValueError("Line-search parameters must be provided if line_search is True")


@dataclass(slots=True, frozen=True)
class _Torus:
    r"""
    Immutable representation of a 2-D invariant torus.

    Parameters
    ----------
    grid : np.ndarray
        Real 6-state samples of shape (n_theta1, n_theta2, 6).
    omega : np.ndarray
        Fundamental frequencies (ω₁, ω₂).
    C0 : float
        Jacobi constant (fixed along the torus family).
    system : System
        Parent CR3BP system (useful for downstream algorithms).
    """

    grid: np.ndarray
    omega: np.ndarray
    C0: float
    system: System


class _InvariantTori:

    def __init__(self, orbit: PeriodicOrbit):
        r"""
        Linear approximation of a 2-D invariant torus bifurcating from a
        centre component of a periodic orbit.

        Parameters
        ----------
        orbit : PeriodicOrbit
            *Corrected* periodic orbit about which the torus is constructed. The
            orbit must expose a valid `period` attribute - no propagation is
            performed here; we only integrate the *variational* equations to
            obtain the _state-transition matrices required by the algorithm.
        """
        if orbit.period is None:
            raise ValueError("The generating orbit must be corrected first (period is None).")

        self._orbit = orbit
        self._monodromy = self.orbit.monodromy
        self._evals, self._evecs = np.linalg.eig(self._monodromy)
        self._dynsys = self.system.dynsys

        # Internal caches populated lazily by _prepare().
        self._theta1: Optional[np.ndarray] = None  # angle along the periodic orbit
        self._ubar: Optional[np.ndarray] = None   # periodic-orbit trajectory samples
        self._y_series: Optional[np.ndarray] = None  # complex eigen-vector field y(\theta_1)
        self._grid: Optional[np.ndarray] = None

    def __str__(self) -> str:
        return f"InvariantTori object for seed orbit={self.orbit} at point={self.libration_point})"

    def __repr__(self) -> str:
        return f"InvariantTori(orbit={self.orbit}, point={self.libration_point})"

    @property
    def orbit(self) -> PeriodicOrbit:
        return self._orbit

    @property
    def libration_point(self) -> LibrationPoint:
        return self._orbit.libration_point

    @property
    def system(self) -> System:
        return self._orbit.system
    
    @property
    def dynsys(self):
        return self._dynsys
    
    @property
    def grid(self) -> np.ndarray:
        if self._grid is None:
            err = 'Invariant torus grid not computed. Call `compute()` first.'
            logger.error(err)
            raise ValueError(err)

        return self._grid
    
    @property
    def period(self) -> float:
        return float(self.orbit.period)
    
    @property
    def jacobi(self) -> float:
        return float(self.orbit.jacobi_constant)

    @property
    def rotation_number(self) -> float | None:
        """Latitudinal rotation number rho (set after GMOS computation)."""
        return getattr(self, "_rotation_number", None)
    
    def as_state(self) -> _Torus:
        r"""
        Return an immutable :class:`_Torus` view of the current grid.

        The fundamental frequencies are derived from the generating periodic
        orbit: :math:`\omega_1 = 2 \pi / T` (longitudinal) and 
        :math:`\omega_2 = \arg(\lambda) / T` where :math:`\lambda` is the
        complex unit-circle eigenvalue of the monodromy matrix.
        """

        # Ensure a torus grid is available.
        if self._grid is None:
            raise ValueError("Invariant torus grid not computed. Call `compute()` first.")

        omega_long = 2.0 * np.pi / self.period

        tol_mag = 1e-6
        cand_idx = [
            i for i, lam in enumerate(self._evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError(
                "No complex eigenvalue of modulus one found in monodromy matrix - cannot determine ω₂."
            )

        idx = max(cand_idx, key=lambda i: np.imag(self._evals[i]))
        lam_c = self._evals[idx]
        omega_lat = np.angle(lam_c) / self.period

        omega = np.array([omega_long, omega_lat], dtype=float)

        C0 = self.jacobi

        # Return an *immutable* copy of the grid to avoid accidental mutation.
        return _Torus(grid=self._grid.copy(), omega=omega, C0=C0, system=self.system)

    def _prepare(self, n_theta1: int = 256, *, method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy", order: int = 8) -> None:
        r"""
        Compute the trajectory, STM samples :math:`\Phi_{\theta_1}(0)` and the rotated
        eigen-vector field :math:`y(\theta_1)` required by the torus parameterisation.

        This routine is executed once and cached; subsequent calls with the
        same *n_theta1* return immediately.
        """
        if self._theta1 is not None and len(self._theta1) == n_theta1:
            # Cached - nothing to do.
            return

        logger.info("Pre-computing STM samples for invariant-torus initialisation (n_theta1=%d)", n_theta1)

        x_series, times, _, PHI_flat = _compute_stm(
            self.libration_point._var_eq_system,
            self.orbit.initial_state,
            self.orbit.period,
            steps=n_theta1,
            forward=1,
            method=method,
            order=order,
        )

        # Convert to convenient shapes
        PHI_mats = PHI_flat[:, :36].reshape(n_theta1, 6, 6)  # \Phi(t) for each sample

        # Non-dimensional angle \theta_1 along the periodic orbit
        theta1 = 2.0 * np.pi * times / self.orbit.period  # shape (n_theta1,)

        # Tolerance for identifying *unit-circle, non-trivial* eigenvalues.
        tol_mag = 1e-6
        cand_idx: list[int] = [
            i for i, lam in enumerate(self._evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError("No complex eigenvalue of modulus one found in monodromy matrix - cannot construct torus.")

        # Choose the eigenvalue with positive imaginary part
        idx = max(cand_idx, key=lambda i: np.imag(self._evals[i]))
        lam_c = self._evals[idx]
        y0 = self._evecs[:, idx]

        # Normalise the eigenvector
        y0 = y0 / np.linalg.norm(y0)

        # Angle α such that \lambda = e^{iα}
        alpha = np.angle(lam_c)

        phase = np.exp(-1j * alpha * theta1 / (2.0 * np.pi))  # shape (n_theta1,)
        y_series = np.empty((n_theta1, 6), dtype=np.complex128)
        for k in range(n_theta1):
            y_series[k] = phase[k] * PHI_mats[k] @ y0

        # Cache results as immutable copies
        self._theta1 = theta1.copy()
        self._ubar = x_series.copy()  # real trajectory samples
        self._y_series = y_series.copy()

        logger.info("Cached STM and eigen-vector field for torus initialisation.")

    def _state(self, theta1: float, theta2: float, epsilon: float = 1e-4) -> np.ndarray:
        r"""
        Return the 6-_state vector :math:`u_grid(\theta_1, \theta_2)` given by equation (15).

        The angle inputs may lie outside :math:`[0, 2\pi)`; they are wrapped
        automatically. Interpolation is performed along :math:`\theta_1` using the cached
        trajectory samples (linear interpolation is adequate for small torus
        amplitudes).
        """
        # Ensure preparation with default resolution
        self._prepare()
        # Wrap angles
        th1 = np.mod(theta1, 2.0 * np.pi)
        th2 = np.mod(theta2, 2.0 * np.pi)

        # Locate neighbouring indices for linear interpolation
        idx = np.searchsorted(self._theta1, th1, side="left")
        idx0 = (idx - 1) % len(self._theta1)
        idx1 = idx % len(self._theta1)
        t0, t1 = self._theta1[idx0], self._theta1[idx1]
        # Handle wrap-around at 2\pi
        if t1 < t0:
            t1 += 2.0 * np.pi
            if th1 < t0:
                th1 += 2.0 * np.pi
        w = 0.0 if t1 == t0 else (th1 - t0) / (t1 - t0)

        ubar = (1.0 - w) * self._ubar[idx0] + w * self._ubar[idx1]
        yvec = (1.0 - w) * self._y_series[idx0] + w * self._y_series[idx1]

        # Real/imag parts
        yr = np.real(yvec)
        yi = np.imag(yvec)

        # Perturbation :math:`\hat{u_grid}(\theta_1, \theta_2)`
        uhat = np.cos(th2) * yr - np.sin(th2) * yi

        return ubar + float(epsilon) * uhat

    def _compute_linear(self, *, epsilon: float, n_theta1: int, n_theta2: int) -> np.ndarray:
        """Return the first-order torus grid (current implementation)."""

        # Ensure STM cache at requested resolution
        self._prepare(n_theta1)

        th2_vals = np.linspace(0.0, 2.0 * np.pi, num=n_theta2, endpoint=False)
        cos_t2 = np.cos(th2_vals)
        sin_t2 = np.sin(th2_vals)

        yr = np.real(self._y_series)  # (n_theta1, 6)
        yi = np.imag(self._y_series)  # (n_theta1, 6)

        u_grid = (
            self._ubar[:, None, :]
            + epsilon
            * (
                cos_t2[None, :, None] * yr[:, None, :]
                - sin_t2[None, :, None] * yi[:, None, :]
            )
        )
        return u_grid

    def _compute_gmos(
        self,
        *,
        epsilon: float = 1e-3,
        n_theta1: int = 64,
        n_theta2: int = 256,
        max_iter: int | None = None,
        tol: float | None = None,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] | None = None,
        order: int | None = None,
        max_delta: float | None = None,
        alpha_reduction: float | None = None,
        min_alpha: float | None = None,
        armijo_c: float | None = None,
        line_search: bool | None = None,
    ) -> np.ndarray:
        """
        Compute quasi-periodic invariant torus using GMOS algorithm.
        
        This implementation follows the paper's formulation more closely with proper
        continuation support and robust initial guess generation.
        """
        # Resolve numerical parameters using the module-level defaults when not supplied
        cfg = _ToriCorrectionConfig()

        max_iter = cfg.max_iter if max_iter is None else max_iter
        tol = cfg.tol if tol is None else tol
        method = cfg.method if method is None else method
        order = cfg.order if order is None else order
        max_delta = cfg.max_delta if max_delta is None else max_delta
        alpha_reduction = cfg.alpha_reduction if alpha_reduction is None else alpha_reduction
        min_alpha = cfg.min_alpha if min_alpha is None else min_alpha
        armijo_c = cfg.armijo_c if armijo_c is None else armijo_c
        line_search = cfg.line_search if line_search is None else line_search

        logger.info("Computing invariant torus using GMOS algorithm")
        logger.info(
            "GMOS parameters: epsilon=%g, n_theta1=%d, n_theta2=%d, max_iter=%d, tol=%.1e",
            epsilon, n_theta1, n_theta2, max_iter, tol
        )
        raise NotImplementedError("GMOS algorithm not implemented yet.")
    
    def _compute_kkg(self, *, epsilon: float, n_theta1: int, n_theta2: int) -> np.ndarray:
        """Compute invariant torus using the KKG algorithm."""
        raise NotImplementedError("KKG algorithm not implemented yet.")

    def compute(
        self,
        *,
        scheme: Literal["linear", "gmos", "kkg"] = "linear",
        epsilon: float = 1e-4,
        n_theta1: int = 256,
        n_theta2: int = 64,
        **kwargs,
    ) -> np.ndarray:
        """Generate and cache a torus grid using the selected *scheme*.

        Parameters
        ----------
        scheme : {'linear', 'gmos', 'kkg'}, default 'linear'
            Algorithm to use.  'linear' is the earlier first-order model;
            'gmos' is the GMOS algorithm;
            'kkg' is the KKG algorithm.
        epsilon : float, default 1e-4
            Amplitude of the torus
        n_theta1 : int, default 256
            Number of points along periodic orbit (longitudinal)
        n_theta2 : int, default 64
            Number of points in transverse direction (latitudinal)
        kwargs : additional parameters forwarded to the chosen backend.
        """

        if scheme == "linear":
            self._grid = self._compute_linear(epsilon=epsilon, n_theta1=n_theta1, n_theta2=n_theta2)
        elif scheme == "gmos":
            self._grid = self._compute_gmos(
                epsilon=epsilon, 
                n_theta1=n_theta1, 
                n_theta2=n_theta2,
                **kwargs
            )
        elif scheme == "kkg":
            self._grid = self._compute_kkg(epsilon=epsilon, n_theta1=n_theta1, n_theta2=n_theta2)

        return self._grid

    def plot(
        self,
        *,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = "invariant_torus.svg",
        **kwargs,
    ):
        r"""
        Render the invariant torus using :pyfunc:`hiten.utils.plots.plot_invariant_torus`.

        Parameters
        ----------
        figsize, save, dark_mode, filepath : forwarded to the plotting helper.
        **kwargs : Additional keyword arguments accepted by
            :pyfunc:`hiten.utils.plots.plot_invariant_torus`.
        """
        return plot_invariant_torus(
            self.grid,
            [self.system.primary, self.system.secondary],
            self.system.distance,
            figsize=figsize,
            save=save,
            dark_mode=dark_mode,
            filepath=filepath,
            **kwargs,
        )

    def _plot_diagnostics(
        self,
        v_curve_initial: np.ndarray,
        v_curve_corr: np.ndarray,
        save_path: str = "gmos_diagnostics.png",
    ) -> None:
        """Plot diagnostic information comparing initial and corrected curves.

        Parameters
        ----------
        v_curve_initial : numpy.ndarray
            Invariant curve obtained from the linear approximation, shape *(N, 6)*.
        v_curve_corr : numpy.ndarray
            Curve after GMOS Newton correction, shape *(N, 6)*.
        save_path : str, default 'gmos_diagnostics.png'
            File path where the figure is saved.
        """

        import matplotlib.pyplot as plt

        theta2_vals = np.linspace(0.0, 2 * np.pi, len(v_curve_corr))

        # Helper for Jacobi constant computation
        def _jacobi(state: np.ndarray) -> float:
            from hiten.algorithms.dynamics.utils.energy import crtbp_energy, energy_to_jacobi
            return energy_to_jacobi(crtbp_energy(state, self.system.mu))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Initial vs corrected curve in x-y plane
        ax = axes[0, 0]
        ax.plot(v_curve_initial[:, 0], v_curve_initial[:, 1], "b-", label="Initial", alpha=0.5)
        ax.plot(v_curve_corr[:, 0], v_curve_corr[:, 1], "r-", label="Corrected")
        ax.plot(self.orbit.initial_state[0], self.orbit.initial_state[1], "ko", markersize=8, label="Periodic orbit")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Invariant Curve (x-y plane)")
        ax.legend()
        ax.axis("equal")

        # Plot 2: Jacobi constant along curve
        ax = axes[0, 1]
        jacobi_vals = [_jacobi(state) for state in v_curve_corr]
        ax.plot(theta2_vals, jacobi_vals, "g-")
        ax.axhline(self.jacobi, color="k", linestyle="--", label="Orbit Jacobi")
        ax.set_xlabel("θ₂")
        ax.set_ylabel("Jacobi constant")
        ax.set_title("Jacobi Constant Variation")
        ax.legend()

        # Plot 3: Distance from periodic orbit
        ax = axes[1, 0]
        distances = np.linalg.norm(v_curve_corr - self.orbit.initial_state, axis=1)
        ax.plot(theta2_vals, distances, "m-")
        ax.set_xlabel("θ₂")
        ax.set_ylabel("Distance from periodic orbit")
        ax.set_title("Curve Amplitude")

        # Plot 4: Invariance error (per point)
        ax = axes[1, 1]
        errors = []
        rho = self.rotation_number if self.rotation_number is not None else 0.0
        for j, state in enumerate(v_curve_corr):
            sol = _propagate_dynsys(
                dynsys=self.dynsys,
                state0=state,
                t0=0.0,
                tf=self.period,
                forward=1,
                steps=2,
                method="scipy",
                order=4,
            )
            j_target = int(np.round(j + rho * len(v_curve_corr) / (2 * np.pi))) % len(v_curve_corr)
            errors.append(np.linalg.norm(sol.states[-1] - v_curve_corr[j_target]))

        ax.semilogy(theta2_vals, errors, "c-")
        ax.set_xlabel("θ₂")
        ax.set_ylabel("Invariance error")
        ax.set_title("Point-wise Invariance Error")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _plot_gmos_diagnostics(self, save_path: str = "gmos_diagnostics.png") -> None:
        """Plot GMOS diagnostic information using stored curves from the last computation.

        This is a convenience method that uses the curves stored during the most recent
        GMOS computation. Call this after `compute(scheme='gmos', ...)`.

        Parameters
        ----------
        save_path : str, default 'gmos_diagnostics.png'
            File path where the figure is saved.

        Raises
        ------
        ValueError
            If GMOS has not been run yet (no stored curves available).
        """
        if not hasattr(self, '_v_curve_initial') or not hasattr(self, '_v_curve_corrected'):
            raise ValueError(
                "No GMOS curves available. Run `compute(scheme='gmos', ...)` first."
            )
        
        self.plot_diagnostics(
            self._v_curve_initial, 
            self._v_curve_corrected, 
            save_path=save_path
        )