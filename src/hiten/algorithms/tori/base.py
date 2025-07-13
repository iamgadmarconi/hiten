from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator

from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_invariant_torus


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
            obtain the state-transition matrices required by the algorithm.
        """
        if orbit.period is None:
            raise ValueError("The generating orbit must be corrected first (period is None).")

        self._orbit = orbit

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
    def grid(self) -> np.ndarray:
        if self._grid is None:
            err = 'Invariant torus grid not computed. Call `compute()` first.'
            logger.error(err)
            raise ValueError(err)

        return self._grid

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

        M = self.orbit.monodromy
        evals, evecs = np.linalg.eig(M)

        # Tolerance for identifying *unit-circle, non-trivial* eigenvalues.
        tol_mag = 1e-6
        cand_idx: list[int] = [
            i for i, lam in enumerate(evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError("No complex eigenvalue of modulus one found in monodromy matrix - cannot construct torus.")

        # Choose the eigenvalue with positive imaginary part
        idx = max(cand_idx, key=lambda i: np.imag(evals[i]))
        lam_c = evals[idx]
        y0 = evecs[:, idx]

        # Normalise the eigenvector
        y0 = y0 / np.linalg.norm(y0)

        # Angle α such that λ = e^{iα}
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

    def state(self, theta1: float, theta2: float, epsilon: float = 1e-4) -> np.ndarray:
        r"""
        Return the 6-state vector :math:`u_grid(\theta_1, \theta_2)` given by equation (15).

        The angle inputs may lie outside :math:`[0, 2\pi)`; they are wrapped
        automatically. Interpolation is performed along :math:`\theta_1` using the cached
        trajectory samples (linear interpolation is adequate for small torus
        amplitudes).
        """
        # Ensure preparation with default resolution
        self._prepare()

        assert self._theta1 is not None and self._ubar is not None and self._y_series is not None  # mypy

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

        assert self._theta1 is not None and self._ubar is not None and self._y_series is not None

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

    def _compute_pde(
        self,
        *,
        n_theta1: int,
        n_theta2: int,
        order_fd: int = 2,
        newton_tol: float = 1e-10,
        max_iter: int = 25,
        initialise_from_linear: bool = True,
        initial_epsilon: float = 1e-3,
        **kwargs,
    ) -> np.ndarray:
        """
        Iterative discrete-PDE solver (second-order scheme).
        """
        if order_fd not in (2, 4):
            raise NotImplementedError("order_fd must be 2 or 4 (central-difference accuracy)")

        raise NotImplementedError("PDE solver not implemented yet.")

    def compute(
        self,
        *,
        scheme: Literal["linear", "pde"] = "linear",
        epsilon: float = 1e-4,
        n_theta1: int = 256,
        n_theta2: int = 64,
        **kwargs,
    ) -> np.ndarray:
        """Generate and cache a torus grid using the selected *scheme*.

        Parameters
        ----------
        scheme : {'linear', 'pde'}, default 'linear'
            Algorithm to use.  'linear' is the earlier first-order model;
            'pde' will invoke the discrete-PDE Newton solver (not yet implemented).
        epsilon, n_theta1, n_theta2 : see documentation of the linear scheme.
        kwargs : additional parameters forwarded to the chosen backend.
        """

        if scheme == "linear":
            self._grid = self._compute_linear(
                epsilon=epsilon, n_theta1=n_theta1, n_theta2=n_theta2
            )
        elif scheme == "pde":
            if "epsilon" in kwargs:
                logger.warning("epsilon ignored by 'pde' scheme - torus size is inherent to the solution.")
            self._grid = self._compute_pde(
                n_theta1=n_theta1,
                n_theta2=n_theta2,
                **kwargs,
            )

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
