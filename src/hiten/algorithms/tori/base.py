from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.corrector.newton import _NewtonCorrector
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_invariant_torus


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
        max_iter: int = 50,
        tol: float = 1e-12,
        method: Literal["scipy", "rk", "symplectic", "adaptive"] = "scipy",
        order: int = 8,
        # Newton–GMOS stabilisation parameters (cf. PeriodicOrbit.correct)
        max_delta: float = 1e-2,
        alpha_reduction: float = 0.5,
        min_alpha: float = 1e-4,
        armijo_c: float = 0.1,
    ) -> np.ndarray:
        """
        Compute quasi-periodic invariant torus using GMOS algorithm.
        
        This implements the algorithm from Gómez-Mondelo (2001) and Olikara-Scheeres (2012),
        computing invariant curves of a stroboscopic map.
        
        Parameters
        ----------
        epsilon : float, default 1e-4
            Initial amplitude of the torus
        n_theta1 : int, default 256
            Number of points along the periodic orbit (longitudinal)
        n_theta2 : int, default 64
            Number of points in the transverse direction (latitudinal)
        max_iter : int, default 50
            Maximum Newton iterations
        tol : float, default 1e-12
            Convergence tolerance
        method : str, default "scipy"
            Integration method
        order : int, default 8
            Integration order
            
        Returns
        -------
        np.ndarray
            The computed torus grid of shape (n_theta1, n_theta2, 6)
        """
        logger.info("Computing invariant torus using GMOS algorithm")
        # High-level summary of the run parameters
        logger.info(
            "GMOS parameters: epsilon=%g, n_theta1=%d, n_theta2=%d, max_iter=%d, tol=%.1e",
            epsilon,
            n_theta1,
            n_theta2,
            max_iter,
            tol,
        )
        
        # Get monodromy matrix and eigenvalues/vectors
        M = self._monodromy
        evals, evecs = np.linalg.eig(M)
        
        # Find complex eigenvalue with unit modulus
        tol_mag = 1e-6
        cand_idx = [
            i for i, lam in enumerate(evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError("No complex eigenvalue of modulus one found")
            
        idx = max(cand_idx, key=lambda i: np.imag(evals[i]))
        lam = evals[idx]
        w = evecs[:, idx]
        
        # Rotation number \rho = arg(\lambda)
        rho = np.angle(lam)
        
        # Stroboscopic time T (period of underlying orbit)
        T = self.orbit.period
        
        # Frequencies
        omega0 = 2.0 * np.pi / T  # Longitudinal frequency
        omega1 = rho / T          # Latitudinal frequency
        
        # Initialize invariant curve using equation (4) from paper
        theta1 = np.linspace(0, 2*np.pi, n_theta2, endpoint=False)
        
        # Get initial states along periodic orbit
        x0 = self.orbit.initial_state
        
        # Create initial guess for invariant curve
        X0 = np.zeros((n_theta2, 6))
        for j in range(n_theta2):
            # Linear approximation from monodromy eigenvector
            perturbation = epsilon * (np.cos(theta1[j]) * np.real(w) - 
                                     np.sin(theta1[j]) * np.imag(w))
            X0[j] = x0 + perturbation

        def _error_2d(curve_2d: np.ndarray) -> np.ndarray:
            """Invariance residual for a (n_theta2,6) curve."""
            X1 = np.zeros_like(curve_2d)
            for j in range(n_theta2):
                sol = _propagate_dynsys(
                    dynsys=self.dynsys,
                    state0=curve_2d[j],
                    t0=0.0,
                    tf=T,
                    forward=1,
                    steps=2,
                    method=method,
                    order=order,
                )
                X1[j] = sol.states[-1]

            X1_dft = np.fft.fft(X1, axis=0)
            k_vals = np.fft.fftfreq(n_theta2, 1 / n_theta2)
            rotation = np.exp(-1j * rho * k_vals)
            X1_rotated = np.real(np.fft.ifft(X1_dft * rotation[:, None], axis=0))
            return (X1_rotated - curve_2d).flatten()

        newton = _NewtonCorrector()
        flat_corr, _ = newton.correct(
            x0=X0.flatten(),
            residual_fn=lambda v: _error_2d(v.reshape(n_theta2, 6)),
            jacobian_fn=None,  # finite-difference is fine here
            norm_fn=lambda r: float(np.linalg.norm(r) / n_theta2),
            tol=tol,
            max_attempts=max_iter,
            max_delta=max_delta,
            alpha_reduction=alpha_reduction,
            min_alpha=min_alpha,
            armijo_c=armijo_c,
        )

        X0 = flat_corr.reshape(n_theta2, 6)
        
        # Construct full 2D torus from invariant curve
        logger.info("Constructing 2D torus grid from invariant curve")
        # The invariant curve gives us one slice at \theta_0 = 0
        # We generate the full torus by propagating along \theta_0
        
        u_grid = np.zeros((n_theta1, n_theta2, 6))
        dt = T / n_theta1
        
        for i in range(n_theta1):
            t = i * dt
            for j in range(n_theta2):
                sol = _propagate_dynsys(
                    dynsys=self.dynsys,
                    state0=X0[j],
                    t0=0.0,
                    tf=t,
                    forward=1,
                    steps=2,
                    method=method,
                    order=order,
                )
                # Account for rotation in \theta_1
                theta1_shift = omega1 * t
                j_shifted = int(np.round(j + theta1_shift * n_theta2 / (2 * np.pi))) % n_theta2
                u_grid[i, j_shifted] = sol.states[-1]
        
        return u_grid
    
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