from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional, Tuple

import numpy as np

from hiten.algorithms.corrector.newton import _NewtonCorrector
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
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
        
        # Find complex eigenvalue with unit modulus
        tol_mag = 1e-6
        cand_idx = [
            i for i, lam in enumerate(self._evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError("No complex eigenvalue of modulus one found")
            
        idx = max(cand_idx, key=lambda i: np.imag(self._evals[i]))
        lam = self._evals[idx]

        # Rotation number rho = arg(lambda)
        rho = np.angle(lam)
        
        N = n_theta2  # shorthand

        def _build_rotation_matrix(N: int, rho_val: float) -> np.ndarray:
            """Return dense real rotation matrix R_{-rho} acting on theta2 grid."""
            k_vals = np.arange(N)
            # treat negative frequencies correctly
            k_vals[N // 2 :] -= N
            phase = np.exp(-1j * k_vals * rho_val)  # (N,)
            # apply operator to each basis vector via FFT (still cheap for N<=512)
            eye_N = np.eye(N)
            R = np.fft.ifft(phase[:, None] * np.fft.fft(eye_N, axis=0), axis=0).real
            return R

        # Cache rotation matrices (size-independent of Newton iterates)
        R_theta2 = _build_rotation_matrix(N, rho)
        R_big = np.kron(R_theta2, np.eye(6))  # (N*6, N*6)
        
        # Stroboscopic time T
        T = self.orbit.period

        self._prepare(2)

        theta2_vals = np.linspace(0.0, 2.0 * np.pi, n_theta2, endpoint=False)
        v_curve = np.zeros((n_theta2, 6))
        for j, th2 in enumerate(theta2_vals):
            v_curve[j] = self._state(0.0, th2, epsilon)
        
        # Define the invariance error function
        def invariance_error(v_flat: np.ndarray) -> np.ndarray:
            v = v_flat.reshape(n_theta2, 6)
            
            # Apply stroboscopic map phi_T to each point
            v_mapped = np.zeros_like(v)
            for j in range(n_theta2):
                sol = _propagate_dynsys(
                    dynsys=self.dynsys,
                    state0=v[j],
                    t0=0.0,
                    tf=T,
                    forward=1,
                    steps=2,
                    method=method,
                    order=order,
                )
                v_mapped[j] = sol.states[-1]
            
            # Apply rotation operator R-rho using DFT
            # Forward DFT
            v_dft = np.fft.fft(v_mapped, axis=0)
            
            # Rotation in Fourier space
            k_vals = np.arange(n_theta2)
            # Handle negative frequencies correctly
            k_vals[n_theta2//2:] -= n_theta2
            rotation = np.exp(-1j * k_vals * rho)
            
            # Apply rotation and inverse DFT
            v_rotated = np.fft.ifft(v_dft * rotation[:, None], axis=0).real
            
            # Invariance error
            error = v_rotated - v
            
            # Add phase condition
            # This prevents arbitrary rotations of the solution
            dv_dtheta2 = np.zeros_like(v)
            for j in range(n_theta2):
                j_next = (j + 1) % n_theta2
                j_prev = (j - 1) % n_theta2
                dv_dtheta2[j] = (v[j_next] - v[j_prev]) / (2 * 2*np.pi/n_theta2)
            
            phase_error = np.sum(v * dv_dtheta2) / n_theta2
            
            # Flatten and append phase condition
            return np.concatenate([error.flatten(), [phase_error * n_theta2]])
        
        # Solve using Newton's method
        # Add one extra equation for phase condition
        v_flat_extended = np.concatenate([v_curve.flatten(), [0.0]])
        
        def residual_fn(x):
            # Extract curve and ignore the dummy variable
            return invariance_error(x[:-1])

        def jacobian_fn(x: np.ndarray) -> np.ndarray:
            """Return Jacobian of the GMOS residual at *x* (including phase row)."""
            v = x[:-1].reshape(N, 6)

            # Compute STM for each theta2 node (block diagonal entries)
            Phi_blocks = np.empty((N, 6, 6))
            for j in range(N):
                _, _, Phi_T, _ = _compute_stm(
                    self.libration_point._var_eq_system,
                    v[j],
                    T,
                    steps=2,
                    method=method,
                    order=order,
                )
                Phi_blocks[j] = Phi_T

            dim = 6 * N
            D_map = np.zeros((dim, dim))
            for j in range(N):
                D_map[j * 6 : (j + 1) * 6, j * 6 : (j + 1) * 6] = Phi_blocks[j]

            J_main = R_big @ D_map - np.eye(dim)

            # Phase condition derivative
            dv_dtheta2 = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (
                2 * 2 * np.pi / N
            )
            phase_row = (dv_dtheta2.flatten()) / N

            # Assemble full Jacobian with extra column/row for dummy variable
            J = np.zeros((dim + 1, dim + 1))
            J[:dim, :dim] = J_main
            J[-1, :dim] = phase_row
            # last column already zeros -> derivative wrt dummy variable
            return J
        
        newton = _NewtonCorrector()
        v_corr_flat, info = newton.correct(
            x0=v_flat_extended,
            residual_fn=residual_fn,
            jacobian_fn=jacobian_fn,
            norm_fn=lambda r: float(np.linalg.norm(r) / np.sqrt(n_theta2 * 6 + 1)),
            tol=tol,
            max_attempts=max_iter,
            line_search=line_search,
            max_delta=max_delta,
            alpha_reduction=alpha_reduction,
            min_alpha=min_alpha,
            armijo_c=armijo_c,
        )

        if not info:
            logger.warning("GMOS Newton iteration did not converge")
        
        # Extract corrected invariant curve
        v_curve_corr = v_corr_flat[:-1].reshape(n_theta2, 6)
        
        # Construct full 2-D torus using interpolation that respects the
        logger.info("Constructing 2D torus from invariant curve (interpolation)")

        omega1 = 2.0 * np.pi / T           # longitudinal frequency (unused here)
        omega2 = rho / T                   # latitudinal frequency

        u_grid = np.zeros((n_theta1, n_theta2, 6))
        # row i = 0 corresponds to theta1 = 0 -> just the invariant curve itself
        u_grid[0, :, :] = v_curve_corr

        theta2_vals = np.linspace(0.0, 2.0 * np.pi, n_theta2, endpoint=False)

        for i in range(1, n_theta1):
            t_i = i * T / n_theta1  # time

            for j, theta2_j in enumerate(theta2_vals):
                # Which point on the invariant curve flows to (theta1_i, theta2_j)?
                theta2_src = (theta2_j - omega2 * t_i) % (2.0 * np.pi)

                # Fractional index along the discrete theta2 grid
                idx_f = theta2_src / (2.0 * np.pi) * n_theta2
                j0 = int(np.floor(idx_f)) % n_theta2
                j1 = (j0 + 1) % n_theta2
                w = idx_f - np.floor(idx_f)

                # Linear interpolation between neighbouring nodes of the invariant curve
                x0_interp = (1.0 - w) * v_curve_corr[j0] + w * v_curve_corr[j1]

                # Propagate this initial condition for time t_i
                sol = _propagate_dynsys(
                    dynsys=self.dynsys,
                    state0=x0_interp,
                    t0=0.0,
                    tf=t_i,
                    forward=1,
                    steps=2,
                    method=method,
                    order=order,
                )

                u_grid[i, j] = sol.states[-1]
        
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