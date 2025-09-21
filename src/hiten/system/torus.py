"""High-level utilities for computing invariant tori in the circular restricted
three-body problem.

This module provides comprehensive tools for computing 2D invariant tori that
bifurcate from periodic orbits in the circular restricted three-body problem.
The implementation supports both linear approximation methods and advanced
algorithms like GMOS (Generalized Method of Characteristics) and KKG.

The torus is parameterized by two angles:
- theta1: longitudinal angle along the periodic orbit
- theta2: latitudinal angle in the transverse direction

The torus surface is given by:
u(theta1, theta2) = ubar(theta1) + epsilon * (cos(theta2) * Re(y(theta1)) - sin(theta2) * Im(y(theta1)))

where ubar is the periodic orbit trajectory and y is the complex eigenvector field.

Notes
-----
The module implements both linear approximation methods and advanced algorithms
for computing invariant tori. The linear approximation provides a good starting
point for more sophisticated methods.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_invariant_torus


@dataclass(slots=True, frozen=True)
class Torus:
    """
    Immutable representation of a 2-D invariant torus.

    This class represents a 2D invariant torus in the circular restricted
    three-body problem, parameterized by two angular coordinates theta1 and theta2.
    The torus is defined by a grid of state vectors and fundamental frequencies.

    Parameters
    ----------
    grid : numpy.ndarray
        Real 6-state samples of shape (n_theta1, n_theta2, 6).
        Each point represents a state vector on the torus surface.
    omega : numpy.ndarray
        Fundamental frequencies (omega_1, omega_2) in nondimensional units.
        omega_1 is the longitudinal frequency, omega_2 is the latitudinal frequency.
    C0 : float
        Jacobi constant (fixed along the torus family) in nondimensional units.
    system : System
        Parent CR3BP system (useful for downstream algorithms).

    Notes
    -----
    The torus is parameterized by two angles:
    - theta1: longitudinal angle along the periodic orbit
    - theta2: latitudinal angle in the transverse direction

    The fundamental frequencies determine the quasi-periodic motion on the torus.
    """

    grid: np.ndarray
    omega: np.ndarray
    C0: float
    system: System


class InvariantTori:
    """
    Linear approximation of a 2-D invariant torus bifurcating from a
    centre component of a periodic orbit.

    This class implements the computation of invariant tori in the circular
    restricted three-body problem using linear approximation methods. The torus
    is constructed from a periodic orbit by analyzing the monodromy matrix
    and computing the associated eigenvector field.

    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        Corrected periodic orbit about which the torus is constructed. The
        orbit must expose a valid period attribute - no propagation is
        performed here; we only integrate the variational equations to
        obtain the state-transition matrices required by the algorithm.

    Mathematical Background
    ----------------------
    The invariant torus is parameterized by two angles:
    - theta1: longitudinal angle along the periodic orbit
    - theta2: latitudinal angle in the transverse direction

    The torus surface is given by:
    u(theta1, theta2) = ubar(theta1) + epsilon * (cos(theta2) * Re(y(theta1)) - sin(theta2) * Im(y(theta1)))

    where ubar is the periodic orbit trajectory and y is the complex eigenvector field.

    References
    ----------
    Szebehely, V. (1967). *Theory of Orbits*. Academic Press.
    """

    def __init__(self, orbit: PeriodicOrbit):
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

        # Continuation bookkeeping for pseudo-arclength.
        self._v_curve_prev: Optional[np.ndarray] = None  # previous invariant curve
        self._family_tangent: Optional[np.ndarray] = None  # tangent along torus family

    def __str__(self) -> str:
        return f"InvariantTori object for seed orbit={self.orbit} at point={self.libration_point})"

    def __repr__(self) -> str:
        return f"InvariantTori(orbit={self.orbit}, point={self.libration_point})"

    @property
    def orbit(self) -> PeriodicOrbit:
        """Periodic orbit about which the torus is constructed."""
        return self._orbit

    @property
    def libration_point(self) -> LibrationPoint:
        """Libration point anchoring the family."""
        return self._orbit.libration_point

    @property
    def system(self) -> System:
        """Parent CR3BP system."""
        return self._orbit.system
    
    @property
    def dynsys(self):
        """Dynamical system."""
        return self._dynsys
    
    @property
    def grid(self) -> np.ndarray:
        """Invariant torus grid."""
        if self._grid is None:
            err = 'Invariant torus grid not computed. Call `compute()` first.'
            logger.error(err)
            raise ValueError(err)

        return self._grid
    
    @property
    def period(self) -> float:
        """Orbit period."""
        return float(self.orbit.period)
    
    @property
    def jacobi(self) -> float:
        """Jacobi constant."""
        return float(self.orbit.jacobi_constant)

    @property
    def rotation_number(self) -> float | None:
        """Latitudinal rotation number rho (set after GMOS computation)."""
        return getattr(self, "_rotation_number", None)
    
    def as_torus(self) -> Torus:
        """
        Return an immutable :class:`~hiten.system.torus.Torus` view of the current grid.

        The fundamental frequencies are derived from the generating periodic
        orbit: omega_1 = 2 * pi / T (longitudinal) and 
        omega_2 = arg(lambda) / T where lambda is the
        complex unit-circle eigenvalue of the monodromy matrix.

        Returns
        -------
        :class:`~hiten.system.torus.Torus`
            Immutable torus representation with computed fundamental frequencies.

        Raises
        ------
        ValueError
            If the torus grid has not been computed yet.
        RuntimeError
            If no suitable complex eigenvalue is found in the monodromy matrix.
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
                "No complex eigenvalue of modulus one found in monodromy matrix - cannot determine Omega_2."
            )

        idx = max(cand_idx, key=lambda i: np.imag(self._evals[i]))
        lam_c = self._evals[idx]
        omega_lat = np.angle(lam_c) / self.period

        omega = np.array([omega_long, omega_lat], dtype=float)

        C0 = self.jacobi

        # Return an *immutable* copy of the grid to avoid accidental mutation.
        return Torus(grid=self._grid.copy(), omega=omega, C0=C0, system=self.system)

    def _prepare(self, n_theta1: int = 256, *, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8) -> None:
        """
        Compute the trajectory, STM samples Phi_theta1(0) and the rotated
        eigen-vector field y(theta1) required by the torus parameterisation.

        This routine is executed once and cached; subsequent calls with the
        same n_theta1 return immediately.

        Parameters
        ----------
        n_theta1 : int, default 256
            Number of discretization points along the periodic orbit.
        method : {'fixed', 'adaptive', 'symplectic'}, default 'adaptive'
            Integration method for computing the state transition matrix.
        order : int, default 8
            Order of the integration method.

        Notes
        -----
        This method computes the state transition matrix samples and the
        complex eigenvector field required for the torus parameterization.
        The results are cached to avoid recomputation.
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

        PHI_mats = PHI_flat[:, :36].reshape((n_theta1, 6, 6))

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

        # Angle alpha such that \lambda = e^{i*alpha}
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
        """
        Return the 6-state vector u_grid(theta1, theta2) given by equation (15).

        The angle inputs may lie outside [0, 2*pi); they are wrapped
        automatically. Interpolation is performed along theta1 using the cached
        trajectory samples (linear interpolation is adequate for small torus
        amplitudes).

        Parameters
        ----------
        theta1 : float
            Longitudinal angle along the periodic orbit.
        theta2 : float
            Latitudinal angle in the transverse direction.
        epsilon : float, default 1e-4
            Amplitude parameter for the torus.

        Returns
        -------
        numpy.ndarray
            6D state vector on the torus surface.

        Notes
        -----
        The state vector is computed using linear interpolation along the
        periodic orbit and the complex eigenvector field.
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

        uhat = np.cos(th2) * yr - np.sin(th2) * yi

        return ubar + float(epsilon) * uhat

    def compute(self, *, epsilon: float, n_theta1: int, n_theta2: int) -> np.ndarray:
        """Compute the invariant torus grid.
        
        Parameters
        ----------
        epsilon : float
            Torus amplitude used in the linear approximation.
        n_theta1 : int
            Number of discretisation points along theta1.
        n_theta2 : int
            Number of discretisation points along theta2.

        Returns
        -------
        numpy.ndarray
            Invariant torus grid.

        Notes
        -----
        This method computes the invariant torus grid using the linear approximation.
        The grid is computed using the cached STM samples and the complex eigenvector field.
        The grid is cached for subsequent plotting and state export.
        """

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
        # Cache computed grid for plotting and state export
        self._grid = u_grid
        return u_grid

    def _initial_section_curve(
        self,
        *,
        epsilon: float,
        n_theta2: int,
        phi_idx: int = 0,
    ) -> tuple[np.ndarray, float]:
        """Return initial curve v(theta2) on the surface of section theta1 = phi.

        Parameters
        ----------
        epsilon : float
            Torus amplitude used in the linear approximation.
        n_theta2 : int
            Number of discretisation points along theta2.
        phi_idx : int, optional
            Index of the longitudinal angle theta1 that defines the section.
            By default the first sample (corresponding to phi = 0) is chosen.

        Returns
        -------
        v_curve : numpy.ndarray, shape (n_theta2, 6)
            The initial invariant curve obtained from the linear torus model.
        rho : float
            Initial estimate of the rotation number rho obtained from the
            complex eigenvalue of the monodromy matrix.

        Notes
        -----
        This method computes the initial invariant curve on a Poincare section
        defined by theta1 = phi. The curve is obtained from the linear
        approximation of the torus.
        """
        # Ensure the STM and eigenvector field are ready so we can access
        # self._ubar and self._y_series.
        if self._theta1 is None:
            self._prepare()

        # Discretisation in theta2.
        theta2_vals = np.linspace(0.0, 2.0 * np.pi, num=n_theta2, endpoint=False)
        cos_t2 = np.cos(theta2_vals)
        sin_t2 = np.sin(theta2_vals)

        ubar_phi = self._ubar[phi_idx]
        yvec_phi = self._y_series[phi_idx]
        yr = np.real(yvec_phi)
        yi = np.imag(yvec_phi)

        v_curve = (
            ubar_phi[None, :]
            + epsilon * (cos_t2[:, None] * yr[None, :] - sin_t2[:, None] * yi[None, :])
        )

        tol_mag = 1e-6
        cand_idx = [
            i
            for i, lam in enumerate(self._evals)
            if abs(abs(lam) - 1.0) < tol_mag and abs(np.imag(lam)) > tol_mag
        ]
        if not cand_idx:
            raise RuntimeError("Cannot compute rotation number: suitable eigenvalue not found.")
        idx = max(cand_idx, key=lambda i: np.imag(self._evals[i]))
        lam_c = self._evals[idx]
        rho = np.angle(lam_c)

        return v_curve.astype(float), float(rho)

    def plot(
        self,
        *,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = "invariant_torus.svg",
        **kwargs,
    ):
        """
        Render the invariant torus using :func:`~hiten.utils.plots.plot_invariant_torus`.

        Parameters
        ----------
        figsize : Tuple[int, int], default (10, 8)
            Figure size in inches.
        save : bool, default False
            Whether to save the plot to a file.
        dark_mode : bool, default True
            Whether to use dark mode styling.
        filepath : str, default "invariant_torus.svg"
            File path for saving the plot.
        **kwargs : dict
            Additional keyword arguments accepted by
            :func:`~hiten.utils.plots.plot_invariant_torus`.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
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
