r"""
hiten.system.manifold
===============

Stable/unstable invariant manifolds of periodic orbits in the spatial circular
restricted three-body problem.

The module offers a high-level interface (:pyclass:`Manifold`) that, given a
generating :pyclass:`PeriodicOrbit`, launches trajectory
integrations along the selected eigen-directions, records their intersections
with the canonical Poincaré section, provides quick 3-D visualisation, and
handles (de)serialisation through :pyfunc:`Manifold.save` /
:pyfunc:`Manifold.load`.

References
----------
Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2016). "Dynamical Systems, the Three-Body Problem
and Space Mission Design".
"""

import os
from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.dynamics.utils.energy import _max_rel_energy_error
from hiten.algorithms.dynamics.utils.linalg import (_totime,
                                                    eigenvalue_decomposition)
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io.common import _ensure_dir
from hiten.utils.io.manifold import load_manifold, save_manifold
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_manifold


@dataclass
class ManifoldResult:
    r"""
    Output container produced by :pyfunc:`Manifold.compute`.

    Attributes
    ----------
    ysos : list[float]
        :math:`y`-coordinates of Poincaré section crossings.
    dysos : list[float]
        Corresponding :math:`\dot y` values.
    states_list : list[numpy.ndarray]
        Propagated state arrays, one per trajectory.
    times_list : list[numpy.ndarray]
        Time grids associated with *states_list*.
    _successes : int
        Number of trajectories that intersected the section.
    _attempts : int
        Total number of trajectories launched.

    Notes
    -----
    The :pyattr:`success_rate` property returns
    :math:`\frac{_successes}{\max(1,\,_attempts)}`.
    """
    ysos: List[float]
    dysos: List[float]
    states_list: List[float]
    times_list: List[float]
    _successes: int
    _attempts: int

    @property
    def success_rate(self) -> float:
        return self._successes / max(self._attempts, 1)
    
    def __iter__(self):
        return iter((self.ysos, self.dysos, self.states_list, self.times_list))


class Manifold:
    r"""
    Compute and cache the invariant manifold of a periodic orbit.

    Parameters
    ----------
    generating_orbit : :pyclass:`PeriodicOrbit`
        Orbit that seeds the manifold.
    stable : bool, default True
        ``True`` selects the stable manifold, ``False`` the unstable one.
    direction : {{'positive', 'negative'}}, default 'positive'
        Sign of the eigenvector used to initialise the manifold branch.
    method : {{'rk', 'scipy', 'symplectic', 'adaptive'}}, default 'scipy'
        Backend integrator passed to :pyfunc:`_propagate_dynsys`.
    order : int, default 6
        Integration order for fixed-step Runge-Kutta methods.

    Attributes
    ----------
    generating_orbit : :pyclass:`PeriodicOrbit`
        Orbit that seeds the manifold.
    libration_point : :pyclass:`LibrationPoint`
        Libration point associated with *generating_orbit*.
    stable, direction : int
        Encoded form of the options in :pyclass:`Manifold`.
    mu : float
        Mass ratio of the underlying CRTBP system.
    method, order
        Numerical integration settings.
    manifold_result : :pyclass:`ManifoldResult` or None
        Cached result returned by the last successful
        :pyfunc:`compute` call.

    Notes
    -----
    Re-invoking :pyfunc:`compute` after a successful run returns the cached
    :pyclass:`ManifoldResult` without recomputation.
    """

    def __init__(
            self, 
            generating_orbit: PeriodicOrbit, 
            stable: bool = True, 
            direction: Literal["positive", "negative"] = "positive", 
        ):
        self._generating_orbit = generating_orbit
        self._libration_point = self._generating_orbit.libration_point
        self._stable = 1 if stable else -1
        self._direction = 1 if direction == "positive" else -1
        self._mu = self._generating_orbit.system.mu

        self._forward = -self._stable
        self._successes = 0
        self._attempts = 0
        self._last_compute_params: dict = None
        self._manifold_result: ManifoldResult = None

    @property
    def generating_orbit(self) -> PeriodicOrbit:
        """Orbit that seeds the manifold."""
        return self._generating_orbit

    @property
    def libration_point(self):
        """Libration point associated with the generating orbit."""
        return self._libration_point

    @property
    def stable(self) -> int:
        """Encoded stability: 1 for stable, -1 for unstable."""
        return self._stable

    @property
    def direction(self) -> int:
        """Encoded direction: 1 for 'positive', -1 for 'negative'."""
        return self._direction

    @property
    def mu(self) -> float:
        """Mass ratio of the underlying CRTBP system."""
        return self._mu

    @property
    def manifold_result(self) -> ManifoldResult:
        """Cached result from the last successful compute call."""
        return self._manifold_result

    def __str__(self):
        return f"Manifold(stable={self._stable}, direction={self._direction}) of {self._generating_orbit}"
    
    def __repr__(self):
        return self.__str__()

    def _get_real_eigenvectors(self, vectors: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return eigenvalues/eigenvectors with zero imaginary part (vectorised)."""
        mask = np.isreal(values)

        # Eigenvalues that are real within numerical precision
        real_vals_arr = values[mask].astype(np.complex128)

        # Corresponding eigenvectors (may be none)
        if np.any(mask):
            real_vecs_arr = vectors[:, mask]
        else:
            real_vecs_arr = np.zeros((vectors.shape[0], 0), dtype=np.complex128)

        return real_vals_arr, real_vecs_arr

    def _compute_manifold_section(
        self,
        period: float,
        fraction: float,
        displacement: float,
        xx: np.ndarray,
        tt: np.ndarray,
        PHI: np.ndarray,
        eigvec: np.ndarray,
    ):
        r"""
        Compute a section of the invariant manifold.

        Parameters
        ----------
        period : float
            Period of the periodic orbit.
        fraction : float
            Fraction of the period to compute the section at.
        xx, tt, PHI, phi_T : numpy.ndarray
            Output of a previous `_compute_stm` call for the same periodic orbit.
            Supplying these arrays lets the function skip the expensive STM
            propagation.
        eigvec : numpy.ndarray
            Pre-selected eigenvector of the monodromy matrix.

        Returns
        -------
        numpy.ndarray
            Initial condition displaced along the invariant-manifold branch.

        Raises
        ------
        ValueError
            If the requested eigenvector is not available.
        """
        mfrac = _totime(tt, fraction * period)
        
        if np.isscalar(mfrac):
            mfrac_idx = mfrac
        else:
            mfrac_idx = mfrac[0]

        phi_frac_flat = PHI[mfrac_idx, :36]
        phi_frac = phi_frac_flat.reshape((6, 6))

        MAN = self._direction * (phi_frac @ eigvec)

        disp_magnitude = np.linalg.norm(MAN[0:3])

        if disp_magnitude < 1e-14:
            logger.warning(f"Very small displacement magnitude: {disp_magnitude:.2e}, setting to 1.0")
            disp_magnitude = 1.0
        d = displacement / disp_magnitude

        fracH = xx[mfrac_idx, :].copy()

        x0W = fracH + d * MAN.real
        x0W = x0W.flatten()
        
        if abs(x0W[2]) < 1.0e-15:
            x0W[2] = 0.0
        if abs(x0W[5]) < 1.0e-15:
            x0W[5] = 0.0

        return x0W

    def compute(self, step: float = 0.02, integration_fraction: float = 0.75, NN: int = 1, displacement: float = 1e-6, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order: int = 6, **kwargs):
        r"""
        Generate manifold trajectories and build a Poincaré map.

        The routine samples the generating orbit at equally spaced fractions
        of its period, displaces each point by *displacement* along the
        selected eigenvector and integrates the resulting initial condition
        for *integration_fraction* of one synodic period.

        Parameters
        ----------
        step : float, optional
            Increment of the dimensionless fraction along the orbit. Default
            0.02 (i.e. 50 samples per orbit).
        integration_fraction : float, optional
            Portion of :math:`2\pi` non-dimensional time units to integrate
            each trajectory. Default 0.75.
        NN : int, default 1
            Index of the real eigenvector to follow (1-based).
        displacement : float, default 1e-6
            Dimensionless displacement applied along the eigenvector.
        **kwargs
            Additional options:

            show_progress : bool, default True
                Display a :pydata:`tqdm` progress bar.
            dt : float, default 1e-3
                Nominal time step for fixed-step integrators.
            energy_tol : float, default 1e-6
                Maximum relative variation of the Jacobi constant allowed along a trajectory.
                Larger deviations indicate numerical error (often triggered by near-singular
                passages) and cause the trajectory to be discarded.
            safe_distance : float, default 2.0
                Safety multiplier applied to the physical radii of both primaries.  A trajectory
                is rejected if it ever comes within ``safe_distance x radius`` of either body.

        Returns
        -------
        ManifoldResult
            See above.

        Raises
        ------
        ValueError
            If called after a previous run with incompatible settings.

        Examples
        --------
        >>> from hiten.system import System, Manifold
        >>> system = System.from_bodies("earth", "moon")
        >>> L2 = system.get_libration_point(2)
        >>> halo_L2 = L2.create_orbit('halo', amplitude_z=0.3, zenith='northern')
        >>> halo_L2.correct()
        >>> halo_L2.propagate()
        >>> manifold = halo_L2.manifold(stable=True, direction='positive')
        >>> result = manifold.compute(step=0.05)
        >>> print(f"Success rate: {result.success_rate:.0%}")
        """
        kwargs.setdefault("show_progress", True)
        kwargs.setdefault("dt", 1e-3)
        kwargs.setdefault("energy_tol", 1e-6)
        kwargs.setdefault("safe_distance", 2.0)

        dist_m = self._generating_orbit.system.distance * 1e3
        pr_nd = self._generating_orbit.system.primary.radius / dist_m
        sr_nd = self._generating_orbit.system.secondary.radius / dist_m
        safe_r1 = kwargs["safe_distance"] * pr_nd
        safe_r2 = kwargs["safe_distance"] * sr_nd
        current_params = {
            "step": step,
            "integration_fraction": integration_fraction,
            "NN": NN,
            "displacement": displacement,
            "safe_distance": kwargs["safe_distance"],
            **kwargs,
        }

        if self._manifold_result is not None and self._last_compute_params == current_params:
            logger.info("Returning cached manifold result for identical parameters.")
            return self._manifold_result

        logger.info("New computation parameters detected or first run, computing manifold.")
        self._manifold_result = None
        self._successes = 0
        self._attempts = 0

        initial_state = self._generating_orbit._initial_state

        try:
            xx, tt, phi_T, PHI = _compute_stm(
                self._libration_point._var_eq_system,
                initial_state,
                self._generating_orbit.period,
                steps=2000,
                forward=self._forward,
                method=method,
                order=order,
            )
        except Exception as e:
            logger.error(f"Failed to propagate STM once: {e}")
            raise

        sn, un, _, Ws, Wu, _ = eigenvalue_decomposition(phi_T, discrete=1)

        snreal_vals, snreal_vecs = self._get_real_eigenvectors(Ws, sn)
        unreal_vals, unreal_vecs = self._get_real_eigenvectors(Wu, un)

        col_idx = NN - 1  # convert 1-based to 0-based
        if self._stable == 1:
            if snreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(
                    f"Requested stable eigenvector {NN} not available. "
                    f"Only {snreal_vecs.shape[1]} real stable eigenvectors found."
                )
            eigvec = snreal_vecs[:, col_idx]
            eigval = np.real(snreal_vals[col_idx])
            logger.debug(
                f"Using stable manifold direction with eigenvalue {eigval:.6f} for {NN}th eigenvector (cached)"
            )
        else:
            if unreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(
                    f"Requested unstable eigenvector {NN} not available. "
                    f"Only {unreal_vecs.shape[1]} real unstable eigenvectors found."
                )
            eigvec = unreal_vecs[:, col_idx]
            eigval = np.real(unreal_vals[col_idx])
            logger.debug(
                f"Using unstable manifold direction with eigenvalue {eigval:.6f} for {NN}th eigenvector (cached)"
            )

        ysos, dysos, states_list, times_list = [], [], [], []

        fractions = np.arange(0.0, 1.0, step)

        iterator = (
            tqdm(fractions, desc="Computing manifold")
            if kwargs["show_progress"]
            else fractions
        )

        for fraction in iterator:
            self._attempts += 1

            try:

                x0W = self._compute_manifold_section(
                    period=self._generating_orbit.period,
                    fraction=fraction,
                    displacement=displacement,
                    xx=xx,
                    tt=tt,
                    PHI=PHI,
                    eigvec=eigvec,
                ).astype(np.float64)
                tf = integration_fraction * 2 * np.pi
                dt = abs(kwargs["dt"])
                steps = max(int(abs(tf) / dt) + 1, 100)

                sol = _propagate_dynsys(
                    dynsys=self._generating_orbit.system._dynsys,
                    state0=x0W,
                    t0=0.0,
                    tf=tf,
                    forward=self._forward,
                    steps=steps,
                    method=method,
                    order=order,
                    flip_indices=slice(0, 6),
                )
                states, times = sol.states, sol.times

                x = states[:, 0]
                y = states[:, 1]
                z = states[:, 2]

                r1 = np.sqrt((x + self._mu) ** 2 + y ** 2 + z ** 2)
                r2 = np.sqrt((x - 1 + self._mu) ** 2 + y ** 2 + z ** 2)

                if (r1.min() < safe_r1) or (r2.min() < safe_r2):
                    logger.debug(
                        f"Fraction {fraction:.3f}: Trajectory discarded due to body-radius proximity (min(r1)={r1.min():.2e}, min(r2)={r2.min():.2e})"
                    )
                    continue

                max_energy_err = _max_rel_energy_error(states, self._mu)

                if max_energy_err > kwargs["energy_tol"]:
                    logger.warning(
                        f"Fraction {fraction:.3f}: Trajectory discarded due to energy drift (|C(t)|/|C(0)|={max_energy_err:.2e} > {kwargs['energy_tol']:.1e})"
                    )
                    continue

                states_list.append(states)
                times_list.append(times)

                # Section-of-section hits (Poincaré map points) are no longer
                # extracted here. This logic will be handled by the poincare module.

            except Exception as e:
                err = f"Error computing manifold: {e}"
                logger.error(err)
                continue
        
        # Note: success/failure of section hits is no longer tracked here.
        
        self._manifold_result = ManifoldResult(
            ysos, dysos, states_list, times_list, self._successes, self._attempts
        )
        self._last_compute_params = current_params
        return self._manifold_result

    def plot(self, dark_mode: bool = True, save: bool = False, filepath: str = 'manifold.svg', **kwargs):
        r"""
        Render a 3-D plot of the computed manifold.

        Parameters
        ----------
        dark_mode : bool, default True
            Apply a dark colour scheme.

        Raises
        ------
        ValueError
            If :pyattr:`manifold_result` is *None*.
        """
        if self._manifold_result is None:
            err = "Manifold result not computed. Please compute the manifold first."
            logger.error(err)
            raise ValueError(err)

        return plot_manifold(
            states_list=self._manifold_result.states_list,
            times_list=self._manifold_result.times_list,
            bodies=[self._generating_orbit._system.primary, self._generating_orbit._system.secondary],
            system_distance=self._generating_orbit._system.distance,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs
        )

    def to_csv(self, filepath: str, **kwargs):
        r"""
        Export manifold trajectory data to a CSV file.

        Each row in the CSV file represents a point in a trajectory,
        and includes a trajectory ID, timestamp, and the 6D state vector
        (x, y, z, vx, vy, vz).

        Parameters
        ----------
        filepath : str
            Path to the output CSV file. Parent directories are created if
            they do not exist.
        **kwargs
            Reserved for future use.

        Raises
        ------
        ValueError
            If :pyattr:`manifold_result` is `None`.
        """
        if self._manifold_result is None:
            err = "Manifold result not computed. Please compute the manifold first."
            logger.error(err)
            raise ValueError(err)

        data = []
        for i, (states, times) in enumerate(zip(self._manifold_result.states_list, self._manifold_result.times_list)):
            for j in range(states.shape[0]):
                data.append(
                    [i, times[j], states[j, 0], states[j, 1], states[j, 2], states[j, 3], states[j, 4], states[j, 5]]
                )
        
        df = pd.DataFrame(data, columns=['trajectory_id', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

        df.to_csv(filepath, index=False)
        logger.info(f"Manifold data successfully exported to {filepath}")

    def save(self, filepath: str, **kwargs) -> None:
        save_manifold(self, filepath, **kwargs)
        return

    @classmethod
    def load(cls, filepath: str) -> "Manifold":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manifold file not found: {filepath}")
        return load_manifold(filepath)
