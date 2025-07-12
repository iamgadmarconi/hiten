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
from hiten.algorithms.dynamics.utils.geometry import surface_of_section
from hiten.algorithms.dynamics.utils.linalg import (_totime,
                                                    eigenvalue_decomposition)
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io import _ensure_dir, _load_manifold, _save_manifold
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
            method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", 
            order: int = 6
        ):
        self._generating_orbit = generating_orbit
        self._libration_point = self._generating_orbit.libration_point
        self._stable = 1 if stable else -1
        self._direction = 1 if direction == "positive" else -1
        self._mu = self._generating_orbit.system.mu
        self._method = method
        self._order = order

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
    def method(self) -> str:
        """Backend integrator used for propagation."""
        return self._method

    @property
    def order(self) -> int:
        """Integration order for fixed-step Runge-Kutta methods."""
        return self._order

    @property
    def manifold_result(self) -> ManifoldResult:
        """Cached result from the last successful compute call."""
        return self._manifold_result

    def __str__(self):
        return f"Manifold(stable={self._stable}, direction={self._direction}) of {self._libration_point}-{self._generating_orbit}"
    
    def __repr__(self):
        return self.__str__()

    def _get_real_eigenvectors(self, vectors: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter for real eigenvalues and their corresponding eigenvectors."""
        real_vals = []
        real_vecs_list = []
        for k in range(len(values)):
            if np.isreal(values[k]):
                real_vals.append(values[k])
                real_vecs_list.append(vectors[:, k])
        
        real_vals_arr = np.array(real_vals, dtype=np.complex128)
        if real_vecs_list:
            real_vecs_arr = np.column_stack(real_vecs_list)
        else:
            real_vecs_arr = np.zeros((vectors.shape[0], 0), dtype=np.complex128)
            
        return real_vals_arr, real_vecs_arr

    def _compute_manifold_section(self, state: np.ndarray, period: float, fraction: float, NN: int = 1, forward: int = 1, displacement: float = 1e-6):
        r"""
        Compute a section of the invariant manifold.

        Parameters
        ----------
        state : numpy.ndarray
            Initial state of the periodic orbit.
        period : float
            Period of the periodic orbit.
        fraction : float
            Fraction of the period to compute the section at.
        NN : int, default 1
            Index of the eigenvector to compute the section for.
        forward : int, default 1
            Direction of integration.
        displacement : float, default 1e-6
            Displacement applied along the eigenvector.

        Returns
        -------
        numpy.ndarray
            Initial condition for the manifold section.

        Raises
        ------
        ValueError
            If the requested eigenvector is not available.
        """
        xx, tt, phi_T, PHI = _compute_stm(self._libration_point._var_eq_system, state, period, steps=2000, forward=forward, method=self._method, order=self._order)

        sn, un, _, Ws, Wu, _ = eigenvalue_decomposition(phi_T, discrete=1)

        snreal_vals, snreal_vecs = self._get_real_eigenvectors(Ws, sn)
        unreal_vals, unreal_vecs = self._get_real_eigenvectors(Wu, un)

        col_idx = NN - 1

        if self._stable == 1:
            if snreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(f"Requested stable eigenvector {NN} not available. Only {snreal_vecs.shape[1]} real stable eigenvectors found.")
            eigval = np.real(snreal_vals[col_idx])
            eigvec = snreal_vecs[:, col_idx]
            logger.debug(f"Using stable manifold direction with eigenvalue {eigval:.6f} for {NN}th eigenvector")

        else:  # unstable
            if unreal_vecs.shape[1] <= col_idx or col_idx < 0:
                raise ValueError(f"Requested unstable eigenvector {NN} not available. Only {unreal_vecs.shape[1]} real unstable eigenvectors found.")
            eigval = np.real(unreal_vals[col_idx])
            eigvec = unreal_vecs[:, col_idx]
            logger.debug(f"Using unstable manifold direction with eigenvalue {eigval:.6f} for {NN}th eigenvector")

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

    def compute(self, step: float = 0.02, integration_fraction: float = 0.75, NN: int = 1, displacement: float = 1e-6, **kwargs):
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
        >>> from hiten.system import System, Manifold, HaloOrbit
        >>> system = System.from_bodies("earth", "moon")
        >>> L2 = system.get_libration_point(2)
        >>> halo_L2 = HaloOrbit(system, L2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        >>> man = Manifold(halo_L2)
        >>> result = man.compute(step=0.05)
        >>> print(f"Success rate: {result.success_rate:.0%}")
        """
        # Create a canonical representation of the parameters for comparison.
        kwargs.setdefault("show_progress", True)
        kwargs.setdefault("dt", 1e-3)
        current_params = {
            "step": step, "integration_fraction": integration_fraction, "NN": NN, "displacement": displacement, **kwargs
        }

        if self._manifold_result is not None and self._last_compute_params == current_params:
            logger.info("Returning cached manifold result for identical parameters.")
            return self._manifold_result

        logger.info("New computation parameters detected or first run, computing manifold.")
        # Invalidate cache and reset counters for a new computation
        self._manifold_result = None
        self._successes = 0
        self._attempts = 0


        initial_state = self._generating_orbit._initial_state

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
                    initial_state,
                    self._generating_orbit.period,
                    fraction,
                    NN=NN,
                    forward=self._forward,
                    displacement=displacement
                )
                x0W = x0W.flatten().astype(np.float64)
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
                    method=self._method, 
                    order=self._order,
                    flip_indices=slice(0, 6)
                )
                states, times = sol.states, sol.times

                states_list.append(states)
                times_list.append(times)

                Xy0, _ = surface_of_section(states, times, self._mu, M=2, C=0)

                if len(Xy0) > 0:
                    Xy0 = Xy0.flatten()
                    ysos.append(Xy0[1])
                    dysos.append(Xy0[4])
                    self._successes += 1
                    logger.debug(f"Fraction {fraction:.3f}: Found Poincaré section point at y={Xy0[1]:.6f}, vy={Xy0[4]:.6f}")

            except Exception as e:
                err = f"Error computing manifold: {e}"
                logger.error(err)
                continue
        
        if self._attempts > 0 and self._successes < self._attempts:
            failed_attempts = self._attempts - self._successes
            failure_rate = (failed_attempts / self._attempts) * 100
            logger.warning(f"Failed to find {failure_rate:.1f}% ({failed_attempts}/{self._attempts}) Poincaré section crossings")
            
        self._manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, self._successes, self._attempts)
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
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        _save_manifold(self, filepath)
        return

    @classmethod
    def load(cls, filepath: str) -> "Manifold":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manifold file not found: {filepath}")
        return _load_manifold(filepath)
