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
import pickle
from dataclasses import dataclass
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from hiten.algorithms.dynamics.rtbp import _propagate_dynsys, compute_stm
from hiten.algorithms.dynamics.utils.geometry import surface_of_section
from hiten.algorithms.dynamics.utils.linalg import (_totime,
                                                    eigenvalue_decomposition)
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.files import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import _plot_body, _set_axes_equal, _set_dark_mode


@dataclass
class manifoldConfig:
    r"""
    Configuration options for :pyclass:`Manifold`.

    Parameter
    ----------
    generating_orbit : :pyclass:`PeriodicOrbit`
        Periodic orbit that generates the manifold.
    stable : bool, default True
        ``True`` selects the stable manifold, ``False`` the unstable one.
    direction : {{'Positive', 'Negative'}}, default 'Positive'
        Sign of the eigenvector used to initialise the manifold branch.
    method : {{'rk', 'scipy', 'symplectic', 'adaptive'}}, default 'scipy'
        Backend integrator passed to :pyfunc:`_propagate_dynsys`.
    order : int, default 6
        Integration order for fixed-step Runge-Kutta methods.
    """
    generating_orbit: PeriodicOrbit
    stable: bool = True
    direction: Literal["Positive", "Negative"] = "Positive"

    method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy"
    order: int = 6


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
    config : manifoldConfig
        Run-time options.

    Attributes
    ----------
    generating_orbit : :pyclass:`PeriodicOrbit`
        Orbit that seeds the manifold.
    libration_point : :pyclass:`LibrationPoint`
        Libration point associated with *generating_orbit*.
    stable, direction : int
        Encoded form of the options in :pyclass:`ManifoldConfig`.
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

    def __init__(self, config: manifoldConfig):
        self.generating_orbit = config.generating_orbit
        self.libration_point = self.generating_orbit.libration_point
        self.stable = 1 if config.stable else -1
        self.direction = 1 if config.direction == "Positive" else -1
        self._forward = -self.stable
        self.mu = self.generating_orbit.system.mu
        self.method = config.method
        self.order = config.order
        self._successes = 0
        self._attempts = 0
        self.manifold_result: ManifoldResult = None

    def __str__(self):
        return f"Manifold(stable={self.stable}, direction={self.direction}) of {self.libration_point}-{self.generating_orbit}"
    
    def __repr__(self):
        return self.__str__()
    
    def compute(self, step: float = 0.02, integration_fraction: float = 0.75, **kwargs):
        r"""
        Generate manifold trajectories and build a Poincaré map.

        The routine samples the generating orbit at equally spaced fractions
        of its period, displaces each point :math:`10^{-6}` units along the
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
        >>> cfg = manifoldConfig(halo_L2)  # halo_L2 is a PeriodicOrbit
        >>> man = Manifold(cfg)
        >>> result = man.compute(step=0.05)
        >>> print(f"Success rate: {result.success_rate:.0%}")
        """

        if self.manifold_result is not None:
            return self.manifold_result

        kwargs.setdefault("show_progress", True)
        kwargs.setdefault("dt", 1e-3)

        initial_state = self.generating_orbit._initial_state

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
                    self.generating_orbit.period,
                    fraction,
                    forward=self._forward,
                )
                x0W = x0W.flatten().astype(np.float64)
                tf = integration_fraction * 2 * np.pi

                dt = abs(kwargs["dt"])

                steps = max(int(abs(tf) / dt) + 1, 100)

                sol = _propagate_dynsys(
                    dynsys=self.generating_orbit.system._dynsys,
                    state0=x0W, 
                    t0=0.0, 
                    tf=tf,
                    forward=self._forward,
                    steps=steps,
                    method=self.method, 
                    order=self.order,
                    flip_indices=slice(0, 6)
                )
                states, times = sol.states, sol.times

                states_list.append(states)
                times_list.append(times)

                Xy0, _ = surface_of_section(states, times, self.mu, M=2, C=0)

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
            
        self.manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, self._successes, self._attempts)
        return self.manifold_result

    def _compute_manifold_section(self, state: np.ndarray, period: float, fraction: float, NN: int = 1, forward: int = 1):
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

        Returns
        -------
        numpy.ndarray
            Initial condition for the manifold section.

        Raises
        ------
        ValueError
            If the requested eigenvector is not available.
        """
        xx, tt, phi_T, PHI = compute_stm(self.libration_point._var_eq_system, state, period, steps=2000, forward=forward, method=self.method, order=self.order)

        sn, un, _, Ws, Wu, _ = eigenvalue_decomposition(phi_T, discrete=1)

        snreal_vals = []
        snreal_vecs = []
        for k in range(len(sn)):
            if np.isreal(sn[k]):
                snreal_vals.append(sn[k])
                snreal_vecs.append(Ws[:, k])

        unreal_vals = []
        unreal_vecs = []
        for k in range(len(un)):
            if np.isreal(un[k]):
                unreal_vals.append(un[k])
                unreal_vecs.append(Wu[:, k])

        snreal_vals = np.array(snreal_vals, dtype=np.complex128)
        unreal_vals = np.array(unreal_vals, dtype=np.complex128)
        snreal_vecs = (np.column_stack(snreal_vecs) 
                    if len(snreal_vecs) else np.zeros((6, 0), dtype=np.complex128))
        unreal_vecs = (np.column_stack(unreal_vecs) 
                    if len(unreal_vecs) else np.zeros((6, 0), dtype=np.complex128))

        col_idx = NN - 1

        if self.stable == 1 and (snreal_vecs.shape[1] <= col_idx or col_idx < 0):
            raise ValueError(f"Requested stable eigenvector {NN} not available. Only {snreal_vecs.shape[1]} real stable eigenvectors found.")
        
        if self.stable == -1 and (unreal_vecs.shape[1] <= col_idx or col_idx < 0):
            raise ValueError(f"Requested unstable eigenvector {NN} not available. Only {unreal_vecs.shape[1]} real unstable eigenvectors found.")

        WS = snreal_vecs[:, col_idx] if self.stable == 1 else None
        WU = unreal_vecs[:, col_idx] if self.stable == -1 else None

        mfrac = _totime(tt, fraction * period)
        
        if np.isscalar(mfrac):
            mfrac_idx = mfrac
        else:
            mfrac_idx = mfrac[0]

        phi_frac_flat = PHI[mfrac_idx, :36]
        phi_frac = phi_frac_flat.reshape((6, 6))

        if self.stable == 1:
            MAN = self.direction * (phi_frac @ WS)
            logger.debug(f"Using stable manifold direction with eigenvalue {np.real(snreal_vals[col_idx]):.6f} for {NN}th eigenvector")

        elif self.stable == -1:
            MAN = self.direction * (phi_frac @ WU)
            logger.debug(f"Using unstable manifold direction with eigenvalue {np.real(unreal_vals[col_idx]):.6f} for {NN}th eigenvector")

        disp_magnitude = np.linalg.norm(MAN[0:3])

        if disp_magnitude < 1e-14:
            logger.warning(f"Very small displacement magnitude: {disp_magnitude:.2e}, setting to 1.0")
            disp_magnitude = 1.0
        d = 1e-6 / disp_magnitude

        fracH = xx[mfrac_idx, :].copy()

        x0W = fracH + d * MAN.real
        x0W = x0W.flatten()
        
        if abs(x0W[2]) < 1.0e-15:
            x0W[2] = 0.0
        if abs(x0W[5]) < 1.0e-15:
            x0W[5] = 0.0

        return x0W
    
    def plot(self, dark_mode: bool = True):
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
        if self.manifold_result is None:
            err = "Manifold result not computed. Please compute the manifold first."
            logger.error(err)
            raise ValueError(err)


        states_list, times_list = self.manifold_result.states_list, self.manifold_result.times_list

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        num_traj = len(states_list)
        cmap = plt.get_cmap('plasma')
        for i, (xW, _) in enumerate(zip(states_list, times_list)):
            color = cmap(i / (num_traj - 1)) if num_traj > 1 else cmap(0.5)
            ax.plot(xW[:, 0], xW[:, 1], xW[:, 2], color=color, lw=2)

        mu = self.mu

        primary_center = np.array([-mu, 0, 0])
        primary_radius = self.generating_orbit._system.primary.radius
        _plot_body(ax, primary_center, primary_radius / self.generating_orbit._system.distance, 'blue', self.generating_orbit._system.primary.name)

        secondary_center = np.array([(1 - mu), 0, 0])
        secondary_radius = self.generating_orbit._system.secondary.radius
        _plot_body(ax, secondary_center, secondary_radius / self.generating_orbit._system.distance, 'grey', self.generating_orbit._system.secondary.name)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        _set_axes_equal(ax)
        ax.set_title('Manifold')
        if dark_mode:
            _set_dark_mode(fig, ax, title=ax.get_title())
        plt.show()

    def save(self, filepath: str, **kwargs) -> None:
        r"""
        Serialise the manifold to *filepath*.

        Parameters
        ----------
        filepath : str
            Destination file. Parent directories are created automatically.
        **kwargs
            Reserved for future use.

        Raises
        ------
        OSError
            If the file cannot be written.
        """

        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

        data = {
            "manifold_type": self.__class__.__name__,
            "stable": bool(self.stable == 1),
            "direction": "Positive" if self.direction == 1 else "Negative",
            "method": self.method,
            "order": self.order,
        }

        try:
            data["generating_orbit"] = {
                "family": getattr(self.generating_orbit, "orbit_family", self.generating_orbit.__class__.__name__),
                "period": getattr(self.generating_orbit, "period", None),
                "initial_state": self.generating_orbit._initial_state.tolist(),
            }
        except Exception:
            pass

        if self.manifold_result is not None:
            mr = self.manifold_result
            data["manifold_result"] = {
                "ysos": mr.ysos,
                "dysos": mr.dysos,
                "states_list": [s.tolist() for s in mr.states_list],
                "times_list": [t.tolist() for t in mr.times_list],
                "_successes": mr._successes,
                "_attempts": mr._attempts,
            }
        else:
            data["manifold_result"] = None

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)

        logger.info("Manifold saved to %s", filepath)

    def load(self, filepath: str, **kwargs) -> None:
        r"""
        Load a manifold previously stored with :pyfunc:`save`.

        Parameters
        ----------
        filepath : str
            File generated by :pyfunc:`save`.
        **kwargs
            Reserved for future use.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manifold file not found: {filepath}")

        with open(filepath, "rb") as fh:
            data = pickle.load(fh)

        if data.get("manifold_type") != self.__class__.__name__:
            logger.warning(
                "Loading %s data into %s instance", data.get("manifold_type", "<unknown>"), self.__class__.__name__
            )

        self.stable = 1 if data.get("stable", True) else -1
        self.direction = 1 if data.get("direction", "Positive") == "Positive" else -1

        self._loaded_generating_orbit_info = data.get("generating_orbit", {})

        mr_data = data.get("manifold_result")
        if mr_data is not None:
            ysos = mr_data["ysos"]
            dysos = mr_data["dysos"]
            states_list = [np.array(s, dtype=float) for s in mr_data["states_list"]]
            times_list = [np.array(t, dtype=float) for t in mr_data["times_list"]]
            _successes = mr_data.get("_successes", 0)
            _attempts = mr_data.get("_attempts", 0)
            self.manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, _successes, _attempts)
        else:
            self.manifold_result = None

        logger.info("Manifold loaded from %s", filepath)
