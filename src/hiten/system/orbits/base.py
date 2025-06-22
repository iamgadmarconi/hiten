r"""
hiten.system.orbits.base
===================

Abstract definitions and convenience utilities for periodic orbit computation
in the circular restricted three-body problem (CR3BP).

The module provides:

* :pyclass:`PeriodicOrbit` - an abstract base class that implements common
  functionality such as energy evaluation, propagation wrappers, plotting and
  differential correction.
* :pyclass:`GenericOrbit` - a minimal concrete implementation useful for
  arbitrary initial conditions when no analytical guess or specific correction
  is required.
* Light-weight configuration containers (:pyclass:`orbitConfig`,
  :pyclass:`correctionConfig`) that encapsulate user input for families,
  libration points and differential correction settings.

References
----------
Szebehely, V. (1967). "Theory of Orbits - The Restricted Problem of Three
Bodies".
"""

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (Any, Callable, Dict, Literal, NamedTuple, Optional,
                    Sequence, Tuple)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from hiten.algorithms.dynamics.rtbp import (_propagate_dynsys, compute_stm,
                                      rtbp_dynsys, stability_indices)
from hiten.algorithms.dynamics.utils.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.dynamics.utils.geometry import _find_y_zero_crossing
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.utils.coordinates import rotating_to_inertial
from hiten.utils.files import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import (_plot_body, _set_axes_equal, _set_dark_mode,
                         animate_trajectories)


@dataclass
class orbitConfig:
    r"""
    Configuration for an orbit family around a specific libration point.

    Parameters
    ----------
    orbit_family : str
        Identifier of the orbit family, e.g. ``"halo"`` or ``"lyapunov"``.
    libration_point : LibrationPoint
        The libration point instance that anchors the family.
    extra_params : dict, optional
        Additional keyword parameters that specialised subclasses may
        require (left untouched by the base implementation).

    Attributes
    ----------
    orbit_family : str
        Normalised to lowercase in :pyfunc:`orbitConfig.__post_init__`.
    libration_point : LibrationPoint
        Same as *Parameters*.
    extra_params : dict
        Same as *Parameters*.
    """
    orbit_family: str
    libration_point: LibrationPoint
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that distance is positive.
        self.orbit_family = self.orbit_family.lower() # Normalize to lowercase


class S(IntEnum): X=0; Y=1; Z=2; VX=3; VY=4; VZ=5


class correctionConfig(NamedTuple):
    r"""
    Settings that drive the differential correction routine.

    The named-tuple is immutable and therefore safe to share across calls.

    Parameters
    ----------
    residual_indices : tuple of int
        Indices of the state vector used to build the residual vector
        :math:`\mathbf R`.
    control_indices : tuple of int
        Indices of the state vector that are allowed to change so as to cancel
        :math:`\mathbf R`.
    extra_jacobian : callable or None, optional
        Function returning an additional contribution that is subtracted from
        the Jacobian before solving the linear system; useful when the event
        definition introduces extra dependencies.
    target : tuple of float, default ``(0.0,)``
        Desired values for the residual components.
    event_func : callable, default :pyfunc:`hiten.utils.geometry._find_y_zero_crossing`
        Event used to terminate half-period propagation.
    method : {"rk", "scipy", "symplectic", "adaptive"}, default "scipy"
        Integrator back-end to use when marching the variational equations.
    order : int, default 8
        Order for the custom integrators.
    steps : int, default 2000
        Number of fixed steps per half-period when *method* is not adaptive.
    """
    residual_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    extra_jacobian: Callable[[np.ndarray,np.ndarray], np.ndarray] | None = None
    target: tuple[float, ...] = (0.0,)
    event_func: Callable[...,tuple[float,np.ndarray]] = _find_y_zero_crossing

    method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy"
    order: int = 8
    steps: int = 2000

class PeriodicOrbit(ABC):
    r"""
    Abstract base-class that encapsulates a CR3BP periodic orbit.

    The constructor either accepts a user supplied initial state or derives an
    analytical first guess via :pyfunc:`PeriodicOrbit._initial_guess` (to be
    implemented by subclasses). All subsequent high-level operations
    (propagation, plotting, stability analysis, differential correction) build
    upon this initial description.

    Parameters
    ----------
    config : orbitConfig
        Orbit family and libration point configuration.
    initial_state : Sequence[float] or None, optional
        Initial condition in rotating canonical units
        :math:`[x, y, z, \dot x, \dot y, \dot z]`. When *None* an analytical
        approximation is attempted.

    Attributes
    ----------
    family : str
        Orbit family name inherited from *config*.
    libration_point : LibrationPoint
        Libration point anchoring the family.
    system : System
        Parent CR3BP hiten.system.
    mu : float
        Mass ratio of the system, accessed as :pyattr:`System.mu`.
    initial_state : ndarray, shape (6,)
        Current initial condition.
    period : float or None
        Orbit period, set after a successful correction.
    trajectory : ndarray or None, shape (N, 6)
        Stored trajectory after :pyfunc:`PeriodicOrbit.propagate`.
    times : ndarray or None, shape (N,)
        Time vector associated with *trajectory*.
    stability_info : tuple or None
        Output of :pyfunc:`hiten.algorithms.dynamics.rtbp.stability_indices`.

    Notes
    -----
    Instantiating the class does **not** perform any propagation. Users must
    call :pyfunc:`PeriodicOrbit.differential_correction` (or manually set
    :pyattr:`period`) followed by :pyfunc:`PeriodicOrbit.propagate`.
    """

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self.family = config.orbit_family
        self.libration_point = config.libration_point
        self._system = self.libration_point.system
        self.mu = self._system.mu

        # Determine how the initial state will be obtained and log accordingly
        if initial_state is not None:
            logger.info(
                "Using provided initial conditions for %s orbit around L%d: %s",
                self.family,
                self.libration_point.idx,
                np.array2string(np.asarray(initial_state, dtype=np.float64), precision=12, suppress_small=True),
            )
            self._initial_state = np.asarray(initial_state, dtype=np.float64)
        else:
            logger.info(
                "No initial conditions provided; computing analytical approximation for %s orbit around L%d.",
                self.family,
                self.libration_point.idx,
            )
            self._initial_state = self._initial_guess()

        self.period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
        # General initialization log
        logger.info(f"Initialized {self.family} orbit around L{self.libration_point.idx}")

    def __str__(self):
        return f"{self.family} orbit around {self.libration_point}."

    def __repr__(self):
        return f"{self.__class__.__name__}(family={self.family}, libration_point={self.libration_point})"

    @property
    def initial_state(self) -> npt.NDArray[np.float64]:
        r"""
        Get the initial state vector of the orbit.
        
        Returns
        -------
        numpy.ndarray
            The initial state vector [x, y, z, vx, vy, vz]
        """
        return self._initial_state
    
    @property
    def trajectory(self) -> Optional[npt.NDArray[np.float64]]:
        r"""
        Get the computed trajectory points.
        
        Returns
        -------
        numpy.ndarray or None
            Array of shape (steps, 6) containing state vectors at each time step,
            or None if the trajectory hasn't been computed yet.
        """
        if self._trajectory is None:
            logger.warning("Trajectory not computed. Call propagate() first.")
        return self._trajectory
    
    @property
    def times(self) -> Optional[npt.NDArray[np.float64]]:
        r"""
        Get the time points corresponding to the trajectory.
        
        Returns
        -------
        numpy.ndarray or None
            Array of time points, or None if the trajectory hasn't been computed yet.
        """
        if self._times is None:
            logger.warning("Time points not computed. Call propagate() first.")
        return self._times
    
    @property
    def stability_info(self) -> Optional[Tuple]:
        r"""
        Get the stability information for the orbit.
        
        Returns
        -------
        tuple or None
            Tuple containing (stability_indices, eigenvalues, eigenvectors),
            or None if stability hasn't been computed yet.
        """
        if self._stability_info is None:
            logger.warning("Stability information not computed. Call compute_stability() first.")
        return self._stability_info

    @property
    def system(self) -> System:
        return self._system

    def _reset(self) -> None:
        r"""
        Reset all computed properties when the initial state is changed.
        Called internally after differential correction or any other operation
        that modifies the initial state.
        """
        self._trajectory = None
        self._times = None
        self._stability_info = None
        self.period = None
        logger.debug("Reset computed orbit properties due to state change")

    @property
    def is_stable(self) -> bool:
        r"""
        Check if the orbit is linearly stable.
        
        Returns
        -------
        bool
            True if all stability indices have magnitude <= 1, False otherwise
        """
        if self._stability_info is None:
            logger.info("Computing stability for stability check")
            self.compute_stability()
        
        indices = self._stability_info[0]  # nu values from stability_indices
        
        # An orbit is stable if all stability indices have magnitude <= 1
        return np.all(np.abs(indices) <= 1.0)

    @property
    def energy(self) -> float:
        r"""
        Compute the energy of the orbit at the initial state.
        
        Returns
        -------
        float
            The energy value
        """
        energy_val = crtbp_energy(self._initial_state, self.mu)
        logger.debug(f"Computed orbit energy: {energy_val}")
        return energy_val
    
    @property
    def jacobi_constant(self) -> float:
        r"""
        Compute the Jacobi constant of the orbit.
        
        Returns
        -------
        float
            The Jacobi constant value
        """
        return energy_to_jacobi(self.energy)

    def _cr3bp_system(self):
        r"""
        Create (or reuse) a _DynamicalSystem wrapper for the CR3BP.
        """
        if not hasattr(self, "_cached_dynsys"):
            self._cached_dynsys = rtbp_dynsys(mu=self.mu, name=str(self))
        return self._cached_dynsys

    def propagate(self, steps: int = 1000, method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order: int = 8) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Propagate the orbit for one period.
        
        Parameters
        ----------
        steps : int, optional
            Number of time steps. Default is 1000.
        method : str, optional
            Integration method. Default is "rk".
        **options
            Additional keyword arguments for the integration method
            
        Returns
        -------
        tuple
            (t, trajectory) containing the time and state arrays
        """
        if self.period is None:
            raise ValueError("Period must be set before propagation")
        
        sol = _propagate_dynsys(
            dynsys=self.system._dynsys,
            state0=self.initial_state,
            t0=0.0,
            tf=self.period,
            forward=1,
            steps=steps,
            method=method,
            order=order,
        )

        self._trajectory = sol.states
        self._times = sol.times

        return self._times, self._trajectory

    def compute_stability(self, **kwargs) -> Tuple:
        r"""
        Compute stability information for the orbit.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the STM computation
            
        Returns
        -------
        tuple
            (stability_indices, eigenvalues, eigenvectors) from the monodromy matrix
        """
        if self.period is None:
            msg = "Period must be set before stability analysis"
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"Computing stability for orbit with period {self.period}")
        # Compute STM over one period
        _, _, monodromy, _ = compute_stm(self.libration_point._var_eq_system, self.initial_state, self.period)
        
        # Analyze stability
        stability = stability_indices(monodromy)
        self._stability_info = stability
        
        is_stable = np.all(np.abs(stability[0]) <= 1.0)
        logger.info(f"Orbit stability: {'stable' if is_stable else 'unstable'}")
        
        return stability

    def plot(self, frame="rotating", show=True, figsize=(10, 8), dark_mode=True, **kwargs):
        r"""
        Plot the orbit trajectory in the specified reference frame.
        
        Parameters
        ----------
        frame : str, optional
            Reference frame to use for plotting. Options are "rotating" or "inertial".
            Default is "rotating".
        show : bool, optional
            Whether to call plt.show() after creating the plot. Default is True.
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (10, 8).
        **kwargs
            Additional keyword arguments passed to the specific plotting function.
            
        Returns
        -------
        tuple
            (fig, ax) containing the figure and axis objects for further customization
            
        Notes
        -----
        This is a convenience method that calls either plot_rotating_frame or
        plot_inertial_frame based on the 'frame' parameter.
        """
        if self._trajectory is None:
            msg = "No trajectory to plot. Call propagate() first."
            logger.error(msg)
            raise RuntimeError(msg)
            
        if frame.lower() == "rotating":
            return self.plot_rotating_frame(show=show, figsize=figsize, dark_mode=dark_mode, **kwargs)
        elif frame.lower() == "inertial":
            return self.plot_inertial_frame(show=show, figsize=figsize, dark_mode=dark_mode, **kwargs)
        else:
            msg = f"Invalid frame '{frame}'. Must be 'rotating' or 'inertial'."
            logger.error(msg)
            raise ValueError(msg)
        
    def animate(self, **kwargs):
        if self._trajectory is None:
            logger.warning("No trajectory to animate. Call propagate() first.")
            return None, None
        
        return animate_trajectories(self._trajectory, self._times, [self._system.primary, self._system.secondary], self._system.distance, **kwargs)

    def plot_rotating_frame(self, show=True, figsize=(10, 8), dark_mode=True, **kwargs):
        r"""
        Plot the orbit trajectory in the rotating reference frame.
        
        Parameters
        ----------
        show : bool, optional
            Whether to call plt.show() after creating the plot. Default is True.
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (10, 8).
        **kwargs
            Additional keyword arguments for plot customization.
            
        Returns
        -------
        tuple
            (fig, ax) containing the figure and axis objects for further customization
        """
        if self._trajectory is None:
            logger.warning("No trajectory to plot. Call propagate() first.")
            return None, None
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get trajectory data
        x = self._trajectory[:, 0]
        y = self._trajectory[:, 1]
        z = self._trajectory[:, 2]
        
        # Plot orbit trajectory
        orbit_color = kwargs.get('orbit_color', 'cyan')
        ax.plot(x, y, z, label=f'{self.family.capitalize()} Orbit', color=orbit_color)
        
        # Plot primary body (canonical position: -mu, 0, 0)
        primary_pos = np.array([-self.mu, 0, 0])
        primary_radius = self._system.primary.radius / self._system.distance  # Convert to canonical units
        _plot_body(ax, primary_pos, primary_radius, self._system.primary.color, self._system.primary.name)
        
        # Plot secondary body (canonical position: 1-mu, 0, 0)
        secondary_pos = np.array([1-self.mu, 0, 0])
        secondary_radius = self._system.secondary.radius / self._system.distance  # Convert to canonical units
        _plot_body(ax, secondary_pos, secondary_radius, self._system.secondary.color, self._system.secondary.name)
        
        # Plot libration point
        ax.scatter(*self.libration_point.position, color='#FF00FF', marker='o', 
                s=5, label=f'{self.libration_point}')
        
        ax.set_xlabel('X [canonical]')
        ax.set_ylabel('Y [canonical]')
        ax.set_zlabel('Z [canonical]')
        
        # Create legend and apply styling
        ax.legend()
        _set_axes_equal(ax)
        
        # Apply dark mode if requested
        if dark_mode:
            _set_dark_mode(fig, ax, title=f'{self.family.capitalize()} Orbit in Rotating Frame')
        else:
            ax.set_title(f'{self.family.capitalize()} Orbit in Rotating Frame')
        
        if show:
            plt.show()
            
        return fig, ax

        
    def plot_inertial_frame(self, show=True, figsize=(10, 8), dark_mode=True, **kwargs):
        r"""
        Plot the orbit trajectory in the primary-centered inertial reference frame.
        
        Parameters
        ----------
        show : bool, optional
            Whether to call plt.show() after creating the plot. Default is True.
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (10, 8).
        **kwargs
            Additional keyword arguments for plot customization.
            
        Returns
        -------
        tuple
            (fig, ax) containing the figure and axis objects for further customization
        """
        if self._trajectory is None or self._times is None:
            logger.warning("No trajectory to plot. Call propagate() first.")
            return None, None
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get trajectory data and convert to inertial frame
        traj_inertial = []
        
        for state, t in zip(self._trajectory, self._times):
            # Convert rotating frame to inertial frame (canonical units)
            inertial_state = rotating_to_inertial(state, t, self.mu)
            traj_inertial.append(inertial_state)
        
        traj_inertial = np.array(traj_inertial)
        x, y, z = traj_inertial[:, 0], traj_inertial[:, 1], traj_inertial[:, 2]
        
        # Plot orbit trajectory
        orbit_color = kwargs.get('orbit_color', 'red')
        ax.plot(x, y, z, label=f'{self.family.capitalize()} Orbit', color=orbit_color)
        
        # Plot primary body at origin
        primary_pos = np.array([0, 0, 0])
        primary_radius = self._system.primary.radius / self._system.distance  # Convert to canonical units
        _plot_body(ax, primary_pos, primary_radius, self._system.primary.color, self._system.primary.name)
        
        # Plot secondary's orbit and position
        theta = self._times  # Time is angle in canonical units
        secondary_x = (1-self.mu) * np.cos(theta)
        secondary_y = (1-self.mu) * np.sin(theta)
        secondary_z = np.zeros_like(theta)
        
        # Plot secondary's orbit
        ax.plot(secondary_x, secondary_y, secondary_z, '--', color=self._system.secondary.color, 
                alpha=0.5, label=f'{self._system.secondary.name} Orbit')
        
        # Plot secondary at final position
        secondary_pos = np.array([secondary_x[-1], secondary_y[-1], secondary_z[-1]])
        secondary_radius = self._system.secondary.radius / self._system.distance  # Convert to canonical units
        _plot_body(ax, secondary_pos, secondary_radius, self._system.secondary.color, self._system.secondary.name)
        
        ax.set_xlabel('X [canonical]')
        ax.set_ylabel('Y [canonical]')
        ax.set_zlabel('Z [canonical]')
        
        # Create legend and apply styling
        ax.legend()
        _set_axes_equal(ax)
        
        # Apply dark mode if requested
        if dark_mode:
            _set_dark_mode(fig, ax, title=f'{self.family.capitalize()} Orbit in Inertial Frame')
        else:
            ax.set_title(f'{self.family.capitalize()} Orbit in Inertial Frame')
        
        if show:
            plt.show()
            
        return fig, ax

    def save(self, filepath: str, **kwargs) -> None:
        r"""
        Save the orbit data to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the orbit data
        **kwargs
            Additional options for saving
            
        Notes
        -----
        This saves the essential orbit information including initial state, 
        period, and trajectory (if computed).
        """
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

        # Create data dictionary with all essential information
        data = {
            'orbit_type': self.__class__.__name__,
            'family': self.family,
            'mu': self.mu,
            'initial_state': self._initial_state.tolist() if self._initial_state is not None else None,
            'period': self.period,
        }
        
        # Add trajectory data if available
        if self._trajectory is not None:
            data['trajectory'] = self._trajectory.tolist()
            data['times'] = self._times.tolist()
        
        # Add stability information if available
        if self._stability_info is not None:
            # Convert numpy arrays to lists for serialization
            stability_data = []
            for item in self._stability_info:
                if isinstance(item, np.ndarray):
                    stability_data.append(item.tolist())
                else:
                    stability_data.append(item)
            data['stability_info'] = stability_data
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save data
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Orbit saved to {filepath}")
    
    def load(self, filepath: str, **kwargs) -> None:
        r"""
        Load orbit data from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved orbit data
        **kwargs
            Additional options for loading
            
        Returns
        -------
        None
            Updates the current instance with loaded data
            
        Raises
        ------
        FileNotFoundError
            If the specified file doesn't exist
        ValueError
            If the file contains incompatible data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")
        
        # Load data
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Verify orbit type
        if data['orbit_type'] != self.__class__.__name__:
            logger.warning(f"Loading {data['orbit_type']} data into {self.__class__.__name__} instance")
            
        # Update orbit properties
        self.mu = data['mu']
        self.family = data['family']
        
        if data['initial_state'] is not None:
            self._initial_state = np.array(data['initial_state'])
        
        self.period = data['period']
        
        # Load trajectory if available
        if 'trajectory' in data:
            self._trajectory = np.array(data['trajectory'])
            self._times = np.array(data['times'])
        
        # Load stability information if available
        if 'stability_info' in data:
            # Convert lists back to numpy arrays
            stability_data = []
            for item in data['stability_info']:
                if isinstance(item, list):
                    stability_data.append(np.array(item))
                else:
                    stability_data.append(item)
            self._stability_info = tuple(stability_data)
            
        logger.info(f"Orbit loaded from {filepath}")

    @property
    @abstractmethod
    def eccentricity(self):
        pass

    @abstractmethod
    def _initial_guess(self, **kwargs):
        pass

    def differential_correction(
            self,
            cfg: correctionConfig,
            *,
            tol: float = 1e-10,
            max_attempts: int = 25,
            forward: int = 1
        ) -> tuple[np.ndarray, float]:
        """
        Perform differential correction to find a periodic orbit.
        
        Parameters
        ----------
        cfg : correctionConfig
            Configuration for the differential correction.
        tol : float, optional
            Tolerance for the correction.
        max_attempts : int, optional
            Maximum number of attempts to find the orbit.
        forward : int, optional
            Direction of propagation.

        Returns
        -------
        tuple
            (state, period)

        Raises
        ------
        RuntimeError
            If the orbit is not found.
        """
        X0 = self.initial_state.copy()
        for k in range(max_attempts + 1):

            t_ev, X_ev = cfg.event_func(dynsys=self.system._dynsys, x0=X0, forward=forward)

            R = X_ev[list(cfg.residual_indices)] - np.array(cfg.target)

            if np.linalg.norm(R, ord=np.inf) < tol:
                self._reset(); self._initial_state = X0
                self.period = 2 * t_ev
                return X0, t_ev

            _, _, Phi, _ = compute_stm(self.libration_point._var_eq_system, X0, t_ev, steps=cfg.steps, method=cfg.method, order=cfg.order)

            J = Phi[np.ix_(cfg.residual_indices, cfg.control_indices)]

            if cfg.extra_jacobian is not None:
                J -= cfg.extra_jacobian(X_ev, Phi)

            if np.linalg.det(J) < 1e-12:
                logger.warning(f"Jacobian determinant is small ({np.linalg.det(J):.2e}), adding regularization.")
                J += np.eye(J.shape[0]) * 1e-12

            delta = np.linalg.solve(J, -R)
            logger.info(f"Differential correction delta: {delta} at iteration {k}")
            X0[list(cfg.control_indices)] += delta

        raise RuntimeError("did not converge")


class GenericOrbit(PeriodicOrbit):
    r"""
    A minimal concrete orbit class for arbitrary initial conditions, with no correction or special guess logic.
    """
    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        super().__init__(config, initial_state)
        if self.period is None:
            self.period = np.pi

    @property
    def eccentricity(self):
        return np.nan

    def _initial_guess(self, **kwargs):
        if hasattr(self, '_initial_state') and self._initial_state is not None:
            return self._initial_state
        raise ValueError("No initial state provided for GenericOrbit.")
