import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from algorithms.dynamics.rtbp import (compute_stm, create_rtbp_system,
                                      stability_indices)
from algorithms.energy import crtbp_energy, energy_to_jacobi
from algorithms.geometry import _find_y_zero_crossing
from algorithms.integrators.rk import RungeKutta
from algorithms.integrators.symplectic import TaoSymplectic
from plots.plots import _plot_body, _set_axes_equal, _set_dark_mode
from system import System
from utils.coordinates import rotating_to_inertial
from utils.log_config import logger


@dataclass
class orbitConfig:
    system: System
    orbit_family: str
    libration_point_idx: int
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that distance is positive.
        self.orbit_family = self.orbit_family.lower() # Normalize to lowercase

        if self.libration_point_idx not in [1, 2, 3, 4, 5]:
            raise ValueError(f"Libration point index must be 1, 2, 3, 4, or 5. Got {self.libration_point_idx}.")


class S(IntEnum): X=0; Y=1; Z=2; VX=3; VY=4; VZ=5


class correctionConfig(NamedTuple):
    residual_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    extra_jacobian: Callable[[np.ndarray,np.ndarray], np.ndarray] | None = None
    target: tuple[float, ...] = (0.0,)
    event_func: Callable[...,tuple[float,np.ndarray]] = _find_y_zero_crossing



class PeriodicOrbit(ABC):

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self._system = config.system
        self.mu = self._system.mu
        self.family = config.orbit_family
        self.libration_point = self._system.get_libration_point(config.libration_point_idx)

        # Determine how the initial state will be obtained and log accordingly
        if initial_state is not None:
            logger.info(
                "Using provided initial conditions for %s orbit around L%d: %s",
                self.family,
                config.libration_point_idx,
                np.array2string(np.asarray(initial_state, dtype=np.float64), precision=12, suppress_small=True),
            )
            self._initial_state = np.asarray(initial_state, dtype=np.float64)
        else:
            logger.info(
                "No initial conditions provided; computing analytical approximation for %s orbit around L%d.",
                self.family,
                config.libration_point_idx,
            )
            self._initial_state = self._initial_guess()

        self.period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
        # General initialization log
        logger.info(f"Initialized {self.family} orbit around L{config.libration_point_idx}")

    def __str__(self):
        return f"{self.family} orbit around {self.libration_point}."

    def __repr__(self):
        return f"{self.__class__.__name__}(family={self.family}, libration_point={self.libration_point})"

    @property
    def initial_state(self) -> npt.NDArray[np.float64]:
        """
        Get the initial state vector of the orbit.
        
        Returns
        -------
        numpy.ndarray
            The initial state vector [x, y, z, vx, vy, vz]
        """
        return self._initial_state
    
    @property
    def trajectory(self) -> Optional[npt.NDArray[np.float64]]:
        """
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
        """
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
        """
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
        """
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
        """
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
        """
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
        """
        Compute the Jacobi constant of the orbit.
        
        Returns
        -------
        float
            The Jacobi constant value
        """
        return energy_to_jacobi(self.energy)

    def _cr3bp_system(self):
        """Create (or reuse) a DynamicalSystem wrapper for the CR3BP."""
        if not hasattr(self, "_cached_dynsys"):
            self._cached_dynsys = create_rtbp_system(mu=self.mu, name=str(self))
        return self._cached_dynsys

    def propagate(
        self,
        steps: int = 1000,
        method: str = "rk8",
        **options,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Propagate the orbit for one period.
        
        Parameters
        ----------
        steps : int, optional
            Number of time steps. Default is 1000.
        method : str, optional
            Integration method. Default is "rk8".
        **options
            Additional keyword arguments for the integration method
            
        Returns
        -------
        tuple
            (t, trajectory) containing the time and state arrays
        """
        if self.period is None:
            raise ValueError("Period must be set before propagation")

        # Build time grid
        t_vals = np.linspace(0.0, self.period, steps, dtype=np.float64)

        method_lc = method.lower()
        if method_lc.startswith("rk"):
            order = int(method_lc[2:])
            integrator = RungeKutta(order=order, **options)
        elif method_lc.startswith("symp"):
            order = int(method_lc[4:])
            integrator = TaoSymplectic(order=order, **options)
        else:
            raise ValueError(f"Unknown integration method '{method}'.")

        dynsys = self._cr3bp_system()
        sol = integrator.integrate(dynsys, self.initial_state, t_vals)

        self._trajectory = sol.states
        self._times = sol.times

        logger.info(
            "Propagation complete using %s (order=%s). Trajectory shape: %s",
            integrator.name,
            integrator.order,
            self._trajectory.shape,
        )

        return self._times, self._trajectory

    def compute_stability(self, **kwargs) -> Tuple:
        """
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
        _, _, monodromy, _ = compute_stm(
            self.initial_state, self.mu, self.period, **kwargs
        )
        
        # Analyze stability
        stability = stability_indices(monodromy)
        self._stability_info = stability
        
        is_stable = np.all(np.abs(stability[0]) <= 1.0)
        logger.info(f"Orbit stability: {'stable' if is_stable else 'unstable'}")
        
        return stability

    def plot(self, frame="rotating", show=True, figsize=(10, 8), dark_mode=True, **kwargs):
        """
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

    def plot_rotating_frame(self, show=True, figsize=(10, 8), dark_mode=True, **kwargs):
        """
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
        """
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
        """
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
        """
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
        X0 = self.initial_state.copy()
        for k in range(max_attempts + 1):
            logger.debug(f"Differential correction iteration {k}")
            t_ev, X_ev = cfg.event_func(X0, self.mu, forward=forward)
            logger.debug(f"called event_func: t_ev: {t_ev}, X_ev: {X_ev}")
            R = X_ev[list(cfg.residual_indices)] - np.array(cfg.target)

            if np.linalg.norm(R, ord=np.inf) < tol:
                self._reset(); self._initial_state = X0
                self.period = 2 * t_ev
                return X0, t_ev

            _, _, Phi, _ = compute_stm(X0, self.mu, t_ev, forward=forward)
            logger.debug(f"Called compute_stm: Phi: {Phi}")
            J = Phi[np.ix_(cfg.residual_indices, cfg.control_indices)]

            if cfg.extra_jacobian is not None:
                J -= cfg.extra_jacobian(X_ev, Phi)

            delta = np.linalg.solve(J, -R)
            logger.info(f"Differential correction delta: {delta} at iteration {k}")
            X0[list(cfg.control_indices)] += delta
        
            logger.info(f"X0: {X0}")

        raise RuntimeError("did not converge")


class GenericOrbit(PeriodicOrbit):
    """
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
