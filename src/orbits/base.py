import warnings
import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Sequence, List, Tuple, Union
import numpy.typing as npt

from system import System
from algorithms.energy import crtbp_energy, energy_to_jacobi
from src.algorithms.integrators.standard import propagate_orbit
from algorithms.dynamics import stability_indices, compute_stm
from plots.plots import plot_orbit_rotating_frame, plot_orbit_inertial_frame
from log_config import logger


@dataclass
class orbitConfig:
    system: System
    orbit_family: str
    libration_point_idx: int
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that distance is positive.
        self.orbit_family = self.orbit_family.lower() # Normalize to lowercase

        if self.orbit_family not in ["halo", "lyapunov", "vertical_lyapunov"]:
            raise NotImplementedError(f"Orbit family {self.orbit_family} not implemented.")

        if self.libration_point_idx not in [1, 2, 3, 4, 5]:
            raise ValueError(f"Libration point index must be 1, 2, 3, 4, or 5. Got {self.libration_point_idx}.")

        # --- Family-specific parameter validation ---
        if self.orbit_family == "halo":
            if 'Zenith' not in self.extra_params:
                raise ValueError("Halo orbits require a 'Zenith' parameter ('northern' or 'southern').")
            zenith = self.extra_params['Zenith'].lower()

            if zenith not in ['northern', 'southern']:
                raise ValueError(f"Invalid Zenith '{self.extra_params['Zenith']}'. Must be 'northern' or 'southern'.")

            self.extra_params['Zenith'] = zenith

            if 'Az' not in self.extra_params:
                raise ValueError("Halo orbits require an 'Az' (z-amplitude) parameter.")

            if not isinstance(self.extra_params['Az'], (int, float)) or self.extra_params['Az'] <= 0:
                raise ValueError("'Az' must be a positive number.")

        elif self.orbit_family == "lyapunov":
            if 'Ax' not in self.extra_params:
                raise ValueError("Lyapunov orbits require an 'Ax' (x-amplitude) parameter.")

        elif self.orbit_family == "lissajous":
            if 'Ax' not in self.extra_params:
                raise ValueError("Lissajous orbits require an 'Ax' (x-amplitude) parameter.")

            if 'Az' not in self.extra_params:
                raise ValueError("Lissajous orbits require an 'Az' (z-amplitude) parameter.")

        if self.extra_params:
            logger.info(f"Extra parameters: {self.extra_params}")


class PeriodicOrbit(ABC):

    def __init__(self, config: orbitConfig, initial_state: Optional[Sequence[float]] = None):
        self._system = config.system
        self.mu = self._system.mu
        self.family = config.orbit_family
        self.libration_point = self._system.get_libration_point(config.libration_point_idx)

        self._initial_state = initial_state if initial_state else self._initial_guess()
        self.period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
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
    def is_unstable(self) -> bool:
        return not self.is_stable

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

    def propagate(self, steps: int = 1000, rtol: float = 1e-12, atol: float = 1e-12, 
                  **kwargs) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Propagate the orbit for one period.
        
        Parameters
        ----------
        steps : int, optional
            Number of time steps. Default is 1000.
        rtol : float, optional
            Relative tolerance for integration. Default is 1e-12.
        atol : float, optional
            Absolute tolerance for integration. Default is 1e-12.
        **kwargs
            Additional keyword arguments passed to the integrator
            
        Returns
        -------
        tuple
            (t, trajectory) containing the time and state arrays
        """
        if self.period is None:
            msg = "Period must be set before propagation"
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"Propagating orbit for period {self.period} with {steps} steps")
        tspan = np.linspace(0, self.period, steps)
        
        sol = propagate_orbit(
            self.initial_state, self.mu, tspan, 
            rtol=rtol, atol=atol, **kwargs
        )
        
        self._trajectory = sol.y.T  # Shape (steps, 6)
        self._times = sol.t
        logger.info(f"Propagation complete. Trajectory shape: {self._trajectory.shape}")
        
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

    def plot(self, frame="rotating", show=True, figsize=(10, 8), **kwargs):
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
            return self.plot_rotating_frame(show=show, figsize=figsize, **kwargs)
        elif frame.lower() == "inertial":
            return self.plot_inertial_frame(show=show, figsize=figsize, **kwargs)
        else:
            msg = f"Invalid frame '{frame}'. Must be 'rotating' or 'inertial'."
            logger.error(msg)
            raise ValueError(msg)

    def plot_rotating_frame(self, show=True, figsize=(10, 8), **kwargs):
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
        
        return plot_orbit_rotating_frame(
            trajectory=self._trajectory,
            mu=self.mu,
            system=self._system,
            libration_point=self.libration_point,
            family=self.family,
            show=show,
            figsize=figsize,
            **kwargs
        )
        
    def plot_inertial_frame(self, show=True, figsize=(10, 8), **kwargs):
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
        
        return plot_orbit_inertial_frame(
            trajectory=self._trajectory,
            times=self._times,
            mu=self.mu,
            system=self._system,
            family=self.family,
            show=show,
            figsize=figsize,
            **kwargs
        )
    
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

    @abstractmethod
    def differential_correction(self, **kwargs):
        """
        Perform differential correction to refine the orbit initial conditions.
        
        This method should update self._initial_state and self.period, then call
        self._reset_computed_properties() to ensure consistency.
        
        Parameters
        ----------
        **kwargs
            Algorithm-specific parameters for the differential correction process
            
        Returns
        -------
        bool
            True if correction was successful, False otherwise
        """
        pass


