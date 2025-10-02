"""Abstract definitions and convenience utilities for periodic orbit computation
in the circular restricted three-body problem (CR3BP).

This module provides the foundational classes for working with periodic orbits
in the CR3BP, including abstract base classes and concrete implementations
for various orbit families.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Szebehely, V. (1967). "Theory of Orbits - The Restricted Problem of Three
Bodies".
"""
import os
from typing import TYPE_CHECKING, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from hiten.algorithms.corrector.config import OrbitCorrectionConfig
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.orbits import (_OrbitPersistenceService,
                                                    _OrbitServices)
from hiten.algorithms.types.states import Trajectory
from hiten.utils.io.common import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import (animate_trajectories, plot_inertial_frame,
                               plot_rotating_frame)

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import OrbitContinuationConfig
    from hiten.algorithms.continuation.options import OrbitContinuationOptions
    from hiten.algorithms.corrector.config import OrbitCorrectionConfig
    from hiten.algorithms.corrector.options import OrbitCorrectionOptions
    from hiten.system.base import System
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.manifold import Manifold
    from hiten.algorithms.corrector.types import CorrectionResult
    from hiten.algorithms.continuation.types import ContinuationResult

class PeriodicOrbit(_HitenBase):
    """
    Abstract base-class that encapsulates a CR3BP periodic orbit.

    The constructor either accepts a user supplied initial state or derives an
    analytical first guess via :meth:`~hiten.system.orbits.base.PeriodicOrbit._initial_guess` (to be
    implemented by subclasses). All subsequent high-level operations
    (propagation, plotting, stability analysis, differential correction) build
    upon this initial description.

    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point instance that anchors the family.
    initial_state : Sequence[float] or None, optional
        Initial condition in rotating canonical units
        [x, y, z, vx, vy, vz]. When None an analytical
        approximation is attempted.

    Attributes
    ----------
    family : str
        Orbit family name (settable property with class-specific defaults).
    libration_point : :class:`~hiten.system.libration.base.LibrationPoint`
        Libration point anchoring the family.
    system : :class:`~hiten.system.base.System`
        Parent CR3BP system.
    mu : float
        Mass ratio of the system, accessed as system.mu (dimensionless).
    initial_state : ndarray, shape (6,)
        Current initial condition in nondimensional units.
    period : float or None
        Orbit period, set after a successful correction (nondimensional units).
    trajectory : ndarray or None, shape (N, 6)
        Stored trajectory after :meth:`~hiten.system.orbits.base.PeriodicOrbit.propagate`.
    times : ndarray or None, shape (N,)
        Time vector associated with trajectory (nondimensional units).
    stability_info : tuple or None
        Output of :func:`~hiten.algorithms.dynamics.rtbp._stability_indices`.

    Notes
    -----
    Instantiating the class does not perform any propagation. Users must
    call :meth:`~hiten.system.orbits.base.PeriodicOrbit.correct` (or manually set
    period) followed by :meth:`~hiten.system.orbits.base.PeriodicOrbit.propagate`.
    """

    _family: str = "base"

    def __init__(self, libration_point: "LibrationPoint", initial_state: Optional[Sequence[float]] = None):
        self._libration_point = libration_point
        self._initial_state = initial_state
        services = _OrbitServices.default(self)
        super().__init__(services)

    def __str__(self):
        return f"{self.family} orbit around {self._libration_point}."

    def __repr__(self):
        return f"{self.__class__.__name__}(family={self.family}, libration_point={self._libration_point})"

    @property
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit.
        
        Returns
        -------
        float or None
            The orbit amplitude in nondimensional units, or None if not set.
        """
        return self.dynamics.amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        """Set the orbit amplitude.
        
        Parameters
        ----------
        value : float
            The orbit amplitude in nondimensional units.
        """
        self.dynamics.amplitude = value

    @property
    def family(self) -> str:
        """
        Get the orbit family name.
        
        Returns
        -------
        str
            The orbit family name.
        """
        return self._family

    @property
    def initial_state(self) -> npt.NDArray[np.float64]:
        """
        Get the initial state vector of the orbit.
        
        Returns
        -------
        numpy.ndarray, shape (6,)
            The initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
        """
        return self.dynamics.initial_state
    
    @property
    def stability_indices(self) -> Optional[Tuple]:
        """The stability indices of the orbit."""
        return self.dynamics.stability_indices

    @property
    def eigenvalues(self) -> Optional[Tuple]:
        """The eigenvalues of the orbit."""
        return self.dynamics.eigenvalues
    
    @property
    def eigenvectors(self) -> Optional[Tuple]:
        """The eigenvectors of the orbit."""
        return self.dynamics.eigenvectors

    @property
    def energy(self) -> float:
        """Orbit energy.
        
        Returns
        -------
        float
            The energy value in nondimensional units.
        """
        return self.dynamics.energy

    @property
    def jacobi(self) -> float:
        """Jacobi constant.
        
        Returns
        -------
        float
            The Jacobi constant value (dimensionless).
        """
        return self.dynamics.jacobi_constant

    @property
    def system(self) -> "System":
        """Get the parent CR3BP system.
        
        Returns
        -------
        :class:`~hiten.system.base.System`
            The parent CR3BP system.
        """
        return self.dynamics.system

    @property
    def libration_point(self) -> "LibrationPoint":
        """Get the libration point around which the orbit is computed.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point around which the orbit is computed.
        """
        return self.dynamics.libration_point

    @property
    def mu(self) -> float:
        """Mass ratio of the system.
        
        Returns
        -------
        float
            The mass ratio (dimensionless).
        """
        return self.dynamics.mu
    
    @property
    def period(self) -> Optional[float]:
        """Orbit period.
        
        Returns
        -------
        float or None
            The orbit period in nondimensional units, or None if not set.
        """
        return self.dynamics.period
    
    @period.setter
    def period(self, value: Optional[float]):
        """Set the orbit period.
        
        Parameters
        ----------
        value : float or None
            The orbit period in nondimensional units, or None to clear.
        """
        self.dynamics.period = value

    @property
    def trajectory(self) -> Optional[Trajectory]:
        """The trajectory of the orbit."""
        return self.dynamics.trajectory
    
    @property
    def trajectories(self) -> List[Trajectory]:
        """List of trajectories (for SynodicMap compatibility)."""
        return self.dynamics.trajectories
    
    @property
    def monodromy(self) -> np.ndarray:
        """
        Compute the monodromy matrix of the orbit.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            The monodromy matrix.
            
        Raises
        ------
        ValueError
            If period is not set.
        """
        return self.dynamics.monodromy

    @property
    def correction_config(self) -> Optional["OrbitCorrectionConfig"]:
        """
        Provides the differential correction configuration.

        For GenericOrbit, this must be set via the `correction_config` property
        to enable differential correction.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig`
            The correction configuration.
            
        Raises
        ------
        NotImplementedError
            If correction_config is not set.
        """
        return self._correction.correction_config

    @correction_config.setter
    def correction_config(self, value: Optional["OrbitCorrectionConfig"]):
        """Set the correction configuration.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig` or None
            The correction configuration to set.
            
        Raises
        ------
        TypeError
            If value is not an instance of :class:`~hiten.algorithms.corrector.config.OrbitCorrectionConfig` or None.
        """
        self._correction.correction_config = value

    @property
    def correction_options(self) -> Optional["OrbitCorrectionOptions"]:
        """Get or set the correction options for this orbit.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions` or None
            The correction options, or None if not set.
        """
        return self._correction.correction_options

    @correction_options.setter
    def correction_options(self, value: Optional["OrbitCorrectionOptions"]):
        """Set the correction options.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions` or None
            The correction options to set.
        """
        self._correction.correction_options = value

    @property
    def continuation_config(self) -> Optional["OrbitContinuationConfig"]:
        """Get or set the continuation parameter for this orbit.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig` or None
            The continuation configuration, or None if not set.
        """
        return self._continuation.continuation_config

    @continuation_config.setter
    def continuation_config(self, cfg: Optional["OrbitContinuationConfig"]):
        """Set the continuation configuration.
        
        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig` or None
            The continuation configuration to set.
            
        Raises
        ------
        TypeError
            If cfg is not an instance of :class:`~hiten.algorithms.continuation.config.OrbitContinuationConfig` or None.
        """
        self._continuation.continuation_config = cfg

    @property
    def continuation_options(self) -> Optional["OrbitContinuationOptions"]:
        """Get or set the continuation options for this orbit.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions` or None
            The continuation options, or None if not set.
        """
        return self._continuation.continuation_options

    @continuation_options.setter
    def continuation_options(self, value: Optional["OrbitContinuationOptions"]):
        """Set the continuation options.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.continuation.options.OrbitContinuationOptions` or None
            The continuation options to set.
        """
        self._continuation.continuation_options = value

    def correct(self, options: Optional["OrbitCorrectionOptions"] = None) -> "CorrectionResult":
        """Differential correction wrapper.
        
        Parameters
        ----------
        options: :class:`~hiten.algorithms.corrector.options.OrbitCorrectionOptions` or None
            Additional keyword arguments passed to the correction method.

            - tol: float
                Convergence tolerance for the residual norm.
            - max_attempts: int
                Maximum number of Newton iterations to attempt before declaring
                convergence failure.
            - max_delta: float
                Maximum allowed infinity norm of Newton steps.
            - line_search_config: :class:`~hiten.algorithms.corrector.config._LineSearchConfig`
                Configuration for line search behavior.
            - finite_difference: bool
                Force finite-difference approximation of Jacobians even when
                analytic Jacobians are available.
            - forward: int
                Integration direction (1 for forward, -1 for backward).
            
        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.CorrectionResult`
            The corrected state and period.
        """
        state, period, result = self._correction.correct(options=options)
        return result

    def generate(self, options: Optional["OrbitContinuationOptions"] = None) -> "ContinuationResult":
        """Generate a family of periodic orbits."""
        result = self._continuation.generate(options=options)
        return result

    def propagate(self, steps: int = 1000, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8) -> Trajectory:
        """Propagate the orbit.
        
        Parameters
        ----------
        steps: int, default 1000
            Number of integration steps. Default is 1000.
        method: Literal["fixed", "adaptive", "symplectic"]
            Integration method. Default is "adaptive".
        order: int, default 8
            Integration order.
            
        Returns
        -------
        :class:`~hiten.algorithms.types.states.Trajectory`
            The propagated trajectory.
        """
        return self.dynamics.propagate(steps=steps, method=method, order=order)

    def manifold(self, stable: bool = True, direction: Literal["positive", "negative"] = "positive") -> "Manifold":
        """Create a manifold object for this orbit.
        
        Parameters
        ----------
        stable : bool, optional
            Whether to create a stable manifold. Default is True.
        direction : Literal["positive", "negative"], optional
            Direction of the manifold ("positive" or "negative"). Default is "positive".
            
        Returns
        -------
        :class:`~hiten.system.manifold.Manifold`
            The manifold object.
        """
        return self.dynamics.manifold(stable=stable, direction=direction)

    def plot(self, frame: Literal["rotating", "inertial"] = "rotating", dark_mode: bool = True, save: bool = False, filepath: str = f'orbit.svg', **kwargs):
        """Plot the orbit trajectory.
        
        Parameters
        ----------
        frame : str, optional
            Reference frame for plotting ("rotating" or "inertial"). Default is "rotating".
        dark_mode : bool, optional
            Whether to use dark mode for plotting. Default is True.
        save : bool, optional
            Whether to save the plot to file. Default is False.
        filepath : str, optional
            Path to save the plot. Default is "orbit.svg".
        **kwargs
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        matplotlib.figure.Figure
            The plot figure.
            
        Raises
        ------
        RuntimeError
            If trajectory is not computed.
        ValueError
            If frame is invalid.
        """
        try:
            if self.trajectory is None:
                raise RuntimeError("No trajectory to plot. Call propagate() first.")
        except ValueError:
            raise RuntimeError("No trajectory to plot. Call propagate() first.")

        states = self.trajectory.states
        times = self.trajectory.times
        
        if frame.lower() == "rotating":
            return plot_rotating_frame(
                states=states, 
                times=times, 
                bodies=[self.system.primary, self.system.secondary], 
                system_distance=self.system.distance, 
                dark_mode=dark_mode, 
                save=save,
                filepath=filepath,
                **kwargs)
        elif frame.lower() == "inertial":
            return plot_inertial_frame(
                states=states, 
                times=times, 
                bodies=[self.system.primary, self.system.secondary], 
                system_distance=self.system.distance, 
                dark_mode=dark_mode, 
                save=save,
                filepath=filepath,
                **kwargs)
        else:
            raise ValueError(f"Invalid frame '{frame}'. Must be 'rotating' or 'inertial'.")
        
    def animate(self, **kwargs):
        """Create an animation of the orbit trajectory.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the animation function.
            
        Returns
        -------
        tuple or None
            Animation objects, or None if trajectory is not computed.
        """
        try:
            if self.trajectory is None:
                logger.warning("No trajectory to animate. Call propagate() first.")
                return None, None
        except ValueError:
            logger.warning("No trajectory to animate. Call propagate() first.")
            return None, None
        
        return animate_trajectories(self.trajectory.states, self.trajectory.times, [self.system.primary, self.system.secondary], self.system.distance, **kwargs)

    def to_csv(self, filepath: str, **kwargs):
        """Export the orbit trajectory to a CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to save the CSV file.
        **kwargs
            Additional keyword arguments passed to pandas.DataFrame.to_csv.
            
        Raises
        ------
        ValueError
            If trajectory is not computed.
        """
        try:
            traj = self.trajectory
            if traj is None:
                raise ValueError("Trajectory not computed. Please call propagate() first.")
        except ValueError as e:
            if "Trajectory not computed" in str(e):
                raise ValueError("Trajectory not computed. Please call propagate() first.") from e
            raise

        data = np.column_stack((traj.times, traj.states))
        df = pd.DataFrame(data, columns=["time", "x", "y", "z", "vx", "vy", "vz"])

        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        df.to_csv(filepath, index=False)
        logger.info(f"Orbit trajectory successfully exported to {filepath}")

    def to_df(self, **kwargs) -> pd.DataFrame:
        """Export the orbit trajectory to a pandas DataFrame.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to pandas.DataFrame.to_csv.
        """
        try:
            traj = self.trajectory
            if traj is None:
                raise ValueError("Trajectory not computed. Please call propagate() first.")
        except ValueError as e:
            if "Trajectory not computed" in str(e):
                raise ValueError("Trajectory not computed. Please call propagate() first.") from e
            raise
        
        return pd.DataFrame(np.column_stack((traj.times, traj.states)), columns=["time", "x", "y", "z", "vx", "vy", "vz"])

    def __setstate__(self, state):
        """Restore the PeriodicOrbit instance after unpickling."""
        super().__setstate__(state)
        self._setup_services(_OrbitServices.default(self))

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load orbit data from a file in place."""
        self.persistence.load_inplace(self, filepath)
        self.dynamics.reset()
        return self

    @classmethod
    def load(cls, filepath: str, **kwargs) -> "PeriodicOrbit":
        """Load an orbit from a file."""
        return cls._load_with_services(
            filepath, 
            _OrbitPersistenceService(), 
            _OrbitServices.default, 
            **kwargs
        )


class GenericOrbit(PeriodicOrbit):
    """
    A minimal concrete orbit class for arbitrary initial conditions.
    
    This class provides a basic implementation of PeriodicOrbit that can be
    used with arbitrary initial conditions. It requires manual configuration
    of correction and continuation parameters.
    
    Parameters
    ----------
    libration_point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point around which the orbit is computed.
    initial_state : Sequence[float], optional
        Initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
        If None, a default period of pi is set.
    """
    
    _family = "generic"
    
    def __init__(self, libration_point: "LibrationPoint", initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state=initial_state)
    
        if self.dynamics.period is None:
            self.dynamics.period = np.pi



