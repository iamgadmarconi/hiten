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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from hiten.algorithms.corrector.config import _LineSearchConfig, _OrbitCorrectionConfig
from hiten.algorithms.corrector.stepping import (make_armijo_stepper,
                                                 make_plain_stepper)
from hiten.algorithms.corrector.backends.newton import _NewtonBackend
from hiten.algorithms.corrector.engine import _OrbitCorrectionEngine
from hiten.algorithms.corrector.interfaces import _PeriodicOrbitCorrectorInterface
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import (_compute_monodromy, _compute_stm,
                                            _stability_indices)
from hiten.algorithms.dynamics.utils.energy import (crtbp_energy,
                                                    energy_to_jacobi)
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.utils.io.common import _ensure_dir
from hiten.utils.io.orbits import (load_periodic_orbit,
                                   load_periodic_orbit_inplace,
                                   save_periodic_orbit)
from hiten.utils.log_config import logger
from hiten.utils.plots import (animate_trajectories, plot_inertial_frame,
                               plot_rotating_frame)

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import _OrbitContinuationConfig
    from hiten.system.manifold import Manifold


class PeriodicOrbit(ABC):
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
    
    # This should be overridden by subclasses
    _family: str = "generic"

    def __init__(self, libration_point: LibrationPoint, initial_state: Optional[Sequence[float]] = None):
        self._libration_point = libration_point
        self._system = self._libration_point.system
        self._mu = self._system.mu

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

        self._period = None
        self._trajectory = None
        self._times = None
        self._stability_info = None
        
        # General initialization log
        logger.info(f"Initialized {self.family} orbit around L{self.libration_point.idx}")

        # Algorithm-level correction parameter overrides (applied to config lazily)
        self._correction_overrides: dict[str, object] = {}

    def __str__(self):
        return f"{self.family} orbit around {self._libration_point}."

    def __repr__(self):
        return f"{self.__class__.__name__}(family={self.family}, libration_point={self._libration_point})"

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
    def libration_point(self) -> LibrationPoint:
        """The libration point instance that anchors the family.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point instance.
        """
        return self._libration_point

    @property
    def initial_state(self) -> npt.NDArray[np.float64]:
        """
        Get the initial state vector of the orbit.
        
        Returns
        -------
        numpy.ndarray, shape (6,)
            The initial state vector [x, y, z, vx, vy, vz] in nondimensional units.
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
            Array of time points in nondimensional units, or None if the trajectory
            hasn't been computed yet.
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
            Tuple containing (_stability_indices, eigenvalues, eigenvectors),
            or None if stability hasn't been computed yet.
        """
        if self._stability_info is None:
            logger.warning("Stability information not computed. Call compute_stability() first.")
        return self._stability_info

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit."""
        pass

    @property
    def period(self) -> Optional[float]:
        """Orbit period, set after a successful correction.
        
        Returns
        -------
        float or None
            The orbit period in nondimensional units, or None if not set.
        """
        return self._period

    @period.setter
    def period(self, value: Optional[float]):
        """Set the orbit period and invalidate cached data.

        Setting the period manually allows users (or serialization logic)
        to override the value obtained via differential correction. Any time
        the period changes we must invalidate cached trajectory, time array
        and stability information so they can be recomputed consistently.
        
        Parameters
        ----------
        value : float or None
            The orbit period in nondimensional units, or None to clear.
            
        Raises
        ------
        ValueError
            If value is not positive.
        """
        # Basic validation: positive period or None
        if value is not None and value <= 0:
            raise ValueError("period must be a positive number or None.")

        # Only act if the period actually changes to avoid unnecessary resets
        current_period = getattr(self, "_period", None)
        if value != current_period:
            # Ensure the private attribute exists before use
            self._period = value

            # Invalidate caches that depend on the period, if they already exist
            if hasattr(self, "_trajectory"):
                self._trajectory = None
            if hasattr(self, "_times"):
                self._times = None
            if hasattr(self, "_stability_info"):
                self._stability_info = None
            if hasattr(self, "_monodromy"):
                self._monodromy = None

            logger.info("Period updated, cached trajectory, times and stability information cleared")

    @property
    def system(self) -> System:
        """Get the parent CR3BP system.
        
        Returns
        -------
        :class:`~hiten.system.base.System`
            The parent CR3BP system.
        """
        return self._system

    @property
    def mu(self) -> float:
        """Mass ratio of the system.
        
        Returns
        -------
        float
            The mass ratio (dimensionless).
        """
        return self._mu

    @property
    def is_stable(self) -> bool:
        """
        Check if the orbit is linearly stable.
        
        Returns
        -------
        bool
            True if all stability indices have magnitude <= 1, False otherwise.
        """
        if self._stability_info is None:
            logger.info("Computing stability for stability check")
            self.compute_stability()
        
        indices = self._stability_info[0]  # nu values from _stability_indices
        
        # An orbit is stable if all stability indices have magnitude <= 1
        return np.all(np.abs(indices) <= 1.0)

    @property
    def energy(self) -> float:
        """
        Compute the energy of the orbit at the initial state.
        
        Returns
        -------
        float
            The energy value in nondimensional units.
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
            The Jacobi constant value (dimensionless).
        """
        return energy_to_jacobi(self.energy)
    
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
        if self.period is None:
            raise ValueError("Period must be set before computing monodromy")
        
        Phi = _compute_monodromy(self.libration_point._var_eq_system, self.initial_state, self.period)
        return Phi

    @property
    @abstractmethod
    def eccentricity(self):
        pass

    @property
    @abstractmethod
    def _correction_config(self) -> "_OrbitCorrectionConfig":
        """Provides the differential correction configuration for this orbit family."""
        pass

    @property
    @abstractmethod
    def _continuation_config(self) -> "_OrbitContinuationConfig":
        """Default parameter for family continuation (must be overridden)."""
        raise NotImplementedError

    def _reset(self) -> None:
        """
        Reset all computed properties when the initial state is changed.
        Called internally after differential correction or any other operation
        that modifies the initial state.
        """
        self._trajectory = None
        self._times = None
        self._stability_info = None
        self._period = None
        self._monodromy = None
        logger.debug("Reset computed orbit properties due to state change")

    @abstractmethod
    def _initial_guess(self, **kwargs):
        """Provides the initial guess for the differential correction."""
        raise NotImplementedError

    def update_correction(self, **kwargs) -> None:
        """Update algorithm-level correction parameters for this orbit.

        Allowed keys: tol, max_attempts, max_delta, line_search_config,
        finite_difference, forward.
        """
        allowed = {"tol", "max_attempts", "max_delta", "line_search_config", "finite_difference", "forward"}
        invalid = [k for k in kwargs.keys() if k not in allowed]
        if invalid:
            raise KeyError(f"Invalid correction parameter(s): {invalid}. Allowed: {sorted(allowed)}")
        self._correction_overrides.update({k: v for k, v in kwargs.items() if v is not None})

    def clear_correction_overrides(self) -> None:
        """Clear any previously set correction parameter overrides."""
        self._correction_overrides.clear()

    def _apply_correction_overrides(self, cfg: "_OrbitCorrectionConfig") -> "_OrbitCorrectionConfig":
        if not self._correction_overrides:
            return cfg
        from dataclasses import replace as _dc_replace
        # Apply only attributes that exist on the config
        valid = {k: v for k, v in self._correction_overrides.items() if hasattr(cfg, k)}
        return _dc_replace(cfg, **valid)

    @abstractmethod
    def _correction_config(self) -> _OrbitCorrectionConfig:
        """Provides the differential correction configuration for this orbit family."""
        raise NotImplementedError

    def correct(
            self,
            *,
            tol: float | None = None,
            max_attempts: int | None = None,
            forward: int | None = None,
            max_delta: float | None = None,
            line_search_config: _LineSearchConfig | bool | None = None,
            finite_difference: bool | None = None,
        ) -> tuple[np.ndarray, float]:
        """Differential correction wrapper.

        This method now delegates the heavy lifting to the generic
        :class:`~hiten.algorithms.corrector.newton._NewtonOrbitCorrector` which
        implements a robust Newton-Armijo scheme.
        
        Parameters
        ----------
        tol: float, optional
            Convergence tolerance for the residual norm. The algorithm terminates
            successfully when the norm of the residual falls below this value.
        max_attempts: int, optional
            Maximum number of Newton iterations to attempt before declaring
            convergence failure.
        forward: int, optional
        max_delta: float, optional
            Maximum allowed infinity norm of Newton steps. 
        line_search_config: _LineSearchConfig | bool | None, optional
            Configuration for line search behavior:

            - True: Enable line search with default parameters
            - False or None: Disable line search (use full Newton steps)
            - :class:`~hiten.algorithms.corrector.config._LineSearchConfig`: Enable line search with custom parameters  

        finite_difference: bool, optional
            Force finite-difference approximation of Jacobians even when
            analytic Jacobians are available.

        Returns
        -------
        tuple
            (corrected_state, period) in nondimensional units.
        """
        # Apply any call-time overrides into per-orbit overrides for this run
        overrides: dict[str, object] = {}
        if tol is not None:
            overrides["tol"] = tol
        if max_attempts is not None:
            overrides["max_attempts"] = max_attempts
        if forward is not None:
            overrides["forward"] = forward
        if max_delta is not None:
            overrides["max_delta"] = max_delta
        if line_search_config is not None:
            overrides["line_search_config"] = line_search_config
        if finite_difference is not None:
            overrides["finite_difference"] = finite_difference
        if overrides:
            self.update_correction(**overrides)

        # Select family configuration, then apply per-orbit overrides
        cfg_base = self._correction_config
        cfg = self._apply_correction_overrides(cfg_base)

        # Build stepper factory based on line search configuration
        if cfg.line_search_config is True:
            stepper_factory = make_armijo_stepper(_LineSearchConfig())
        elif cfg.line_search_config is False or cfg.line_search_config is None:
            stepper_factory = make_plain_stepper()
        else:
            # Provided a custom _LineSearchConfig
            stepper_factory = make_armijo_stepper(cfg.line_search_config)

        backend = _NewtonBackend(stepper_factory=stepper_factory)
        interface = _PeriodicOrbitCorrectorInterface()
        engine = _OrbitCorrectionEngine(backend=backend, interface=interface)

        result, half_period = engine.solve(self, cfg)
        interface.apply_results_to_orbit(self, corrected_state=result.x_corrected, half_period=half_period)
        return result.x_corrected, half_period

    def propagate(self, steps: int = 1000, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Propagate the orbit for one period.
        
        Parameters
        ----------
        steps : int, optional
            Number of time steps. Default is 1000.
        method : str, optional
            Integration method. Default is "adaptive".
        order : int, optional
            Integration order. Default is 8.
            
        Returns
        -------
        tuple
            (times, trajectory) containing the time and state arrays in
            nondimensional units.
            
        Raises
        ------
        ValueError
            If period is not set.
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
        """
        Compute stability information for the orbit.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the STM computation.
            
        Returns
        -------
        tuple
            (_stability_indices, eigenvalues, eigenvectors) from the monodromy matrix.
            
        Raises
        ------
        ValueError
            If period is not set.
        """
        if self.period is None:
            msg = "Period must be set before stability analysis"
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"Computing stability for orbit with period {self.period}")
        # Compute STM over one period
        _, _, Phi, _ = _compute_stm(self.libration_point._var_eq_system, self.initial_state, self.period)
        
        # Analyze stability
        stability = _stability_indices(Phi)
        self._stability_info = stability
        
        is_stable = np.all(np.abs(stability[0]) <= 1.0)
        logger.info(f"Orbit stability: {'stable' if is_stable else 'unstable'}")
        
        return stability

    def manifold(self, stable: bool = True, direction: Literal["positive", "negative"] = "positive") -> "Manifold":
        """Create a manifold object for this orbit.
        
        Parameters
        ----------
        stable : bool, optional
            Whether to create a stable manifold. Default is True.
        direction : str, optional
            Direction of the manifold ("positive" or "negative"). Default is "positive".
            
        Returns
        -------
        :class:`~hiten.system.manifold.Manifold`
            The manifold object.
        """
        from hiten.system.manifold import Manifold
        return Manifold(self, stable=stable, direction=direction)

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
        if self._trajectory is None:
            msg = "No trajectory to plot. Call propagate() first."
            logger.error(msg)
            raise RuntimeError(msg)
            
        if frame.lower() == "rotating":
            return plot_rotating_frame(
                states=self._trajectory, 
                times=self._times, 
                bodies=[self._system.primary, self._system.secondary], 
                system_distance=self._system.distance, 
                dark_mode=dark_mode, 
                save=save,
                filepath=filepath,
                **kwargs)
        elif frame.lower() == "inertial":
            return plot_inertial_frame(
                states=self._trajectory, 
                times=self._times, 
                bodies=[self._system.primary, self._system.secondary], 
                system_distance=self._system.distance, 
                dark_mode=dark_mode, 
                save=save,
                filepath=filepath,
                **kwargs)
        else:
            msg = f"Invalid frame '{frame}'. Must be 'rotating' or 'inertial'."
            logger.error(msg)
            raise ValueError(msg)
        
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
        if self._trajectory is None:
            logger.warning("No trajectory to animate. Call propagate() first.")
            return None, None
        
        return animate_trajectories(self._trajectory, self._times, [self._system.primary, self._system.secondary], self._system.distance, **kwargs)

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
        if self._trajectory is None or self._times is None:
            err = "Trajectory not computed. Please call propagate() first."
            logger.error(err)
            raise ValueError(err)

        # Assemble the data: time followed by the six-dimensional state vector
        data = np.column_stack((self._times, self._trajectory))
        df = pd.DataFrame(data, columns=["time", "x", "y", "z", "vx", "vy", "vz"])

        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        df.to_csv(filepath, index=False)
        logger.info(f"Orbit trajectory successfully exported to {filepath}")

    def save(self, filepath: str, **kwargs) -> None:
        """Save the orbit to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the orbit file.
        **kwargs
            Additional keyword arguments passed to the save function.
        """
        save_periodic_orbit(self, filepath, **kwargs)
        return

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load orbit data from a file in place.
        
        Parameters
        ----------
        filepath : str
            Path to the orbit file.
        **kwargs
            Additional keyword arguments passed to the load function.
            
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")

        load_periodic_orbit_inplace(self, filepath, **kwargs)
        return

    @classmethod
    def load(cls, filepath: str, **kwargs) -> "PeriodicOrbit":
        """Load an orbit from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the orbit file.
        **kwargs
            Additional keyword arguments passed to the load function.
            
        Returns
        -------
        :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The loaded orbit instance.
            
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")

        return load_periodic_orbit(filepath, **kwargs)

    def __setstate__(self, state):
        """Restore the PeriodicOrbit instance after unpickling.

        The cached dynamical system used for high-performance propagation is
        removed before pickling (it may contain numba objects) and recreated
        lazily on first access after unpickling.
        
        Parameters
        ----------
        state : dict
            The object state dictionary from pickling.
        """
        # Simply update the dictionary - the cached dynamical system will be
        # rebuilt lazily when needed.
        self.__dict__.update(state)

    def __getstate__(self):
        """Custom state extractor to enable pickling.

        We strip attributes that might keep references to non-pickleable numba
        objects (e.g. the cached dynamical system) while leaving all the
        essential orbital data untouched.
        
        Returns
        -------
        dict
            The object state dictionary with unpickleable objects removed.
        """
        state = self.__dict__.copy()
        # Remove the cached CR3BP dynamical system wrapper
        if "_cached_dynsys" in state:
            state["_cached_dynsys"] = None
        return state


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
    
    def __init__(self, libration_point: LibrationPoint, initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state)
        self._custom_correction_config: Optional["_OrbitCorrectionConfig"] = None
        self._custom_continuation_config: Optional["_OrbitContinuationConfig"] = None
        if self._period is None:
            self._period = np.pi

        self._amplitude = None

    @property
    def correction_config(self) -> Optional["_OrbitCorrectionConfig"]:
        """
        Get or set the user-defined differential correction configuration.

        This property must be set to a valid :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
        instance before calling :meth:`~hiten.system.orbits.base.PeriodicOrbit.correct` on a
        :class:`~hiten.system.orbits.base.GenericOrbit` object.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig` or None
            The correction configuration, or None if not set.
        """
        return self._custom_correction_config

    @correction_config.setter
    def correction_config(self, value: Optional["_OrbitCorrectionConfig"]):
        """Set the correction configuration.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig` or None
            The correction configuration to set.
            
        Raises
        ------
        TypeError
            If value is not an instance of :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig` or None.
        """
        from hiten.algorithms.corrector.config import _OrbitCorrectionConfig
        if value is not None and not isinstance(value, _OrbitCorrectionConfig):
            raise TypeError("correction_config must be an instance of _OrbitCorrectionConfig or None.")
        self._custom_correction_config = value

    @property
    def eccentricity(self):
        """Eccentricity is not well-defined for generic orbits.
        
        Returns
        -------
        float
            NaN since eccentricity is not defined for generic orbits.
        """
        return np.nan

    @property
    def _correction_config(self) -> "_OrbitCorrectionConfig":
        """
        Provides the differential correction configuration.

        For GenericOrbit, this must be set via the `correction_config` property
        to enable differential correction.
        
        Returns
        -------
        :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`
            The correction configuration.
            
        Raises
        ------
        NotImplementedError
            If correction_config is not set.
        """
        if self.correction_config is not None:
            return self.correction_config
        raise NotImplementedError(
            "Differential correction is not defined for a GenericOrbit unless the "
            "`correction_config` property is set with a valid :class:`~hiten.algorithms.corrector.config._OrbitCorrectionConfig`."
        )

    @property
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit.
        
        Returns
        -------
        float or None
            The orbit amplitude in nondimensional units, or None if not set.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        """Set the orbit amplitude.
        
        Parameters
        ----------
        value : float
            The orbit amplitude in nondimensional units.
        """
        self._amplitude = value

    @property
    def continuation_config(self) -> Optional["_OrbitContinuationConfig"]:
        """Get or set the continuation parameter for this orbit.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig` or None
            The continuation configuration, or None if not set.
        """
        return self._custom_continuation_config

    @continuation_config.setter
    def continuation_config(self, cfg: Optional["_OrbitContinuationConfig"]):
        """Set the continuation configuration.
        
        Parameters
        ----------
        cfg : :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig` or None
            The continuation configuration to set.
            
        Raises
        ------
        TypeError
            If cfg is not an instance of :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig` or None.
        """
        from hiten.algorithms.continuation.config import \
            _OrbitContinuationConfig
        if cfg is not None and not isinstance(cfg, _OrbitContinuationConfig):
            raise TypeError("continuation_config must be a _OrbitContinuationConfig instance or None")
        self._custom_continuation_config = cfg

    @property
    def _continuation_config(self) -> "_OrbitContinuationConfig":  # used by engines
        """Provides the continuation configuration for engines.
        
        Returns
        -------
        :class:`~hiten.algorithms.continuation.config._OrbitContinuationConfig`
            The continuation configuration.
            
        Raises
        ------
        NotImplementedError
            If continuation_config is not set.
        """
        if self._custom_continuation_config is None:
            raise NotImplementedError(
                "GenericOrbit requires 'continuation_config' to be set before using continuation engines."
            )
        return self._custom_continuation_config

    def _initial_guess(self, **kwargs):
        """Generate initial guess for GenericOrbit.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments (unused).
            
        Returns
        -------
        numpy.ndarray, shape (6,)
            The initial state vector in nondimensional units.
            
        Raises
        ------
        ValueError
            If no initial state is provided.
        """
        if hasattr(self, '_initial_state') and self._initial_state is not None:
            return self._initial_state
        raise ValueError("No initial state provided for GenericOrbit.")
