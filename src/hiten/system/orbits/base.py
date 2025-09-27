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

from hiten.algorithms.common.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.corrector.config import (_LineSearchConfig,
                                               _OrbitCorrectionConfig)
from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.orbits import (_OrbitCorrectionService,
                                                    _OrbitPersistenceService,
                                                    _OrbitServices)
from hiten.algorithms.types.states import (ReferenceFrame, SynodicStateVector,
                                           Trajectory)
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.utils.io.common import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import (animate_trajectories, plot_inertial_frame,
                               plot_rotating_frame)

if TYPE_CHECKING:
    from hiten.algorithms.continuation.config import _OrbitContinuationConfig
    from hiten.system.manifold import Manifold


class PeriodicOrbit(_HitenBase, ABC):
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
        services = _OrbitServices.default(self)
        super().__init__(services)

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
        return self.dynamics.stability_indices

    @property
    def eigenvalues(self) -> Optional[Tuple]:
        return self.dynamics.eigenvalues
    
    @property
    def eigenvectors(self) -> Optional[Tuple]:
        return self.dynamics.eigenvectors

    @property
    def system(self) -> System:
        """Get the parent CR3BP system.
        
        Returns
        -------
        :class:`~hiten.system.base.System`
            The parent CR3BP system.
        """
        return self.dynamics.system

    @property
    def libration_point(self) -> LibrationPoint:
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
        """Differential correction wrapper."""
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

        result = self._services.correction.correct(self, overrides=overrides)
        self._initial_state = result.corrected_state
        self._period = result.period
        self._trajectory = None
        self._times = None
        self._stability_info = None
        return result.corrected_state, result.period

    def propagate(self, steps: int = 1000, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8) -> Trajectory:
        """Propagate the orbit."""
        return self.dynamics.propagate(steps=steps, method=method, order=order)

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
        """Save the orbit to a file."""
        self._services.persistence.save(self, filepath, **kwargs)

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load orbit data from a file in place."""
        self._services.persistence.load_inplace(self, filepath, **kwargs)
        if getattr(self, "_system", None) is not None:
            self._services = _OrbitServices.for_system(self._system)
        return

    @classmethod
    def load(cls, filepath: str, **kwargs) -> "PeriodicOrbit":
        """Load an orbit from a file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")
        
        def services_factory(orbit):
            system = getattr(orbit, "_system", None)
            if system is None:
                raise ValueError("Serialized orbit is missing system metadata and cannot be rehydrated.")
            return _OrbitServices.for_system(system)
        
        return cls._load_with_services(
            filepath, 
            _OrbitPersistenceService(), 
            services_factory, 
            **kwargs
        )

    def __setstate__(self, state):
        """Restore the PeriodicOrbit instance after unpickling."""
        super().__setstate__(state)
        self._setup_services(_OrbitServices.for_system(self._system))

    def __getstate__(self):
        """Custom state extractor to enable pickling."""
        state = self.__dict__.copy()
        state.pop("_services", None)
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
