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
* Light-weight configuration containers (:pyclass:`_CorrectionConfig`) that 
  encapsulate user input for differential correction settings.

References
----------
Szebehely, V. (1967). "Theory of Orbits - The Restricted Problem of Three
Bodies".
"""
import os
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import (TYPE_CHECKING, Callable, Literal, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy as np
import numpy.typing as npt
import pandas as pd

from hiten.algorithms.corrector.newton import _OrbitCorrector
from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import (_compute_monodromy, _compute_stm,
                                            _stability_indices)
from hiten.algorithms.dynamics.utils.energy import (crtbp_energy,
                                                    energy_to_jacobi)
from hiten.algorithms.dynamics.utils.geometry import _find_y_zero_crossing
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.utils.io import (_ensure_dir, _load_periodic_orbit,
                            _load_periodic_orbit_inplace, _save_periodic_orbit)
from hiten.utils.log_config import logger
from hiten.utils.plots import (animate_trajectories, plot_inertial_frame,
                               plot_rotating_frame)

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold


class S(IntEnum): X=0; Y=1; Z=2; VX=3; VY=4; VZ=5


class _CorrectionConfig(NamedTuple):
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
        _Integrator back-end to use when marching the variational equations.
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


class _ContinuationConfig(NamedTuple):
    state: S | None
    amplitude: bool = False
    getter: Callable[["PeriodicOrbit"], float] | None = None
    extra_params: dict | None = None


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
    libration_point : LibrationPoint
        The libration point instance that anchors the family.
    initial_state : Sequence[float] or None, optional
        Initial condition in rotating canonical units
        :math:`[x, y, z, \dot x, \dot y, \dot z]`. When *None* an analytical
        approximation is attempted.

    Attributes
    ----------
    family : str
        Orbit family name (settable property with class-specific defaults).
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
        Output of :pyfunc:`hiten.algorithms.dynamics.rtbp._stability_indices`.

    Notes
    -----
    Instantiating the class does **not** perform any propagation. Users must
    call :pyfunc:`PeriodicOrbit.correct` (or manually set
    :pyattr:`period`) followed by :pyfunc:`PeriodicOrbit.propagate`.
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

    def __str__(self):
        return f"{self.family} orbit around {self._libration_point}."

    def __repr__(self):
        return f"{self.__class__.__name__}(family={self.family}, libration_point={self._libration_point})"

    @property
    def family(self) -> str:
        r"""
        Get the orbit family name.
        
        Returns
        -------
        str
            The orbit family name
        """
        return self._family

    @property
    def libration_point(self) -> LibrationPoint:
        """The libration point instance that anchors the family."""
        return self._libration_point

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
        """Orbit period, set after a successful correction."""
        return self._period

    @period.setter
    def period(self, value: Optional[float]):
        """Set the orbit period and invalidate cached data.

        Setting the period manually allows users (or serialization logic)
        to override the value obtained via differential correction. Any time
        the period changes we must invalidate cached trajectory, time array
        and stability information so they can be recomputed consistently.
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
        return self._system

    @property
    def mu(self) -> float:
        """Mass ratio of the system."""
        return self._mu

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
        
        indices = self._stability_info[0]  # nu values from _stability_indices
        
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
    
    @property
    def monodromy(self) -> np.ndarray:
        r"""
        Compute the monodromy matrix of the orbit.
        
        Returns
        -------
        numpy.ndarray
            The monodromy matrix
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
    def _correction_config(self) -> _CorrectionConfig:
        """Provides the differential correction configuration for this orbit family."""
        pass

    @property
    @abstractmethod
    def _continuation_config(self) -> _ContinuationConfig:
        """Default parameter for family continuation (must be overridden)."""
        raise NotImplementedError

    def _reset(self) -> None:
        r"""
        Reset all computed properties when the initial state is changed.
        Called internally after differential correction or any other operation
        that modifies the initial state.
        """
        self._trajectory = None
        self._times = None
        self._stability_info = None
        self._period = None
        logger.debug("Reset computed orbit properties due to state change")

    @abstractmethod
    def _initial_guess(self, **kwargs):
        pass

    def correct(
            self,
            *,
            tol: float = 1e-10,
            max_attempts: int = 25,
            forward: int = 1,
            max_delta: float | None = None,
            alpha_reduction: float = 0.5,
            min_alpha: float = 1e-4,
            armijo_c: float = 0.02,
            finite_difference: bool = False,
        ) -> tuple[np.ndarray, float]:
        """Differential correction wrapper.

        This method now delegates the heavy lifting to the generic
        :class:`hiten.algorithms.corrector.newton._OrbitCorrector` which
        implements a robust Newton-Armijo scheme.
        """
        return _OrbitCorrector().correct(
            self,
            tol=tol,
            max_attempts=max_attempts,
            forward=forward,
            max_delta=max_delta,
            alpha_reduction=alpha_reduction,
            min_alpha=min_alpha,
            armijo_c=armijo_c,
            finite_difference=finite_difference,
        )

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
            (_stability_indices, eigenvalues, eigenvectors) from the monodromy matrix
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

    def manifold(self, stable: bool = True, direction: Literal["positive", "negative"] = "positive", method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy", order: int = 6) -> "Manifold":
        from hiten.system.manifold import Manifold
        return Manifold(self, stable=stable, direction=direction, method=method, order=order)

    def plot(self, frame: Literal["rotating", "inertial"] = "rotating", dark_mode: bool = True, save: bool = False, filepath: str = f'orbit.svg', **kwargs):
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
        if self._trajectory is None:
            logger.warning("No trajectory to animate. Call propagate() first.")
            return None, None
        
        return animate_trajectories(self._trajectory, self._times, [self._system.primary, self._system.secondary], self._system.distance, **kwargs)

    def to_csv(self, filepath: str, **kwargs):
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
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        _save_periodic_orbit(self, filepath)
        return

    def load_inplace(self, filepath: str, **kwargs) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")

        _load_periodic_orbit_inplace(self, filepath)
        return

    @classmethod
    def load(cls, filepath: str, **kwargs) -> "PeriodicOrbit":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Orbit file not found: {filepath}")

        return _load_periodic_orbit(filepath)

    def __setstate__(self, state):
        """Restore the PeriodicOrbit instance after unpickling.

        The cached dynamical system used for high-performance propagation is
        removed before pickling (it may contain numba objects) and recreated
        lazily on first access after unpickling.
        """
        # Simply update the dictionary - the cached dynamical system will be
        # rebuilt lazily when :pyfunc:`_cr3bp_system` is first invoked.
        self.__dict__.update(state)

    def __getstate__(self):
        """Custom state extractor to enable pickling.

        We strip attributes that might keep references to non-pickleable numba
        objects (e.g. the cached dynamical system) while leaving all the
        essential orbital data untouched.
        """
        state = self.__dict__.copy()
        # Remove the cached CR3BP dynamical system wrapper
        if "_cached_dynsys" in state:
            state["_cached_dynsys"] = None
        return state


class GenericOrbit(PeriodicOrbit):
    r"""
    A minimal concrete orbit class for arbitrary initial conditions, with no correction or special guess logic.
    """
    
    _family = "generic"
    
    def __init__(self, libration_point: LibrationPoint, initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state)
        self._custom_correction_config: Optional[_CorrectionConfig] = None
        self._custom_continuation_config: Optional[_ContinuationConfig] = None
        if self._period is None:
            self._period = np.pi

        self._amplitude = None

    @property
    def correction_config(self) -> Optional[_CorrectionConfig]:
        """
        Get or set the user-defined differential correction configuration.

        This property must be set to a valid :py:class:`_CorrectionConfig`
        instance before calling :py:meth:`correct` on a
        :py:class:`GenericOrbit` object.
        """
        return self._custom_correction_config

    @correction_config.setter
    def correction_config(self, value: Optional[_CorrectionConfig]):
        if value is not None and not isinstance(value, _CorrectionConfig):
            raise TypeError("correction_config must be an instance of _CorrectionConfig or None.")
        self._custom_correction_config = value

    @property
    def eccentricity(self):
        return np.nan

    @property
    def _correction_config(self) -> _CorrectionConfig:
        """
        Provides the differential correction configuration.

        For GenericOrbit, this must be set via the `correction_config` property
        to enable differential correction.
        """
        if self.correction_config is not None:
            return self.correction_config
        raise NotImplementedError(
            "Differential correction is not defined for a GenericOrbit unless the "
            "`correction_config` property is set with a valid _CorrectionConfig."
        )

    @property
    def amplitude(self) -> float:
        """(Read-only) Current amplitude of the orbit."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float):
        self._amplitude = value

    @property
    def continuation_config(self) -> Optional[_ContinuationConfig]:
        """Get or set the continuation parameter for this orbit."""
        return self._custom_continuation_config

    @continuation_config.setter
    def continuation_config(self, cfg: Optional[_ContinuationConfig]):
        if cfg is not None and not isinstance(cfg, _ContinuationConfig):
            raise TypeError("continuation_config must be a _ContinuationConfig instance or None")
        self._custom_continuation_config = cfg

    @property
    def _continuation_config(self) -> _ContinuationConfig:  # used by engines
        if self._custom_continuation_config is None:
            raise NotImplementedError(
                "GenericOrbit requires 'continuation_config' to be set before using continuation engines."
            )
        return self._custom_continuation_config

    def _initial_guess(self, **kwargs):
        if hasattr(self, '_initial_state') and self._initial_state is not None:
            return self._initial_state
        raise ValueError("No initial state provided for GenericOrbit.")
