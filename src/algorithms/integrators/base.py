from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from algorithms.dynamics.base import DynamicalSystemProtocol


@dataclass
class Solution:
    """
    Container for integration results.
    
    Attributes
    ----------
    times : numpy.ndarray
        Array of time points, shape (n_points,)
    states : numpy.ndarray
        Array of state vectors, shape (n_points, n_dim)
    derivatives : numpy.ndarray or None, optional
        Array of time derivatives f(t, y) evaluated at the stored time points,
        shape (n_points, n_dim).  When provided, cubic Hermite interpolation is
        used; otherwise interpolation falls back to linear.
    """
    times: np.ndarray
    states: np.ndarray
    derivatives: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if len(self.times) != len(self.states):
            raise ValueError(
                f"Times and states must have same length: "
                f"{len(self.times)} != {len(self.states)}"
            )
        if self.derivatives is not None and len(self.derivatives) != len(self.times):
            raise ValueError(
                "If provided, derivatives must have the same length as times "
                f"({len(self.derivatives)} != {len(self.times)})"
            )

    def interpolate(self, t: Union[np.ndarray, float]) -> np.ndarray:
        """Evaluate the solution at arbitrary time points by interpolation.

        If *derivatives* are available, a cubic Hermite interpolant is used on
        every interval; otherwise linear interpolation is applied.

        Parameters
        ----------
        t : float or array_like
            Time (or array of times) at which to evaluate the solution.  Must
            lie within the integration interval ``[times[0], times[-1]]``.

        Returns
        -------
        ndarray
            Interpolated state(s) with shape ``(n_dim,)`` for a scalar *t* or
            ``(n_times, n_dim)`` for an array input.
        """
        t_arr = np.atleast_1d(t).astype(float)

        if np.any(t_arr < self.times[0]) or np.any(t_arr > self.times[-1]):
            raise ValueError("Interpolation times must lie within the solution interval.")

        # Pre-allocate output array.
        n_dim = self.states.shape[1]
        y_out = np.empty((t_arr.size, n_dim), dtype=self.states.dtype)

        # For each query time, locate the bracketing interval.
        idxs = np.searchsorted(self.times, t_arr, side="right") - 1
        idxs = np.clip(idxs, 0, len(self.times) - 2)

        t0 = self.times[idxs]
        t1 = self.times[idxs + 1]
        y0 = self.states[idxs]
        y1 = self.states[idxs + 1]

        h = (t1 - t0)
        s = (t_arr - t0) / h  # Normalised position in interval, 0 ≤ s ≤ 1

        if self.derivatives is None:
            # Linear interpolation.
            y_out[:] = y0 + ((y1 - y0).T * s).T
        else:
            f0 = self.derivatives[idxs]
            f1 = self.derivatives[idxs + 1]

            s2 = s * s
            s3 = s2 * s
            h00 = 2 * s3 - 3 * s2 + 1
            h10 = s3 - 2 * s2 + s
            h01 = -2 * s3 + 3 * s2
            h11 = s3 - s2

            # Broadcast the Hermite basis functions to match state dimensions.
            y_out[:] = (
                (h00[:, None] * y0) +
                (h10[:, None] * (h[:, None] * f0)) +
                (h01[:, None] * y1) +
                (h11[:, None] * (h[:, None] * f1))
            )

        # Return scalar shape if scalar input.
        if np.isscalar(t):
            return y_out[0]
        return y_out


class Integrator(ABC):
    """
    Abstract base class for numerical integrators.
    
    This class defines the common interface that all integrators must implement.
    Concrete integrators should inherit from this class and implement the
    abstract methods.
    
    Parameters
    ----------
    name : str
        Human-readable name of the integrator
    **options
        Integrator-specific options (stored in self.options)
    """
    
    def __init__(self, name: str, **options):
        self.name = name
        self.options = options
    
    @property
    @abstractmethod
    def order(self) -> Optional[int]:
        """
        Order of accuracy of the integrator.
        
        Returns
        -------
        int or None
            Order of the method, or None if not applicable
        """
        pass
    
    @abstractmethod
    def integrate(
        self,
        system: DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray,
        **kwargs
    ) -> Solution:
        """
        Integrate the dynamical system from initial conditions.
        
        Parameters
        ----------
        system : DynamicalSystemProtocol
            The dynamical system to integrate
        y0 : numpy.ndarray
            Initial state vector, shape (system.dim,)
        t_vals : numpy.ndarray
            Array of time points at which to evaluate the solution
        **kwargs
            Additional integration options
            
        Returns
        -------
        Solution
            Integration results containing times and states
            
        Raises
        ------
        ValueError
            If the system is incompatible with this integrator
        """
        pass
    
    def validate_system(self, system: DynamicalSystemProtocol) -> None:
        """
        Validate that the system is compatible with this integrator.
        
        Parameters
        ----------
        system : DynamicalSystemProtocol
            The system to validate
            
        Raises
        ------
        ValueError
            If the system is incompatible with this integrator
        """
        if not hasattr(system, 'rhs'):
            raise ValueError(f"System must implement 'rhs' method for {self.name}")
    
    def validate_inputs(
        self,
        system: DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray
    ) -> None:
        """
        Validate integration inputs.
        
        Parameters
        ----------
        system : DynamicalSystemProtocol
            The dynamical system to integrate
        y0 : numpy.ndarray
            Initial state vector
        t_vals : numpy.ndarray
            Array of time points
            
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        self.validate_system(system)
        
        if len(y0) != system.dim:
            raise ValueError(
                f"Initial state dimension {len(y0)} != system dimension {system.dim}"
            )
        
        if len(t_vals) < 2:
            raise ValueError("Must provide at least 2 time points")
        
        # Check that time values are monotonic (either strictly increasing or decreasing)
        dt = np.diff(t_vals)
        if not (np.all(dt > 0) or np.all(dt < 0)):
            raise ValueError("Time values must be strictly monotonic (either increasing or decreasing)")
