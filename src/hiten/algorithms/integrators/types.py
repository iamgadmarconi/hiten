from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class EventResult:
    """Result container for a single event detection.

    Attributes
    ----------
    hit : bool
        True if an event crossing was detected and refined.
    t_event : float | None
        Detected event time (None when hit is False).
    y_event : numpy.ndarray | None
        Detected event state (None when hit is False).
    g_event : float | None
        Event function value at the detected time/state (None when hit is False).
    """

    hit: bool
    t_event: Optional[float]
    y_event: Optional[np.ndarray]
    g_event: Optional[float]


@dataclass
class _Solution:
    """Store a discrete solution returned by an integrator.

    Parameters
    ----------
    times : numpy.ndarray, shape (n,)
        Monotonically ordered time grid.
    states : numpy.ndarray, shape (n, d)
        State vectors corresponding to *times*.
    derivatives : numpy.ndarray or None, optional, shape (n, d)
        Evaluations of f(t,y) at the stored nodes. When
        available a cubic Hermite interpolant is employed by
        :func:`~hiten.algorithms.integrators.base._Solution.interpolate`; otherwise linear interpolation is used.

    Raises
    ------
    ValueError
        If the lengths of *times*, *states*, or *derivatives* (when provided)
        are inconsistent.

    Notes
    -----
    The class is a `dataclasses.dataclass` and behaves like an
    immutable record.
    """
    times: np.ndarray
    states: np.ndarray
    derivatives: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate the consistency of times, states, and derivatives arrays"""
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
        """Evaluate the trajectory at intermediate time points.

        If :attr:`~hiten.algorithms.integrators.base._Solution.derivatives` 
        are provided a cubic Hermite scheme of order three is employed on every step; 
        otherwise straight linear interpolation is used.

        Parameters
        ----------
        t : float or array_like
            Query time or array of times contained in
            [times[0], times[-1]].

        Returns
        -------
        numpy.ndarray
            Interpolated state with shape (d,) when *t* is scalar or
            (m, d) when *t* comprises m points.

        Raises
        ------
        ValueError
            If any entry of *t* lies outside the stored integration interval.

        Examples
        --------
        >>> sol = integrator.integrate(sys, y0, np.linspace(0, 10, 11))
        >>> y_mid = sol.interpolate(5.5)
        """
        t_arr = np.atleast_1d(t).astype(float)

        # Support both ascending and descending time grids
        ascending = self.times[0] <= self.times[-1]
        t_min = self.times[0] if ascending else self.times[-1]
        t_max = self.times[-1] if ascending else self.times[0]

        if np.any(t_arr < t_min) or np.any(t_arr > t_max):
            raise ValueError("Interpolation times must lie within the solution interval.")

        # Pre-allocate output array.
        n_dim = self.states.shape[1]
        y_out = np.empty((t_arr.size, n_dim), dtype=self.states.dtype)

        # For each query time, locate the bracketing interval.
        if ascending:
            idxs = np.searchsorted(self.times, t_arr, side="right") - 1
        else:
            # searchsorted assumes ascending sequences; invert sign to reuse it
            idxs = np.searchsorted(-self.times, -t_arr, side="right") - 1
        idxs = np.clip(idxs, 0, len(self.times) - 2)

        t0 = self.times[idxs]
        t1 = self.times[idxs + 1]
        y0 = self.states[idxs]
        y1 = self.states[idxs + 1]

        h = (t1 - t0)
        s = (t_arr - t0) / h  # Normalised position in interval, 0 <= s <= 1

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