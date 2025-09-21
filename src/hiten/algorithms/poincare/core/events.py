"""Event classes for Poincare section detection and intersection handling.

This module provides classes for defining Poincare section events and
handling trajectory-section intersections. It defines the interface
for surface event detection and provides concrete implementations
for common section types.
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class _SurfaceEvent(ABC):
    """Abstract base class for Poincare section surface events.

    This abstract base class defines the interface for surface events
    that detect trajectory crossings through Poincare sections. Concrete
    subclasses specify a signed scalar function g(x) whose zeros define
    the hypersurface in phase space that serves as the Poincare section.

    The class stores the direction of admissible crossings to avoid
    duplicating this logic in the generic crossing detection code.

    Parameters
    ----------
    direction : {1, -1, None}, default=None
        Select which zero-crossings are accepted:
        - 1: negative -> positive sign change
        - -1: positive -> negative sign change  
        - None: either direction

    Notes
    -----
    The surface is defined by the equation g(x) = 0, where g(x) is
    a signed scalar function. The direction parameter controls which
    sign changes are considered valid crossings.

    All time units are in nondimensional units unless otherwise
    specified.
    """

    _dir: Literal[1, -1, None]

    def __init__(self, *, direction: Literal[1, -1, None] = None) -> None:
        if direction not in (1, -1, None):
            raise ValueError("direction must be 1, -1 or None")
        self._dir = direction

    @abstractmethod
    def value(self, state: "np.ndarray") -> float:
        """Return the value of the surface function g(state).

        Parameters
        ----------
        state : ndarray, shape (n,)
            State vector at which to evaluate the surface function.

        Returns
        -------
        float
            The value of g(state), where g(x) = 0 defines the section.

        Notes
        -----
        This method must be implemented by concrete subclasses to
        define the specific surface function. The sign of the returned
        value determines which side of the surface the state is on.
        """
        pass

    def is_crossing(self, prev_val: float, curr_val: float) -> bool:
        """Check if a trajectory has crossed the section.

        Parameters
        ----------
        prev_val : float
            Previous value of the surface function.
        curr_val : float
            Current value of the surface function.

        Returns
        -------
        bool
            True if the trajectory has crossed the section in the
            requested direction, False otherwise.

        Notes
        -----
        This method implements the crossing detection logic based on
        the direction parameter. It checks for sign changes that
        indicate a trajectory has passed through the section.
        """
        if self._dir is None:
            res = prev_val * curr_val <= 0.0 and prev_val != curr_val
        elif self._dir == 1:
            res = prev_val < 0.0 <= curr_val
        else:
            res = prev_val > 0.0 >= curr_val
        return res

    @property
    def direction(self) -> Literal[1, -1, None]:
        """Get the crossing direction for this surface event.

        Returns
        -------
        {1, -1, None}
            The crossing direction:
            - 1: negative -> positive sign change
            - -1: positive -> negative sign change
            - None: either direction

        Notes
        -----
        This property returns the direction parameter that was set
        during initialization. It controls which sign changes are
        considered valid crossings.
        """
        return self._dir


class _PlaneEvent(_SurfaceEvent):
    """Concrete surface event representing a hyperplane coord = value.

    This class implements a surface event for hyperplanes defined by
    setting a single coordinate to a constant value. It supports both
    coordinate name-based and index-based specification.

    Parameters
    ----------
    coord : str or int
        Coordinate identifier. A string is resolved via a built-in
        name-to-index map (supports both synodic and center manifold
        names such as 'x', 'q3', 'p2', etc.). An int is interpreted
        directly as an index into the state vector.
    value : float, default=0.0
        Plane offset along the chosen coordinate (nondimensional units).
    direction : {1, -1, None}, optional
        Crossing direction filter, passed to the parent class.
        Controls which sign changes are considered valid crossings.

    Notes
    -----
    The surface is defined by the equation coord - value = 0. The
    coordinate can be specified either by name (using the built-in
    mapping) or by direct index into the state vector.

    Supported coordinate names:
    - Synodic frame: 'x', 'y', 'z', 'vx', 'vy', 'vz'
    - Center manifold: 'q2', 'p2', 'q3', 'p3'

    All time units are in nondimensional units unless otherwise
    specified.
    """

    _IDX_MAP = {"x": 0, "y": 1, "z": 2, "vx": 3, "vy": 4, "vz": 5,
                "q2": 0, "p2": 1, "q3": 2, "p3": 3}

    def __init__(
        self,
        *,
        coord: str | int,
        value: float = 0.0,
        direction: Literal[1, -1, None] = None,
    ) -> None:
        super().__init__(direction=direction)

        if isinstance(coord, str):
            try:
                self._idx = int(self._IDX_MAP[coord.lower()])
                self._name = coord.lower()
            except KeyError as exc:
                raise ValueError(f"Unknown coordinate name '{coord}'.") from exc
        else:
            idx_int = int(coord)
            if idx_int < 0:
                raise ValueError("coord index must be non-negative")
            self._idx = idx_int
            self._name = None

        self._value = float(value)

    def value(self, state: np.ndarray) -> float:
        """Return the value of the plane function coord - value.

        Parameters
        ----------
        state : ndarray, shape (n,)
            State vector at which to evaluate the plane function.

        Returns
        -------
        float
            The value of coord - value, where coord is the specified
            coordinate and value is the plane offset.

        Notes
        -----
        This method implements the surface function for a hyperplane.
        The sign of the returned value indicates which side of the
        plane the state is on.
        """
        return float(state[self._idx] - self._value)    

    @property
    def index(self) -> int:
        """Get the state vector index of the plane coordinate.

        Returns
        -------
        int
            The index into the state vector for the coordinate
            that defines this plane.

        Notes
        -----
        This property returns the resolved index, whether it was
        specified directly or resolved from a coordinate name.
        """
        return self._idx

    @property
    def offset(self) -> float:
        """Get the plane offset value.

        Returns
        -------
        float
            The offset value along the chosen coordinate (nondimensional units).

        Notes
        -----
        This property returns the value that was set during
        initialization. The plane is defined by coord = offset.
        """
        return self._value

    @property
    def name(self) -> str | None:
        """Return the original coordinate name if provided, else None.

        Notes
        -----
        This can be used by fast-path backends to detect simple synodic planes
        like 'x', 'y', or 'z'.
        """
        return self._name
