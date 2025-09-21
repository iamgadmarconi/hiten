"""Synodic (rotating-frame) section events.

This module defines an affine hyperplane event usable as a Poincare
surface in the synodic CRTBP frame. It generalizes axis-aligned planes
by allowing arbitrary normals in the 6D state space, providing
flexibility for defining complex section geometries.

The main class :class:`~hiten.algorithms.poincare.synodic.events._AffinePlaneEvent` 
extends the base surface event to provide specialized functionality for synodic 
Poincare sections, including support for arbitrary hyperplane orientations 
and convenient axis-aligned plane construction.

"""

from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.events import _PlaneEvent, _SurfaceEvent


class _AffinePlaneEvent(_SurfaceEvent):
    """Affine hyperplane event in the synodic frame.

    This class defines a Poincare section surface as an affine hyperplane
    in the 6D state space of the circular restricted three-body problem.
    It extends the base surface event to provide specialized functionality
    for synodic Poincare sections with arbitrary hyperplane orientations.

    The section is defined by the zero level-set of:

        g(state) = n * state - c = 0

    where state = (x, y, z, vx, vy, vz) and n is a 6D normal vector.

    Parameters
    ----------
    normal : array_like, shape (6,)
        Hyperplane normal in synodic coordinates (nondimensional units).
        The vector is used as-is (no normalization) so its scale must be
        consistent with the offset parameter.
    offset : float, default 0.0
        Hyperplane offset along the normal (nondimensional units).
        The section is defined by n * state = offset.
    direction : {1, -1, None}, optional
        Crossing direction filter passed to the base surface event.
        If None, no direction filtering is applied.

    Notes
    -----
    This class provides a flexible way to define Poincare sections in
    the synodic frame. Unlike axis-aligned planes, it allows arbitrary
    orientations in the 6D state space, enabling complex section geometries.

    The class performs validation to ensure the normal vector has
    the correct dimensions and contains only finite values.

    All geometric parameters are in nondimensional units unless
    otherwise specified.
    """

    def __init__(
        self,
        *,
        normal: Sequence[float] | np.ndarray,
        offset: float = 0.0,
        direction: Literal[1, -1, None] = None,
    ) -> None:
        super().__init__(direction=direction)

        n_arr = np.asarray(normal, dtype=float)
        if n_arr.ndim != 1 or n_arr.size != 6:
            raise ValueError("normal must be a 1-D array of length 6")
        if not np.all(np.isfinite(n_arr)):
            raise ValueError("normal must contain only finite values")

        self._n = n_arr
        self._c = float(offset)

    def value(self, state: "np.ndarray") -> float:
        """Compute the surface function value for a given state.

        Parameters
        ----------
        state : ndarray, shape (6,)
            State vector in synodic coordinates (nondimensional units).

        Returns
        -------
        float
            The surface function value g(state) = n * state - c.

        Notes
        -----
        This method computes the surface function value for the given
        state vector. The function returns zero when the state is on
        the hyperplane, positive when on one side, and negative when
        on the other side.

        The computation uses the dot product of the normal vector
        with the state vector, minus the offset.
        """
        return float(self._n @ state - self._c)

    @property
    def normal(self) -> np.ndarray:
        """Get the hyperplane normal vector.

        Returns
        -------
        ndarray, shape (6,)
            The hyperplane normal vector in synodic coordinates.
        """
        return self._n

    @property
    def offset(self) -> float:
        """Get the hyperplane offset value.

        Returns
        -------
        float
            The hyperplane offset value (nondimensional units).
        """
        return self._c

    @classmethod
    def axis_plane(
        cls,
        coord: str | int,
        *,
        c: float = 0.0,
        direction: Literal[1, -1, None] = None,
    ) -> "_AffinePlaneEvent":
        """Convenience constructor for axis-aligned planes.

        Parameters
        ----------
        coord : str or int
            Coordinate name or index for the plane normal.
            String names: "x", "y", "z", "vx", "vy", "vz"
            Integer indices: 0-5 corresponding to the coordinates above.
        c : float, default 0.0
            Plane offset value (nondimensional units).
        direction : {1, -1, None}, optional
            Crossing direction filter for the surface event.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.events._AffinePlaneEvent`
            An affine plane event with axis-aligned normal.

        Raises
        ------
        ValueError
            If coordinate name is unknown or index is out of range.

        Examples
        --------
        Create a plane at x = 1 - mu::

        >>> plane = hiten.algorithms.poincare.synodic.events._AffinePlaneEvent.axis_plane("x", c=1-mu)

        Create a plane at y = 0::

        >>> plane = hiten.algorithms.poincare.synodic.events._AffinePlaneEvent.axis_plane(1, c=0.0)

        Notes
        -----
        This class method provides a convenient way to create axis-aligned
        planes without manually constructing the normal vector. It creates
        a normal vector with a 1 in the specified coordinate position
        and 0 elsewhere.

        All geometric parameters are in nondimensional units.
        """
        if isinstance(coord, str):
            try:
                idx = int(_PlaneEvent._IDX_MAP[coord.lower()])
            except KeyError as exc:
                raise ValueError(f"Unknown coordinate name '{coord}'.") from exc
        else:
            idx = int(coord)
            if idx < 0 or idx > 5:
                raise ValueError("coord index must be between 0 and 5")

        n = np.zeros(6, dtype=float)
        n[idx] = 1.0
        return cls(normal=n, offset=c, direction=direction)

    def __repr__(self) -> str:
        return (f"AffinePlaneEvent(n={np.array2string(self._n, precision=3)}, "
                f"c={self._c:.6g}, dir={self.direction})")



