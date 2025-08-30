"""
Synodic (rotating-frame) section events.

This module defines an affine hyperplane event usable as a Poincaré
surface in the synodic CRTBP frame.  It generalises axis-aligned planes
by allowing arbitrary normals in the 6-D state space.
"""

from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.poincare.core.events import _SurfaceEvent, _PlaneEvent


class _AffinePlaneEvent(_SurfaceEvent):
    """Affine hyperplane event in the synodic frame.

    The section is defined by the zero level-set of::

        g(state) = n · state - c

    where ``state = (x, y, z, vx, vy, vz)`` and ``n`` is a 6-D normal vector.

    Parameters
    ----------
    normal : array_like, shape (6,)
        Hyperplane normal in synodic coordinates. The vector is used as-is
        (no normalisation) so its scale must be consistent with ``offset``.
    offset : float, default 0.0
        Hyperplane offset along the normal; the section is ``n · x = offset``.
    direction : 1 | -1 | None, optional
        Crossing direction filter passed to :class:`_SurfaceEvent`.
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
        return float(self._n @ state - self._c)

    @property
    def normal(self) -> np.ndarray:
        return self._n

    @property
    def offset(self) -> float:
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

        Examples
        --------
        ``x = 1 - mu``: ``AffinePlaneEvent.axis_plane("x", c=1-mu)``
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



