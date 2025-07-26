from abc import ABC, abstractmethod
from typing import Literal, NamedTuple

import numpy as np


class _SurfaceEvent(ABC):
    r"""
    Abstract definition of a surface of section event.

    Concrete subclasses specify a *signed* scalar function :math:`g(\mathbf x)`
    whose zeros define the 5-D hypersurface in phase-space that we treat as the
    Poincaré section.  The class also stores the *direction* of admissible
    crossings so the generic crossing logic does not have to duplicate this
    bookkeeping.

    Parameters
    ----------
    direction : 1, -1, None, default None
        Select which zero-crossings are accepted:

        * ``1`` - *negative -> positive* sign change;
        * ``-1`` - *positive -> negative* sign change;
        * ``None`` - either direction.
    """

    _dir: Literal[1, -1, None]

    def __init__(self, *, direction: Literal[1, -1, None] = None) -> None:
        if direction not in (1, -1, None):
            raise ValueError("direction must be 1, -1 or None")
        self._dir = direction

    @abstractmethod
    def value(self, state: "np.ndarray") -> float:
        """Return *g(state)* whose root specifies the section."""

    def is_crossing(self, prev_val: float, curr_val: float) -> bool:
        """Return *True* if (prev_val, curr_val) brackets a root in the requested direction."""
        if self._dir is None:
            res = prev_val * curr_val <= 0.0 and prev_val != curr_val
        elif self._dir == 1:
            res = prev_val < 0.0 <= curr_val
        else:
            res = prev_val > 0.0 >= curr_val
        return res

    @property
    def direction(self) -> Literal[1, -1, None]:
        return self._dir


class _SectionHit(NamedTuple):
    r"""
    Container for a single trajectory-section intersection.

    Attributes
    ----------
    time
        Absolute integration time (signed according to propagation direction).
    state
        n-D state vector at the crossing (immutable copy).
    point2d
        2-D coordinates of the point in the section plane (e.g. (q2, p2) or
        (x, \dot x)).  Stored separately so callers do not have to re-project
        the 6-vector every time.
    """

    time: float
    state: np.ndarray  # shape (n,)
    point2d: np.ndarray  # shape (2,)

    def __repr__(self):
        return (f"SectionHit(t={self.time:.3e}, state={np.array2string(self.state, precision=3)}, "
                f"pt={np.array2string(self.point2d, precision=3)})")


class _PlaneEvent(_SurfaceEvent):
    """Concrete surface event representing a hyper-plane *coord = value*.

    Parameters
    ----------
    coord : str | int
        Coordinate identifier.  A string is resolved via a built-in name → index
        map (supports both synodic and CM names such as ``'x'``, ``'q3'``); an
        *int* is interpreted directly as an index into the state vector.
    value : float, default 0.0
        Plane offset along the chosen coordinate.
    direction : 1 | -1 | None, optional
        Passed to :class:`_SurfaceEvent` - controls admissible crossing
        direction.
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
            except KeyError as exc:
                raise ValueError(f"Unknown coordinate name '{coord}'.") from exc
        else:
            idx_int = int(coord)
            if idx_int < 0:
                raise ValueError("coord index must be non-negative")
            self._idx = idx_int

        self._value = float(value)

    def value(self, state: np.ndarray) -> float:
        return float(state[self._idx] - self._value)    

    @property
    def index(self) -> int:
        """State vector index of the plane coordinate."""
        return self._idx

    @property
    def offset(self) -> float:
        return self._value
