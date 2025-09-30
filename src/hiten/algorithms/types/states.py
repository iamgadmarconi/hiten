"""Type definitions and state vector containers for the CR3BP.

This module provides comprehensive type definitions and state vector containers
for the circular restricted three-body problem. It includes enumerations for
different coordinate systems and state vector containers with convenient
property access and validation.

Notes
-----
All state vector containers provide both array-like access and property access
for convenience. The containers are mutable and support validation of input
data to ensure consistency with the expected coordinate system.
"""

from enum import Enum, IntEnum
from typing import (TYPE_CHECKING, Iterator, Optional, Sequence, Tuple, Type,
                    Union, overload)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from hiten.algorithms.integrators.types import _Solution


class ReferenceFrame(Enum):
    """
    Reference frame container.

    INERTIAL: non-rotating barycentric (or system-specific) inertial frame. [x, y, z, vx, vy, vz]
    ROTATING: synodic frame rotating with primaries. [x, y, z, vx, vy, vz]
    CENTER_MANIFOLD: center manifold coordinate system. [q1, q2, q3, p1, p2, p3]
    RESTRICTED_CENTER_MANIFOLD: restricted center manifold coordinate system. [q2, p2, q3, p3]
    """
    INERTIAL = "inertial"
    ROTATING = "rotating"
    CENTER_MANIFOLD = "center_manifold"
    RESTRICTED_CENTER_MANIFOLD = "restricted_center_manifold"


class SynodicState(IntEnum):
    """
    Enumeration for synodic frame coordinates.
    
    This enumeration defines the indices for the 6D state vector in the
    rotating synodic frame of the circular restricted three-body problem.
    The coordinates are ordered as position components followed by velocity
    components.
    
    Attributes
    ----------
    X : int
        X position component (index 0)
    Y : int
        Y position component (index 1)
    Z : int
        Z position component (index 2)
    VX : int
        X velocity component (index 3)
    VY : int
        Y velocity component (index 4)
    VZ : int
        Z velocity component (index 5)
        
    Notes
    -----
    The synodic frame rotates with the primary bodies, so the coordinates
    represent position and velocity in the rotating reference frame.
    """
    X=0
    Y=1
    Z=2
    VX=3
    VY=4
    VZ=5

class CenterManifoldState(IntEnum):
    """
    Enumeration for center manifold coordinates.
    
    This enumeration defines the indices for the 6D state vector in the
    center manifold coordinate system. The coordinates are ordered as
    position components (q1, q2, q3) followed by momentum components
    (p1, p2, p3).
    
    Attributes
    ----------
    q1 : int
        First position component (index 0)
    q2 : int
        Second position component (index 1)
    q3 : int
        Third position component (index 2)
    p1 : int
        First momentum component (index 3)
    p2 : int
        Second momentum component (index 4)
    p3 : int
        Third momentum component (index 5)
        
    Notes
    -----
    The center manifold coordinates are canonical coordinates that
    preserve the Hamiltonian structure of the system.
    """
    q1=0
    q2=1
    q3=2
    p1=3
    p2=4
    p3=5

class RestrictedCenterManifoldState(IntEnum):
    """
    Enumeration for restricted center manifold coordinates.
    
    This enumeration defines the indices for the 4D state vector in the
    restricted center manifold coordinate system. The coordinates are
    ordered as position components (q2, q3) followed by momentum components
    (p2, p3).

    Attributes
    ----------
    q2 : int
        Second position component (index 0)
    p2 : int
        Second momentum component (index 1)
    q3 : int
        Third position component (index 2)
    p3 : int
        Third momentum component (index 3)
    """
    q2=0
    p2=1
    q3=2
    p3=3


class RestrictedCenterManifoldState(IntEnum):
    """
    Enumeration for restricted center manifold coordinates.
    
    This enumeration defines the indices for the 4D state vector in the
    restricted center manifold coordinate system. The coordinates are
    ordered as position components (q2, q3) followed by momentum components
    (p2, p3).
    
    Attributes
    ----------
    q2 : int
        Second position component (index 0)
    p2 : int
        Second momentum component (index 1)
    q3 : int
        Third position component (index 2)
    p3 : int
        Third momentum component (index 3)
        
    Notes
    -----
    The restricted center manifold coordinates are a reduced set of
    canonical coordinates that capture the essential dynamics while
    reducing computational complexity.
    """
    q2=0
    p2=1
    q3=2
    p3=3

class _BaseStateContainer:
    """
    Minimal mutable container for a single state vector, indexed by an IntEnum.

    This is the base class for all state vector containers. It provides
    array-like access, property access, and validation for state vectors
    in different coordinate systems.

    Subclasses must set ``_enum`` to the corresponding IntEnum class and can
    optionally expose convenience properties (e.g., ``x``, ``y``, ``vx``).

    Parameters
    ----------
    values : Sequence[float], optional
        Initial values for the state vector. If None, initializes to zeros.
    **kwargs
        Named parameters for individual components (e.g., x=1.0, y=2.0).

    Attributes
    ----------
    _enum : type[IntEnum]
        The enumeration class defining the coordinate indices
    _values : npt.NDArray[np.float64]
        The underlying state vector data

    Notes
    -----
    The container supports both array-like access (e.g., state[0]) and
    property access (e.g., state.x) for convenience. All values are
    stored as 64-bit floating-point numbers.
    """

    _enum: type[IntEnum] = None  # to be set by subclasses

    def __init__(
        self,
        values: Sequence[float] | npt.NDArray[np.float64] | None = None,
        *,
        copy: bool = True,
        frame: Optional[ReferenceFrame] = None,
        **kwargs,
    ):
        if self._enum is None:
            raise TypeError("_BaseStateContainer cannot be instantiated directly; subclass and set _enum")

        size = len(self._enum)

        if values is not None:
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim != 1:
                raise ValueError("values must be a 1D sequence")
            if arr.shape[0] != size:
                raise ValueError(f"values must have length {size}")
            self._values: npt.NDArray[np.float64] = arr.copy() if copy else arr
        else:
            self._values = np.zeros((size,), dtype=np.float64)

        # Reference frame metadata (optional; used mainly for Cartesian states)
        default_frame = getattr(type(self), "DEFAULT_FRAME", None)
        self._frame: Optional[ReferenceFrame] = frame if frame is not None else default_frame

        # Allow initialization via named fields, e.g., x=..., y=...,
        # or using enum member names (case-insensitive). 'frame' is handled above.
        for key, val in kwargs.items():
            self._assign_by_name(key, float(val))

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Return a writable 1D numpy array of the state values."""
        return self._values

    def to_numpy(self, copy: bool = True) -> npt.NDArray[np.float64]:
        """Return the underlying data as numpy array; copy by default."""
        return self._values.copy() if copy else self._values

    def copy(self):
        """Deep copy of the container preserving the concrete subclass type."""
        return self.__class__(self._values.copy())

    @property
    def dim(self) -> int:
        """Return the dimension of the state vector."""
        return self._values.shape[0]

    def as_tuple(self) -> Tuple[float, ...]:
        """Return the state vector as a tuple."""
        return tuple(float(v) for v in self._values.tolist())

    def as_dict(self) -> dict:
        """Return the state vector as a dictionary."""
        members = type(self)._enum 
        return {name.lower(): float(self._values[members[name]]) for name in members.__members__.keys()}

    def _resolve_index(self, key: Union[int, str, IntEnum]) -> int:
        """Resolve the index of a key."""
        if isinstance(key, int):
            return key
        if isinstance(key, IntEnum):
            return int(key)
        if isinstance(key, str):
            members = type(self)._enum
            name = key.upper()
            if name in members.__members__:
                return int(members[name])
        raise TypeError("key must be int, IntEnum member, or a valid field name")

    def _assign_by_name(self, name: str, value: float) -> None:
        """Assign a value to a key by name."""
        idx = self._resolve_index(name)
        self._values[idx] = value

    @overload
    def __getitem__(self, key: int) -> float: ...

    @overload
    def __getitem__(self, key: str) -> float: ...

    @overload
    def __getitem__(self, key: IntEnum) -> float: ...

    def __getitem__(self, key: Union[int, str, IntEnum]) -> float:
        idx = self._resolve_index(key)
        return float(self._values[idx])

    @overload
    def __setitem__(self, key: int, value: float) -> None: ...

    @overload
    def __setitem__(self, key: str, value: float) -> None: ...

    @overload
    def __setitem__(self, key: IntEnum, value: float) -> None: ...

    def __setitem__(self, key: Union[int, str, IntEnum], value: float) -> None:
        idx = self._resolve_index(key)
        self._values[idx] = float(value)

    def __len__(self) -> int:
        return self.dim

    def __iter__(self) -> Iterator[float]:
        for v in self._values:
            yield float(v)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        items = ", ".join(f"{k}={v:.6g}" for k, v in self.as_dict().items())
        frame_str = f", frame={self._frame.value}" if getattr(self, "_frame", None) is not None else ""
        return f"{cls_name}({items}{frame_str})"

    @property
    def frame(self) -> Optional[ReferenceFrame]:
        """Reference frame metadata, if available."""
        return self._frame

    @classmethod
    def from_array_view(cls, array: npt.NDArray[np.float64], *, frame: Optional[ReferenceFrame] = None) -> "_BaseStateContainer":
        """Construct a container that wraps a 1D NumPy array without copying.

        The provided array must be 1D and have length equal to the enum size.
        Mutations through the container will mutate the original array.
        """
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("array must be 1D")
        if arr.shape[0] != len(cls._enum):
            raise ValueError(f"array must have length {len(cls._enum)}")
        return cls(arr, copy=False, frame=frame)


class SynodicStateVector(_BaseStateContainer):
    """
    Container for synodic frame state vectors.
    
    This class provides a convenient container for 6D state vectors in the
    rotating synodic frame of the circular restricted three-body problem.
    It supports both array-like access and property access for all components.
    
    Parameters
    ----------
    values : Sequence[float], optional
        Initial values for the state vector. If None, initializes to zeros.
    **kwargs
        Named parameters for individual components (e.g., x=1.0, y=2.0, vx=0.5).

    Attributes
    ----------
    x : float
        X position component
    y : float
        Y position component
    z : float
        Z position component
    vx : float
        X velocity component
    vy : float
        Y velocity component
    vz : float
        Z velocity component
        
    Notes
    -----
    The synodic frame rotates with the primary bodies, so the coordinates
    represent position and velocity in the rotating reference frame.
    """
    _enum = SynodicState
    DEFAULT_FRAME = ReferenceFrame.ROTATING

    @property
    def x(self) -> float:
        """Return the X position component."""
        return float(self._values[SynodicState.X])

    @x.setter
    def x(self, value: float) -> None:
        """Assign a value to the X position component."""
        self._values[SynodicState.X] = float(value)

    @property
    def y(self) -> float:
        """Return the Y position component."""
        return float(self._values[SynodicState.Y])

    @y.setter
    def y(self, value: float) -> None:
        """Assign a value to the Y position component."""
        self._values[SynodicState.Y] = float(value)

    @property
    def z(self) -> float:
        """Return the Z position component."""
        return float(self._values[SynodicState.Z])

    @z.setter
    def z(self, value: float) -> None:
        """Assign a value to the Z position component."""
        self._values[SynodicState.Z] = float(value)

    @property
    def vx(self) -> float:
        """Return the X velocity component."""
        return float(self._values[SynodicState.VX])

    @vx.setter
    def vx(self, value: float) -> None:
        """Assign a value to the X velocity component."""
        self._values[SynodicState.VX] = float(value)

    @property
    def vy(self) -> float:
        """Return the Y velocity component."""
        return float(self._values[SynodicState.VY])

    @vy.setter
    def vy(self, value: float) -> None:
        """Assign a value to the Y velocity component."""
        self._values[SynodicState.VY] = float(value)

    @property
    def vz(self) -> float:
        """Return the Z velocity component."""
        return float(self._values[SynodicState.VZ])

    @vz.setter
    def vz(self, value: float) -> None:
        """Assign a value to the Z velocity component."""
        self._values[SynodicState.VZ] = float(value)


class CenterManifoldStateVector(_BaseStateContainer):
    """
    Container for center manifold state vectors.
    
    This class provides a convenient container for 6D state vectors in the
    center manifold coordinate system. It supports both array-like access
    and property access for all canonical coordinates.
    
    Parameters
    ----------
    values : Sequence[float], optional
        Initial values for the state vector. If None, initializes to zeros.
    **kwargs
        Named parameters for individual components (e.g., q1=1.0, q2=2.0, p1=0.5).

    Attributes
    ----------
    q1 : float
        First position component
    q2 : float
        Second position component
    q3 : float
        Third position component
    p1 : float
        First momentum component
    p2 : float
        Second momentum component
    p3 : float
        Third momentum component
        
    Notes
    -----
    The center manifold coordinates are canonical coordinates that
    preserve the Hamiltonian structure of the system.
    """
    _enum = CenterManifoldState
    DEFAULT_FRAME = ReferenceFrame.CENTER_MANIFOLD

    @property
    def q1(self) -> float:
        """Return the first position component."""
        return float(self._values[CenterManifoldState.q1])

    @q1.setter
    def q1(self, value: float) -> None:
        """Assign a value to the first position component."""
        self._values[CenterManifoldState.q1] = float(value)

    @property
    def q2(self) -> float:
        """Return the second position component."""
        return float(self._values[CenterManifoldState.q2])

    @q2.setter
    def q2(self, value: float) -> None:
        """Assign a value to the second position component."""
        self._values[CenterManifoldState.q2] = float(value)

    @property
    def q3(self) -> float:
        """Return the third position component."""
        return float(self._values[CenterManifoldState.q3])

    @q3.setter
    def q3(self, value: float) -> None:
        """Assign a value to the third position component."""
        self._values[CenterManifoldState.q3] = float(value)

    @property
    def p1(self) -> float:
        """Return the first momentum component."""
        return float(self._values[CenterManifoldState.p1])

    @p1.setter
    def p1(self, value: float) -> None:
        """Assign a value to the first momentum component."""
        self._values[CenterManifoldState.p1] = float(value)

    @property
    def p2(self) -> float:
        """Return the second momentum component."""
        return float(self._values[CenterManifoldState.p2])

    @p2.setter
    def p2(self, value: float) -> None:
        """Assign a value to the second momentum component."""
        self._values[CenterManifoldState.p2] = float(value)

    @property
    def p3(self) -> float:
        """Return the third momentum component."""
        return float(self._values[CenterManifoldState.p3])

    @p3.setter
    def p3(self, value: float) -> None:
        """Assign a value to the third momentum component."""
        self._values[CenterManifoldState.p3] = float(value)


class RestrictedCenterManifoldStateVector(_BaseStateContainer):
    """
    Container for restricted center manifold state vectors.
    
    This class provides a convenient container for 4D state vectors in the
    restricted center manifold coordinate system. It supports both array-like
    access and property access for the reduced set of canonical coordinates.
    
    Parameters
    ----------
    values : Sequence[float], optional
        Initial values for the state vector. If None, initializes to zeros.
    **kwargs
        Named parameters for individual components (e.g., q2=1.0, q3=2.0, p2=0.5).

    Attributes
    ----------
    q2 : float
        Second position component
    p2 : float
        Second momentum component
    q3 : float
        Third position component
    p3 : float
        Third momentum component
        
    Notes
    -----
    The restricted center manifold coordinates are a reduced set of
    canonical coordinates that capture the essential dynamics while
    reducing computational complexity.
    """
    _enum = RestrictedCenterManifoldState
    DEFAULT_FRAME = ReferenceFrame.RESTRICTED_CENTER_MANIFOLD

    @property
    def q2(self) -> float:
        """Return the second position component."""
        return float(self._values[RestrictedCenterManifoldState.q2])

    @q2.setter
    def q2(self, value: float) -> None:
        """Assign a value to the second position component."""
        self._values[RestrictedCenterManifoldState.q2] = float(value)

    @property
    def p2(self) -> float:
        """Return the second momentum component."""
        return float(self._values[RestrictedCenterManifoldState.p2])

    @p2.setter
    def p2(self, value: float) -> None:
        """Assign a value to the second momentum component."""
        self._values[RestrictedCenterManifoldState.p2] = float(value)

    @property
    def q3(self) -> float:
        """Return the third position component."""
        return float(self._values[RestrictedCenterManifoldState.q3])

    @q3.setter
    def q3(self, value: float) -> None:
        """Assign a value to the third position component."""
        self._values[RestrictedCenterManifoldState.q3] = float(value)

    @property
    def p3(self) -> float:
        """Return the third momentum component."""
        return float(self._values[RestrictedCenterManifoldState.p3])

    @p3.setter
    def p3(self, value: float) -> None:
        """Assign a value to the third momentum component."""
        self._values[RestrictedCenterManifoldState.p3] = float(value)

class Trajectory:
    """
    Lightweight container for trajectory data: a time array and matching state vectors.

    This class provides a convenient container for storing trajectory data
    with time and state arrays. It supports array-like access, slicing,
    and iteration over time-state pairs.

    Parameters
    ----------
    times : Sequence[float]
        Monotonically ordered time grid of length N.
    states : Sequence[Sequence[float]]
        State matrix of shape (N, D) with D-dimensional states corresponding to each time.

    Attributes
    ----------
    times : npt.NDArray[np.float64]
        Time array of length N
    states : npt.NDArray[np.float64]
        State matrix of shape (N, D)
    n_samples : int
        Number of stored samples N
    dim : int
        State-space dimension D
    t0 : float
        Initial time
    tf : float
        Final time
    duration : float
        Total elapsed time tf - t0

    Notes
    -----
    - The container is read-only by convention; create a new instance for slices or transformations.
    - Time values may be strictly increasing or strictly decreasing, but must be strictly monotonic.
    - All data is stored as 64-bit floating-point numbers for consistency.
    """

    def __init__(
        self,
        times: Sequence[float],
        states: Sequence[Sequence[float]],
        state_vector_cls: Optional[Type[_BaseStateContainer]] = None,
        frame: Optional[ReferenceFrame] = None,
    ):
        t_arr = np.asarray(times, dtype=np.float64)
        x_arr = np.asarray(states, dtype=np.float64)

        if t_arr.ndim != 1:
            raise ValueError("times must be a 1D sequence")
        if x_arr.ndim != 2:
            raise ValueError("states must be a 2D array-like of shape (N, D)")
        if len(t_arr) != len(x_arr):
            raise ValueError(
                f"times and states must have the same length: {len(t_arr)} != {len(x_arr)}"
            )
        if len(t_arr) < 2:
            raise ValueError("trajectory must contain at least two samples")

        dt = np.diff(t_arr)
        if not (np.all(dt > 0.0) or np.all(dt < 0.0)):
            raise ValueError("times must be strictly monotonic (all increasing or all decreasing)")

        self._times: npt.NDArray[np.float64] = t_arr
        self._states: npt.NDArray[np.float64] = x_arr
        self._state_vector_cls: Optional[Type[_BaseStateContainer]] = state_vector_cls
        self._frame: Optional[ReferenceFrame] = frame

    @property
    def times(self) -> npt.NDArray[np.float64]:
        """Return the time array."""
        return self._times

    @property
    def states(self) -> npt.NDArray[np.float64]:
        """Return the state array."""
        return self._states

    @property
    def n_samples(self) -> int:
        """Number of stored samples N."""
        return self._times.shape[0]

    @property
    def dim(self) -> int:
        """State-space dimension D."""
        return self._states.shape[1]

    @property
    def t0(self) -> float:
        """Initial time."""
        return float(self._times[0])

    @property
    def tf(self) -> float:
        """Final time."""
        return float(self._times[-1])

    @property
    def duration(self) -> float:
        """Total elapsed time tf - t0 (can be negative if times are decreasing)."""
        return float(self._times[-1] - self._times[0])

    def as_arrays(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the underlying arrays as a tuple (times, states)."""
        return self._times, self._states

    @property
    def state_vector_cls(self) -> Optional[Type[_BaseStateContainer]]:
        """The container class used for row views, if any (e.g., SynodicStateVector)."""
        return self._state_vector_cls

    @property
    def index_enum(self) -> Optional[type[IntEnum]]:
        """Return the IntEnum representing the state indices, if a container class is bound."""
        if self._state_vector_cls is None:
            return None
        return self._state_vector_cls._enum

    @property
    def frame(self) -> Optional[ReferenceFrame]:
        """Reference frame for the trajectory's states (e.g., rotating vs inertial)."""
        return self._frame

    def with_state_vector(self, cls: Type[_BaseStateContainer]) -> "Trajectory":
        """Return a new trajectory referencing the same arrays with a state vector class bound."""
        new_traj = Trajectory(self._times, self._states, state_vector_cls=cls, frame=self._frame)
        return new_traj

    def with_frame(self, frame: ReferenceFrame) -> "Trajectory":
        """Return a new trajectory referencing the same arrays with a reference frame set."""
        return Trajectory(self._times, self._states, state_vector_cls=self._state_vector_cls, frame=frame)

    def vector_at(self, index: int, *, copy: bool = False) -> _BaseStateContainer:
        """Return the state at index wrapped as a state-vector container.

        If copy=False, the wrapper references the underlying row view.
        """
        if self._state_vector_cls is None:
            raise ValueError("No state_vector_cls configured. Use with_state_vector(...) first.")
        row = self._states[index]
        return self._state_vector_cls(row, copy=copy, frame=self._frame)

    def iter_vectors(self, *, copy: bool = False) -> Iterator[_BaseStateContainer]:
        """Iterate over state rows yielding state-vector containers.

        If copy=False, each yielded container wraps a view into the underlying array.
        """
        if self._state_vector_cls is None:
            raise ValueError("No state_vector_cls configured. Use with_state_vector(...) first.")
        for i in range(self.n_samples):
            yield self._state_vector_cls(self._states[i], copy=copy, frame=self._frame)

    def __len__(self) -> int:
        return self.n_samples

    def __iter__(self) -> Iterator[Tuple[float, npt.NDArray[np.float64]]]:
        for i in range(self.n_samples):
            yield float(self._times[i]), self._states[i]

    @overload
    def __getitem__(self, key: int) -> Tuple[float, npt.NDArray[np.float64]]: ...

    @overload
    def __getitem__(self, key: slice) -> "Trajectory": ...

    def __getitem__(self, key: Union[int, slice]):
        if isinstance(key, int):
            return float(self._times[key]), self._states[key]
        if isinstance(key, slice):
            return Trajectory(self._times[key], self._states[key], state_vector_cls=self._state_vector_cls, frame=self._frame)

    def __repr__(self) -> str:
        direction = "increasing" if self._times[1] > self._times[0] else "decreasing"
        return (
            f"Trajectory(N={self.n_samples}, D={self.dim}, "
            f"t0={self.t0:.6g}, tf={self.tf:.6g}, {direction})"
        )

    @classmethod
    def from_arrays(
        cls,
        times: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        *,
        state_vector_cls: Optional[Type[_BaseStateContainer]] = None,
        frame: Optional[ReferenceFrame] = None,
    ) -> "Trajectory":
        """Construct from numpy arrays (validated)."""
        return cls(times, states, state_vector_cls=state_vector_cls, frame=frame)

    @classmethod
    def from_solution(
        cls,
        solution: "_Solution",
        *,
        state_vector_cls: Optional[Type[_BaseStateContainer]] = None,
        frame: Optional[ReferenceFrame] = None,
    ) -> "Trajectory":
        """Construct from an integrator _Solution without copying arrays."""
        return cls(solution.times, solution.states, state_vector_cls=state_vector_cls, frame=frame)

    def reversed(self) -> "Trajectory":
        """Return a new trajectory with reversed time order."""
        return Trajectory(self._times[::-1].copy(), self._states[::-1].copy())

    def to_arrays(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return the underlying arrays as a tuple (times, states)."""
        return self._times, self._states