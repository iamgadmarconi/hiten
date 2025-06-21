from abc import ABC, abstractmethod
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class DynamicalSystemProtocol(Protocol):
    """
    Protocol defining the interface for dynamical systems.
    
    This protocol specifies the minimum interface that any dynamical system
    must implement to be compatible with the integrator framework.
    """
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        ...
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        ...
            

class _DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.
    
    This class provides common functionality for all dynamical systems
    while requiring subclasses to implement the specific dynamics.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the dynamical system.
        
        Parameters
        ----------
        dim : int
            Dimension of the state space
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self._dim
    
    @property
    @abstractmethod
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        pass
    
    def validate_state(self, y: np.ndarray) -> None:
        """
        Validate that a state vector has the correct dimension.
        
        Parameters
        ----------
        y : numpy.ndarray
            State vector to validate
            
        Raises
        ------
        ValueError
            If the state vector has incorrect dimension
        """
        if len(y) != self.dim:
            raise ValueError(f"State vector dimension {len(y)} != system dimension {self.dim}")


class _DirectedSystem(_DynamicalSystem):
    def __init__(self,
                 base_or_dim: "_DynamicalSystem | int",
                 fwd: int = 1,
                 flip_indices: "slice | Sequence[int] | None" = None):

        if isinstance(base_or_dim, _DynamicalSystem):
            self._base: "_DynamicalSystem | None" = base_or_dim
            dim = base_or_dim.dim
        else:
            self._base = None
            dim = int(base_or_dim)

        super().__init__(dim=dim)

        self._fwd: int = 1 if fwd >= 0 else -1
        self._flip_idx = flip_indices

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:

        if self._base is None:
            raise AttributeError("`rhs` not implemented: subclass must provide "
                                 "its own implementation when no base system "
                                 "is wrapped.")

        base_rhs = self._base.rhs
        flip_idx = self._flip_idx

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = base_rhs(t, y)

            if self._fwd == -1:
                if flip_idx is None:
                    dy = -dy
                else:
                    dy = dy.copy()
                    dy[flip_idx] *= -1
            return dy

        return _rhs

    def __repr__(self):
        return (f"DirectedSystem(dim={self.dim}, fwd={self._fwd}, "
                f"flip_idx={self._flip_idx})")

    def __getattr__(self, item):
        if self._base is None:
            raise AttributeError(item)
        return getattr(self._base, item)