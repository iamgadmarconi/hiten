from abc import ABC, abstractmethod
from typing import Callable, Protocol, runtime_checkable

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
            

class DynamicalSystem(ABC):
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
