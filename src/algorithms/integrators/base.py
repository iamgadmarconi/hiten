from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

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
    """
    times: np.ndarray
    states: np.ndarray
    
    def __post_init__(self):
        if len(self.times) != len(self.states):
            raise ValueError(
                f"Times and states must have same length: "
                f"{len(self.times)} != {len(self.states)}"
            )


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
