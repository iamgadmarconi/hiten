from typing import Callable, List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class _DynamicalSystemProtocol(Protocol):
    r"""Define the protocol for the minimal interface for dynamical systems.

    This protocol specifies the required attributes that any dynamical system
    must implement to be compatible with the integrator framework. It uses
    structural typing to allow duck typing while maintaining type safety.

    Attributes
    ----------
    dim : int
        Dimension of the state space (number of state variables).
    rhs : Callable[[float, ndarray], ndarray]
        Right-hand side function f(t, y) that computes the time derivative
        dy/dt given time t and state vector y.
        
    Notes
    -----
    The @runtime_checkable decorator allows isinstance() checks against
    this protocol at runtime, enabling flexible type validation.
    """
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        ...
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Right-hand side function for ODE integration."""
        ...
    
    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """Return a plain Python function implementing f(t, y).
        
        Notes
        -----
        Concrete implementations deriving from :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`
        must provide this method. The base class compiles and caches the returned
        function to expose the standardized, compiled :attr:`rhs`.
        """
        ...


@runtime_checkable
class _HamiltonianSystemProtocol(_DynamicalSystemProtocol, Protocol):
    r"""Define the protocol for the interface for Hamiltonian dynamical systems.
    
    Extends the base dynamical system protocol with Hamiltonian-specific
    methods required by symplectic integrators. Provides access to partial
    derivatives and polynomial representation of the Hamiltonian.
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol` : Base protocol
    :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem` : Concrete implementation
    """
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom.
        
        Returns
        -------
        int
            Degrees of freedom count. Total state dimension is 2 * n_dof.
        """
        ...
    
    def dH_dQ(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        r"""Compute partial derivatives of Hamiltonian with respect to positions.
        
        Parameters
        ----------
        Q : ndarray, shape (n_dof,)
            Position coordinates.
        P : ndarray, shape (n_dof,)
            Momentum coordinates.
            
        Returns
        -------
        ndarray, shape (n_dof,)
            Partial derivatives dH/dQ.
        """
        ...
    
    def dH_dP(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        r"""Compute partial derivatives of Hamiltonian with respect to momenta.
        
        Parameters
        ----------
        Q : ndarray, shape (n_dof,)
            Position coordinates.
        P : ndarray, shape (n_dof,)
            Momentum coordinates.
            
        Returns
        -------
        ndarray, shape (n_dof,)
            Partial derivatives dH/dP.
        """
        ...

    def poly_H(self) -> List[List[np.ndarray]]:
        r"""Return polynomial representation of the Hamiltonian.
        
        Returns
        -------
        List[List[ndarray]]
            Nested list structure containing polynomial coefficients
            organized by degree and variable.
        """
        ...

    @property
    def rhs_params(self) -> tuple:
        r"""Return low-level RHS parameters for parametric kernels.

        Returns
        -------
        tuple
            A 3-tuple ``(jac_H, clmo_H, n_dof)`` suitable for passing to
            compiled Hamiltonian RHS kernels. ``jac_H`` and ``clmo_H`` are
            sequence-like containers of numpy arrays; ``n_dof`` is an int.
        """
        ...