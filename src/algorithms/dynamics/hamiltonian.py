from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numba import njit
from numba.typed import List

from algorithms.polynomial.operations import (polynomial_evaluate,
                                                     polynomial_jacobian)
from algorithms.dynamics.base import _DynamicalSystem, DynamicalSystemProtocol
from algorithms.integrators.symplectic import _eval_dH_dP, _eval_dH_dQ
from config import FASTMATH


@njit(cache=True, fastmath=FASTMATH)
def _hamiltonian_rhs(
    state6: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    n_dof: int,
) -> np.ndarray:
    """Compute time derivative (Qdot, Pdot) for the 2*n_dof Hamiltonian system."""

    dH_dQ = np.empty(n_dof)
    dH_dP = np.empty(n_dof)

    for i in range(n_dof):
        dH_dQ[i] = polynomial_evaluate(jac_H[i], state6.astype(np.complex128), clmo).real
        dH_dP[i] = polynomial_evaluate(jac_H[n_dof + i], state6.astype(np.complex128), clmo).real

    rhs = np.empty_like(state6)
    rhs[:n_dof] = dH_dP  # dq/dt
    rhs[n_dof : 2 * n_dof] = -dH_dQ  # dp/dt
    return rhs

@runtime_checkable
class HamiltonianSystemProtocol(DynamicalSystemProtocol, Protocol):
    """
    Protocol for Hamiltonian dynamical systems.
    
    Extends DynamicalSystemProtocol with methods specific to Hamiltonian mechanics.
    These methods are required by symplectic integrators.
    """
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (dim = 2 * n_dof)."""
        ...
    
    def dH_dQ(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute partial derivatives of Hamiltonian with respect to positions.
        
        Parameters
        ----------
        Q : numpy.ndarray
            Position coordinates, shape (n_dof,)
        P : numpy.ndarray
            Momentum coordinates, shape (n_dof,)
            
        Returns
        -------
        numpy.ndarray
            Partial derivatives ∂H/∂Q, shape (n_dof,)
        """
        ...
    
    def dH_dP(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Compute partial derivatives of Hamiltonian with respect to momenta.
        
        Parameters
        ----------
        Q : numpy.ndarray
            Position coordinates, shape (n_dof,)
        P : numpy.ndarray
            Momentum coordinates, shape (n_dof,)
            
        Returns
        -------
        numpy.ndarray
            Partial derivatives ∂H/∂P, shape (n_dof,)
        """
        ...


class HamiltonianSystem(_DynamicalSystem):

    def __init__(
        self,
        jac_H: List[List[np.ndarray]],
        clmo_H: List[np.ndarray],
        n_dof: int,
        name: str = "Hamiltonian System"
    ):
        super().__init__(dim=2 * n_dof)
        
        if n_dof <= 0:
            raise ValueError(f"Number of degrees of freedom must be positive, got {n_dof}")
        
        self._n_dof = n_dof
        self.jac_H = jac_H
        self.clmo_H = clmo_H
        self.name = name
        
        self._validate_polynomial_data()
    
    @property
    def n_dof(self) -> int:
        return self._n_dof
    
    def _validate_polynomial_data(self) -> None:
        expected_vars = 2 * self.n_dof
        
        if len(self.jac_H) != expected_vars:
            raise ValueError(
                f"Jacobian must have {expected_vars} variables, got {len(self.jac_H)}"
            )
        
        if not self.clmo_H:
            raise ValueError("Coefficient layout mapping objects cannot be empty")

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        # Capture instance-specific data in a closure for the RHS function.
        jac_H, clmo_H, n_dof = self.jac_H, self.clmo_H, self.n_dof

        @njit(cache=True, fastmath=FASTMATH)
        def _rhs_closure(t: float, state: np.ndarray) -> np.ndarray:
            # The 't' argument is unused in this autonomous system but required
            # by the standard ODE solver interface.
            return _hamiltonian_rhs(state, jac_H, clmo_H, n_dof)
        
        return _rhs_closure
    
    def dH_dQ(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        self._validate_coordinates(Q, P)

        return _eval_dH_dQ(Q, P, self.jac_H, self.clmo_H)
    
    def dH_dP(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        self._validate_coordinates(Q, P)

        return _eval_dH_dP(Q, P, self.jac_H, self.clmo_H)
    
    def _validate_coordinates(self, Q: np.ndarray, P: np.ndarray) -> None:
        if len(Q) != self.n_dof:
            raise ValueError(f"Position dimension {len(Q)} != n_dof {self.n_dof}")
        if len(P) != self.n_dof:
            raise ValueError(f"Momentum dimension {len(P)} != n_dof {self.n_dof}")
    
    def __repr__(self) -> str:
        return f"HamiltonianSystem(name='{self.name}', n_dof={self.n_dof})"


def create_hamiltonian_system(
    H_blocks: List[np.ndarray],
    max_degree: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    n_dof: int = 3,
    name: str = "Center Manifold Hamiltonian"
) -> HamiltonianSystem:
    jac_H = polynomial_jacobian(H_blocks, max_degree, psi_table, clmo_table, encode_dict_list)

    jac_H_typed = List()
    for var_derivs in jac_H:
        var_list = List()
        for degree_coeffs in var_derivs:
            var_list.append(degree_coeffs)
        jac_H_typed.append(var_list)

    clmo_H = List()
    for clmo in clmo_table:
        clmo_H.append(clmo)

    return HamiltonianSystem(jac_H_typed, clmo_H, n_dof, name)
