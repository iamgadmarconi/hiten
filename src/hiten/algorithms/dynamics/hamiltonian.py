r"""
dynamics.hamiltonian
====================

Core utilities for constructing and integrating finite-dimensional polynomial
Hamiltonian systems that arise in the centre-manifold reduction of the spatial
circular restricted three-body problem (CRTBP).

The module translates a list of packed polynomial blocks - typically produced
by the normal-form pipeline - into a lightweight, JIT-compiled rhs suitable
for both explicit Runge-Kutta and symplectic integrators.  All heavy symbolic
work is delegated to :pyfunc:`hiten.algorithms.polynomial.operations.polynomial_jacobian` and
the numba-compiled helpers in :pyfunc:`hiten.algorithms.integrators.symplectic`.

References
----------
Jorba, À. (1999) "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".
"""

from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.dynamics.base import (DynamicalSystemProtocol,
                                            _DynamicalSystem)
from hiten.algorithms.integrators.symplectic import _eval_dH_dP, _eval_dH_dQ
from hiten.algorithms.polynomial.operations import (polynomial_evaluate,
                                                    polynomial_jacobian)
from hiten.utils.config import FASTMATH


@njit(cache=False, fastmath=FASTMATH)
def _hamiltonian_rhs(
    state6: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    n_dof: int,
) -> np.ndarray:
    r"""
    Compute time derivative (Qdot, Pdot) for the :math:`2*n_dof` Hamiltonian :pyfunc:`hiten.system`.

    Parameters
    ----------
    state6 : numpy.ndarray
        State vector of the :math:`2*n_dof` Hamiltonian :pyfunc:`hiten.system`.
    jac_H : numba.typed.List of numba.typed.List of numpy.ndarray
        Jacobian of the Hamiltonian.
    clmo : numba.typed.List of numpy.ndarray
        Coefficient-layout mapping objects used by
        :pyfunc:`polynomial_evaluate`.
    n_dof : int
        Number of degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Time derivative (Qdot, Pdot) of the 2*n_dof Hamiltonian hiten.system.
    """

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
    r"""
    Protocol for Hamiltonian dynamical systems.
    
    Extends DynamicalSystemProtocol with methods specific to Hamiltonian mechanics.
    These methods are required by symplectic integrators.
    """
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom (dim = 2 * n_dof)."""
        ...
    
    def dH_dQ(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        r"""
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
        r"""
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
    r"""
    Lightweight polynomial Hamiltonian wrapper.

    The class stores the Jacobian of a polynomial Hamiltonian in packed form
    and exposes the information required by symplectic integrators, namely
    :pyfunc:`dH_dQ`, :pyfunc:`dH_dP` and an autonomous right-hand side
    complying with the common ODE interface ``f(t, y)``.

    Parameters
    ----------
    jac_H : numba.typed.List of numba.typed.List of numpy.ndarray
        Hierarchical list holding the coefficient arrays of the partial
        derivatives :math:`\partial H/\partial x_i` grouped by variable and
        total degree.  The outermost index selects the variable, the second
        the polynomial degree.
    clmo_H : numba.typed.List of numpy.ndarray
        Coefficient-layout mapping objects used by
        :pyfunc:`hiten.algorithms.polynomial.operations.polynomial_evaluate`.
    n_dof : int
        Number of degrees of freedom, the full phase-space dimension is
        ``2 * n_dof``.
    name : str, default="Hamiltonian System"
        Human-readable identifier used in :pyfunc:`__repr__`.

    Raises
    ------
    ValueError
        If *n_dof* is not positive or if the shapes of *jac_H* / *clmo_H* are
        inconsistent.

    Notes
    -----
    The internal RHS closure is JIT-compiled on first call via ``numba.njit``
    with the global flag :pydata:`hiten.utils.config.FASTMATH`.

    Examples
    --------
    >>> sys = HamiltonianSystem(jac_H, clmo, n_dof=3)
    >>> ydot = sys.rhs(0.0, y0)  # integrates as a standard autonomous ODE
    """

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
        r"""
        Return a lightweight wrapper around the compiled Hamiltonian RHS.

        The heavy numerical work is carried out by the JIT-compiled helper
        :pyfunc:`_hamiltonian_rhs`.  We purposefully keep this outer wrapper
        as a regular Python function to avoid capturing *numba.typed.List*
        objects inside another Numba-compiled closure - such closures cannot
        be pickled during compilation and therefore trigger errors in the
        caching layer.  The marginal overhead of this thin Python layer is
        negligible compared to the cost of evaluating high-order polynomial
        Hamiltonians, while it guarantees compatibility with the JIT runtime.
        """

        jac_H, clmo_H, n_dof = self.jac_H, self.clmo_H, self.n_dof

        def _rhs_closure(t: float, state: np.ndarray) -> np.ndarray:
            # The 't' argument is unused in this autonomous system but
            # required by the standard ODE solver interface.
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
    r"""
    Factory helper that converts packed polynomial data into a runtime Hamiltonian.

    Parameters
    ----------
    H_blocks : list of numpy.ndarray
        Packed coefficient arrays :math:`[H_0, H_2, \dots, H_N]` returned by the
        centre-manifold pipeline.
    max_degree : int
        Maximum total degree :math:`N` represented in *H_blocks*.
    psi_table : numpy.ndarray
        Lookup table mapping monomial exponents to packed indices (see
        :pyfunc:`hiten.algorithms.polynomial.base.init_index_tables`).
    clmo_table : list of numpy.ndarray
        Per-degree coefficient-layout mapping objects.
    encode_dict_list : list of dict
        Encoder dictionaries required by :pyfunc:`hiten.algorithms.polynomial.operations.polynomial_jacobian`.
    n_dof : int, default=3
        Number of degrees of freedom.
    name : str, default="Center Manifold Hamiltonian"
        Identifier forwarded to the underlying :pyclass:`HamiltonianSystem`.

    Returns
    -------
    HamiltonianSystem
        Ready-to-integrate instance wrapping the supplied Hamiltonian.

    Notes
    -----
    The function performs three lightweight steps:

    1. Build the Jacobian via :pyfunc:`hiten.algorithms.polynomial.operations.polynomial_jacobian`.
    2. Cast nested Python lists into ``numba.typed.List`` so that JIT compiled
       helpers can access them at native speed.
    3. Instantiate and return :pyclass:`HamiltonianSystem`.
    """
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
