r"""Provide polynomial Hamiltonian systems for center manifold dynamics.

This module provides utilities for constructing and integrating finite-dimensional
polynomial Hamiltonian systems that arise from center-manifold reduction of the
spatial Circular Restricted Three-Body Problem (CR3BP).

The module transforms packed polynomial coefficient blocks from the normal-form
pipeline into lightweight, JIT-compiled dynamical systems suitable for both
explicit Runge-Kutta and symplectic integrators.

The heavy symbolic computations are delegated to the polynomial operations
module, while Numba compilation ensures efficient runtime evaluation.

References
----------
Jorba, A. (1999). A methodology for the numerical computation of normal forms,
centre manifolds and first integrals of Hamiltonian systems.
*Experimental Mathematics*, 8(2), 155-195.
"""

from typing import Callable, List

import numpy as np
from numba import njit
from numba.typed import List

from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.integrators.symplectic import _eval_dH_dP, _eval_dH_dQ
from hiten.algorithms.polynomial.operations import (_polynomial_evaluate,
                                                    _polynomial_jacobian)
from hiten.algorithms.utils.config import FASTMATH


@njit(cache=False, fastmath=FASTMATH)
def _hamiltonian_rhs(
    state6: np.ndarray,
    jac_H: List[List[np.ndarray]],
    clmo: List[np.ndarray],
    n_dof: int,
) -> np.ndarray:
    r"""Compute Hamilton's equations for polynomial Hamiltonian system.

    JIT-compiled function that evaluates the time derivatives (dQ/dt, dP/dt)
    for a polynomial Hamiltonian system using Hamilton's canonical equations:
    dQ/dt = dH/dP, dP/dt = -dH/dQ.

    Parameters
    ----------
    state6 : ndarray
        State vector [Q1, Q2, Q3, P1, P2, P3] for the 2*n_dof Hamiltonian system.
    jac_H : List[List[ndarray]]
        Nested Numba-typed list containing Jacobian coefficients of the Hamiltonian.
        Structure: jac_H[i][j] contains degree-j coefficients for partial
        derivative with respect to variable i.
    clmo : List[ndarray]
        Numba-typed list of coefficient-layout mapping objects for polynomial
        evaluation at each degree.
    n_dof : int
        Number of degrees of freedom (half the state dimension).

    Returns
    -------
    ndarray
        Time derivative vector [dQ/dt, dP/dt] = [dH/dP, -dH/dQ].
        
    Notes
    -----
    - Uses JIT compilation for efficient polynomial evaluation
    - Handles complex arithmetic internally but returns real derivatives
    - Assumes autonomous Hamiltonian (no explicit time dependence)
    
    See Also
    --------
    :func:`~hiten.algorithms.polynomial.operations._polynomial_evaluate` :
        Polynomial evaluation
    :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem` : Uses
        this function for RHS computation
    """

    dH_dQ = np.empty(n_dof)
    dH_dP = np.empty(n_dof)

    for i in range(n_dof):
        dH_dQ[i] = _polynomial_evaluate(jac_H[i], state6.astype(np.complex128), clmo).real
        dH_dP[i] = _polynomial_evaluate(jac_H[n_dof + i], state6.astype(np.complex128), clmo).real

    rhs = np.empty_like(state6)
    rhs[:n_dof] = dH_dP  # dq/dt
    rhs[n_dof : 2 * n_dof] = -dH_dQ  # dp/dt
    return rhs



class _HamiltonianSystem(_DynamicalSystem):
    r"""Define a polynomial Hamiltonian system for numerical integration.

    Implements a dynamical system based on a polynomial Hamiltonian function.
    Stores the Jacobian in packed form and provides both standard ODE interface
    (for general integrators) and Hamiltonian-specific methods (for symplectic
    integrators).

    The system automatically computes Hamilton's equations: dQ/dt = dH/dP,
    dP/dt = -dH/dQ using JIT-compiled polynomial evaluation.

    Parameters
    ----------
    H_blocks : List[ndarray]
        Packed coefficient arrays [H_0, H_2, ..., H_N] from center-manifold
        pipeline, where H_k contains degree-k polynomial coefficients.
    degree : int
        Maximum polynomial degree N represented in H_blocks.
    psi_table : ndarray
        Lookup table mapping monomial exponents to packed array indices.
    clmo_table : List[ndarray]
        Coefficient-layout mapping objects for each polynomial degree.
    encode_back_dict_list : List[dict]
        Encoder dictionaries for polynomial Jacobian computation.
    n_dof : int
        Number of degrees of freedom. Total state dimension is 2 * n_dof.
    name : str, optional
        Human-readable system identifier. Default is "Hamiltonian System".

    Attributes
    ----------
    n_dof : int
        Number of degrees of freedom.
    jac_H : List[List[ndarray]]
        Numba-typed nested list containing Jacobian coefficients.
    clmo_H : List[ndarray]
        Numba-typed list of coefficient-layout mapping objects.

    Raises
    ------
    ValueError
        If n_dof is not positive or polynomial data shapes are inconsistent.

    Notes
    -----
    - RHS function is JIT-compiled on first call for efficiency
    - Supports both autonomous ODE interface and Hamiltonian-specific methods
    - Compatible with general ODE solvers and symplectic integrators
    - Polynomial evaluation uses complex arithmetic internally but returns real derivatives

    Examples
    --------
    >>> # Create system from polynomial data
    >>> sys = _HamiltonianSystem(H_blocks, degree, psi_table, clmo_table, 
    ...                          encode_dict_list, n_dof=3)
    >>> # Use with ODE solver
    >>> ydot = sys.rhs(0.0, state)  # Standard ODE interface
    >>> # Use with symplectic integrator
    >>> dH_dQ = sys.dH_dQ(Q, P)  # Hamiltonian derivatives
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystemProtocol` :
        Interface specification
    :func:`~hiten.algorithms.dynamics.hamiltonian.create_hamiltonian_system` :
        Factory function
    :func:`~hiten.algorithms.dynamics.hamiltonian._hamiltonian_rhs` :
        JIT-compiled RHS computation
    """

    def __init__(
        self,
        H_blocks: List[np.ndarray],
        degree: int,
        psi_table: np.ndarray,
        clmo_table: List[np.ndarray],
        encode_dict_list: List,
        n_dof: int,
        name: str = "Hamiltonian System"
    ):
        super().__init__(dim=2 * n_dof)
        
        if n_dof <= 0:
            raise ValueError(f"Number of degrees of freedom must be positive, got {n_dof}")
        
        jac_H = _polynomial_jacobian(H_blocks, degree, psi_table, clmo_table, encode_dict_list)
        # Store in numba.typed.List to avoid reflected Python list issues inside njit.
        jac_H_typed = List()
        for var_derivs in jac_H:
            var_list = List()
            for degree_coeffs in var_derivs:
                var_list.append(degree_coeffs)
            jac_H_typed.append(var_list)

        clmo_H = List()
        for clmo in clmo_table:
            clmo_H.append(clmo)
        
        self._n_dof = n_dof
        self.jac_H = jac_H_typed
        self.clmo_H = clmo_H
        self.H_blocks = H_blocks
        self.degree = degree
        self.psi_table = psi_table
        self.clmo_table = clmo_table
        self.encode_dict_list = encode_dict_list
        self.name = name
        
        self._validate_polynomial_data()
        
        # Hamiltonian system uses base class caching/compilation for rhs
    
    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom.
        
        Returns
        -------
        int
            Degrees of freedom count. Total state dimension is 2 * n_dof.
        """
        return self._n_dof
    
    def _validate_polynomial_data(self) -> None:
        """Validate consistency of polynomial data structures.
        
        Raises
        ------
        ValueError
            If Jacobian dimensions or coefficient mapping objects are invalid.
        """
        expected_vars = 2 * self.n_dof
        
        if len(self.jac_H) != expected_vars:
            raise ValueError(
                f"Jacobian must have {expected_vars} variables, got {len(self.jac_H)}"
            )
        
        if not self.clmo_H:
            raise ValueError("Coefficient layout mapping objects cannot be empty")

    def _build_rhs_impl(self) -> Callable[[float, np.ndarray], np.ndarray]:
        """
        Return a compiled function implementing Hamilton's equations.
        
        Returns
        -------
        Callable[[float, ndarray], ndarray]
            Compiled function implementing Hamilton's equations.
        """

        jac_H, clmo_H, n_dof = self.jac_H, self.clmo_H, self.n_dof

        def _rhs_impl(t: float, state: np.ndarray) -> np.ndarray:
            # Autonomous: t is unused; required for interface consistency
            return _hamiltonian_rhs(state, jac_H, clmo_H, n_dof)

        return _rhs_impl
    
    @property
    def clmo(self) -> List[np.ndarray]:
        """Coefficient-layout mapping objects for polynomial evaluation.
        
        Returns
        -------
        List[ndarray]
            Numba-typed list of coefficient-layout mapping objects.
            
        See Also
        --------
        :func:`~hiten.algorithms.polynomial.operations._polynomial_evaluate` :
            Uses these objects
        """
        return self.clmo_H
    
    def dH_dQ(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute partial derivatives dH/dQ for symplectic integration.

        Parameters
        ----------
        Q : ndarray, shape (n_dof,)
            Position coordinates.
        P : ndarray, shape (n_dof,)
            Momentum coordinates.

        Returns
        -------
        ndarray, shape (n_dof,)
            Partial derivatives of Hamiltonian with respect to positions.
            
        Raises
        ------
        ValueError
            If Q or P dimensions don't match n_dof.
            
        See Also
        --------
        :func:`~hiten.algorithms.integrators.symplectic._eval_dH_dQ` :
            Implementation
        :meth:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem.dH_dP` :
            Momentum derivatives
        """
        self._validate_coordinates(Q, P)
        return _eval_dH_dQ(Q, P, self.jac_H, self.clmo_H)
    
    def dH_dP(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Compute partial derivatives dH/dP for symplectic integration.

        Parameters
        ----------
        Q : ndarray, shape (n_dof,)
            Position coordinates.
        P : ndarray, shape (n_dof,)
            Momentum coordinates.

        Returns
        -------
        ndarray, shape (n_dof,)
            Partial derivatives of Hamiltonian with respect to momenta.
            
        Raises
        ------
        ValueError
            If Q or P dimensions don't match n_dof.
            
        See Also
        --------
        :func:`~hiten.algorithms.integrators.symplectic._eval_dH_dP` :
            Implementation
        :meth:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem.dH_dQ` :
            Position derivatives
        """
        self._validate_coordinates(Q, P)
        return _eval_dH_dP(Q, P, self.jac_H, self.clmo_H)
    
    def poly_H(self) -> List[List[np.ndarray]]:
        """Return polynomial coefficient blocks of the Hamiltonian.

        Returns
        -------
        List[List[ndarray]]
            Nested list of polynomial coefficient arrays organized by degree.
            Structure: H_blocks[k] contains degree-k coefficients.
            
        See Also
        --------
        :func:`~hiten.algorithms.dynamics.hamiltonian.create_hamiltonian_system` :
            Uses these blocks for system creation
        """
        return self.H_blocks
    
    def _validate_coordinates(self, Q: np.ndarray, P: np.ndarray) -> None:
        """Validate coordinate array dimensions.
        
        Parameters
        ----------
        Q : ndarray
            Position coordinates to validate.
        P : ndarray
            Momentum coordinates to validate.
            
        Raises
        ------
        ValueError
            If coordinate dimensions don't match system n_dof.
        """
        if len(Q) != self.n_dof:
            raise ValueError(f"Position dimension {len(Q)} != n_dof {self.n_dof}")
        if len(P) != self.n_dof:
            raise ValueError(f"Momentum dimension {len(P)} != n_dof {self.n_dof}")
    
    def __repr__(self) -> str:
        """String representation of the Hamiltonian system.
        
        Returns
        -------
        str
            Formatted string showing system name and degrees of freedom.
        """
        return f"_HamiltonianSystem(name='{self.name}', n_dof={self.n_dof})"

    @property
    def rhs_params(self) -> tuple:
        """Return low-level RHS parameters for parametric kernels.

        Returns
        -------
        tuple
            (jac_H, clmo_H, n_dof) for use with compiled Hamiltonian RHS.
        """
        return (self.jac_H, self.clmo_H, self.n_dof)


def create_hamiltonian_system(
    H_blocks: List[np.ndarray],
    degree: int,
    psi_table: np.ndarray,
    clmo_table: List[np.ndarray],
    encode_dict_list: List,
    n_dof: int = 3,
    name: str = "Center Manifold Hamiltonian"
) -> _HamiltonianSystem:
    r"""Create polynomial Hamiltonian system from coefficient data.

    Factory function that converts packed polynomial coefficient blocks from
    the center-manifold pipeline into a ready-to-integrate Hamiltonian system.
    Handles all necessary data structure conversions and validations.

    Parameters
    ----------
    H_blocks : List[ndarray]
        Packed coefficient arrays [H_0, H_2, ..., H_N] from center-manifold
        reduction, where H_k contains degree-k polynomial coefficients.
    degree : int
        Maximum polynomial degree N represented in H_blocks.
    psi_table : ndarray
        Lookup table mapping monomial exponents to packed array indices.
    clmo_table : List[ndarray]
        Coefficient-layout mapping objects for each polynomial degree.
    encode_dict_list : List[dict]
        Encoder dictionaries for polynomial Jacobian computation.
    n_dof : int, optional
        Number of degrees of freedom. Default is 3 (typical for CR3BP).
    name : str, optional
        Human-readable system identifier. Default is "Center Manifold Hamiltonian".

    Returns
    -------
    _HamiltonianSystem
        Configured Hamiltonian system ready for numerical integration.
        
    Examples
    --------
    >>> # Create system from center manifold data
    >>> sys = create_hamiltonian_system(H_blocks, degree, psi_table, 
    ...                                  clmo_table, encode_dict_list, n_dof=3)
    >>> # Integrate with ODE solver
    >>> sol = solve_ivp(sys.rhs, [0, 10], initial_state)
    
    See Also
    --------
    :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem` : Underlying system implementation
    :func:`~hiten.algorithms.polynomial.operations._polynomial_jacobian` : Jacobian computation
    """
    return _HamiltonianSystem(H_blocks, degree, psi_table, clmo_table, encode_dict_list, n_dof, name)
