from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Optional, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class _ConstraintGrad:
    d_r: np.ndarray        # shape (3,)
    d_v_plus: np.ndarray   # shape (3,)
    d_v_minus: np.ndarray  # shape (3,)
    d_t: float             # scalar


_ConstraintJacobianFn = Callable[[np.ndarray, np.ndarray, np.ndarray, float], _ConstraintGrad]


class _ConstraintEval(Protocol):
    def __call__(
        self,
        r_k: np.ndarray,
        v_plus_k: np.ndarray,
        v_minus_k: np.ndarray,
        t_k: float,
    ) -> float:
        ...


@dataclass(frozen=True)
class _NodePartials:
    """Per-node derivatives and kinematics derived from STMs for node k.

    The notation I_(k-1,k) means the STM from node k to node k-1.
    This is consistent with the notation in the paper.

    The notation I_k_km1 means the STM from node k to node k-1.
    This notation is for clarity and to be used in the code.

    Attributes
    ----------
    A_k_km1 : np.ndarray
        A_(k-1,k)
    B_k_km1 : np.ndarray
        B_(k-1,k)
    C_k_km1 : np.ndarray
        C_(k-1,k)
    D_k_km1 : np.ndarray
        D_(k-1,k)
    A_km1_k : np.ndarray
        A_(k,k-1)
    B_km1_k : np.ndarray
        B_(k,k-1)
    C_km1_k : np.ndarray
        C_(k,k-1)
    D_km1_k : np.ndarray
        D_(k,k-1)
    A_k_kp1 : np.ndarray
        A_(k+1,k)
    B_k_kp1 : np.ndarray
        B_(k+1,k)
    C_k_kp1 : np.ndarray
        C_(k+1,k)
    D_k_kp1 : np.ndarray
        D_(k+1,k)
    A_kp1_k : np.ndarray
        A_(k,k+1)
    B_kp1_k : np.ndarray
        B_(k,k+1)
    C_kp1_k : np.ndarray
        C_(k,k+1)
    D_kp1_k : np.ndarray
        D_(k,k+1)
    R_km1 : np.ndarray
        R_(k-1)
    R_k : np.ndarray
        R_k
    R_kp1 : np.ndarray
        R_(k+1)
    V_km1_minus: np.ndarray
        V_(k-1)^-
    V_km1_plus: np.ndarray
        V_(k-1)^+
    V_k_minus: np.ndarray
        V_k^-
    V_k_plus: np.ndarray
        V_k^+
    V_kp1_minus: np.ndarray
        V_(k+1)^-
    V_kp1_plus: np.ndarray
        V_(k+1)^+
    a_k_minus: np.ndarray
        a_k^-
    a_k_plus: np.ndarray
        a_k^+
    t_km1: float
        t_(k-1)
    t_k: float
        t_k
    t_kp1: float
        t_(k+1)
    """
    A_k_km1 : np.ndarray
    B_k_km1 : np.ndarray
    C_k_km1 : np.ndarray
    D_k_km1 : np.ndarray

    A_km1_k : np.ndarray
    B_km1_k : np.ndarray
    C_km1_k : np.ndarray
    D_km1_k : np.ndarray

    A_k_kp1 : np.ndarray
    B_k_kp1 : np.ndarray
    C_k_kp1 : np.ndarray
    D_k_kp1 : np.ndarray

    A_kp1_k : np.ndarray
    B_kp1_k : np.ndarray
    C_kp1_k : np.ndarray
    D_kp1_k : np.ndarray

    R_km1 : np.ndarray
    R_k : np.ndarray
    R_kp1 : np.ndarray

    V_km1_minus : np.ndarray
    V_km1_plus : np.ndarray
    V_k_minus : np.ndarray
    V_k_plus : np.ndarray
    V_kp1_minus : np.ndarray
    V_kp1_plus : np.ndarray

    a_k_minus : np.ndarray
    a_k_plus : np.ndarray

    t_km1 : float
    t_k : float
    t_kp1 : float


@dataclass(frozen=True)
class _ConstraintContext:
    x_patches: Sequence[np.ndarray]
    xf_patches: np.ndarray
    t_patches: np.ndarray
    stms: np.ndarray
    node_partials: Mapping[int, _NodePartials]
    segment_num: int


class _ConstraintBase(ABC):
    """Base class for all multiple-shooting constraints.
    
    All constraints must implement the build_rows method to provide
    their contribution to the constraint system.
    """

    def __init__(self, name: str) -> None:
        """Initialize the base constraint.
        
        Parameters
        ----------
        name : str
            Name of the constraint.
        """
        self.name = name
    
    @abstractmethod
    def build_rows(self, ctx: "_ConstraintContext") -> tuple[np.ndarray, np.ndarray]:
        """Build constraint rows for the system matrix.
        
        Parameters
        ----------
        ctx : _ConstraintContext
            Context containing current state and derivatives.
            
        Returns
        -------
        rows : ndarray
            Constraint matrix rows.
        rhs : ndarray
            Right-hand side values.
        """
        ...

    def _fd_grad(
        self,
        r_k: np.ndarray,
        v_plus_k: np.ndarray,
        v_minus_k: np.ndarray,
        t_k: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Central finite-difference gradient of alpha w.r.t (R_k, V_k^+, V_k^-, t_k)."""
        h = self.fd_step if self.fd_step is not None else 1e-6
        h = float(max(1e-12, h))

        def feval(r, vp, vm, t):
            return float(self.eval_fn(r, vp, vm, t))

        g_r = np.zeros(3)
        for i in range(3):
            dr = np.zeros(3); dr[i] = h
            g_r[i] = (feval(r_k + dr, v_plus_k, v_minus_k, t_k) - feval(r_k - dr, v_plus_k, v_minus_k, t_k)) / (2 * h)

        g_p = np.zeros(3)
        for i in range(3):
            dv = np.zeros(3); dv[i] = h
            g_p[i] = (feval(r_k, v_plus_k + dv, v_minus_k, t_k) - feval(r_k, v_plus_k - dv, v_minus_k, t_k)) / (2 * h)

        g_m = np.zeros(3)
        for i in range(3):
            dv = np.zeros(3); dv[i] = h
            g_m[i] = (feval(r_k, v_plus_k, v_minus_k + dv, t_k) - feval(r_k, v_plus_k, v_minus_k - dv, t_k)) / (2 * h)

        g_t = (feval(r_k, v_plus_k, v_minus_k, t_k + h) - feval(r_k, v_plus_k, v_minus_k, t_k - h)) / (2 * h)

        return g_r, g_p, g_m, float(g_t)


class PeriodicityConstraint(_ConstraintBase):
    """Preset provider for periodicity (R0=RN and V1^+=V_N^-).
    
    Parameters
    ----------
    name : str, optional
        Name of the constraint (default: "periodicity").
    """

    _name = "periodicity"

    def __init__(
        self,
    ) -> None:
        """Initialize the periodicity constraint."""
        super().__init__(name=self._name)

    def build_rows(self, ctx: _ConstraintContext) -> tuple[np.ndarray, np.ndarray]:
        pass