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
    """Per-node derivatives and kinematics derived from STMs for node k."""

    A_backward: np.ndarray
    B_backward: np.ndarray
    D_backward: np.ndarray
    A_forward: np.ndarray
    B_forward: np.ndarray
    D_forward: np.ndarray
    v_km1_plus: np.ndarray
    v_k_minus: np.ndarray
    v_k_plus: np.ndarray
    v_kp1_minus: np.ndarray
    a_k_plus: np.ndarray
    a_k_minus: np.ndarray


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
        if ctx.segment_num < 2:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return np.zeros((0, n_cols)), np.zeros(0)

        n_cols = (ctx.segment_num - 2) * 4 + 12
        rows = np.zeros((6, n_cols))
        rhs = np.zeros(6)

        def r_cols(j: int) -> slice:
            base = j * 4
            return slice(base, base + 3)

        def t_col(j: int) -> int:
            return j * 4 + 3

        # Velocity periodicity
        p1 = ctx.node_partials.get(1) 
        pNm1 = ctx.node_partials.get(ctx.segment_num - 1)

        if p1 is None or pNm1 is None:
            return rows, rhs

        def _B_inv(B_block):
            try:
                return np.linalg.solve(B_block, np.eye(3))
            except np.linalg.LinAlgError:
                return np.linalg.pinv(B_block, rcond=1e-12)

        B12 = p1.B_forward # B_{12}
        B12_inv = _B_inv(B12) # B_{12}^{-1}
        A12 = p1.A_forward # A_{12}
        D12 = p1.D_forward # D_{12}

        B_21 = p1.B_backward # B_{21}
        B_21_inv = _B_inv(B_21) # B_{21}^{-1}
        A_21 = p1.A_backward # A_{21}
        D_21 = p1.D_backward # D_{21}

        B_f_inv = _B_inv(pNm1.B_forward)  # B_{N-1,N}^{-1}
        A_f = pNm1.A_forward # A_{N-1,N}
        D_f = pNm1.D_forward # D_{N-1,N}

        # Read terminal backward blocks from node_partials (added by backend)
        parts_N = ctx.node_partials.get(ctx.segment_num)
        if parts_N is None:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return rows, rhs
        B_NN1 = parts_N.B_backward
        D_NN1 = parts_N.D_backward
        B_NN1_inv = _B_inv(B_NN1) # B_{N,N-1}^{-1}

        r1 = np.asarray(ctx.x_patches[1][:3], dtype=float) # r_1
        rN_minus = np.asarray(ctx.xf_patches[ctx.segment_num - 1][:3], dtype=float) # r_N^-

        v1_plus = np.asarray(ctx.x_patches[1][3:6], dtype=float)  # v_1^+
        v2_minus = p1.v_kp1_minus  # v_2^-
        vN_plus = np.asarray(ctx.x_patches[ctx.segment_num][3:6], dtype=float)  # v_N^+
        vN_minus = np.asarray(ctx.xf_patches[ctx.segment_num - 1][3:6], dtype=float)  # v_N^-

        a_1_plus = p1.a_k_plus  # a_1^+
        a_N_minus = pNm1.a_k_minus  # a_N^-

        rows[0:3, r_cols(1)] = np.eye(3) # I ; dR_1
        # Left boundary (k=1) velocity columns use forward blocks (B_{12}^{-1}, A_{12}, D_{12})
        # Left boundary, literal (53) orientation using backward/forward mix per paper
        rows[3:6, r_cols(1)] = (-B_21_inv @ A_21) # -B_{21}^{-1} A_{21} ; dR_1

        rows[0:3, t_col(1)] = np.zeros(3) # 0 ; dt_1
        rows[3:6, t_col(1)] = (a_1_plus - D12 @ (B12_inv @ v1_plus)).reshape(3) # a_1^+ - D_{12} B_{12}^{-1} v_1^+ ; dt_1

        rows[0:3, r_cols(2)] = np.zeros(3) # 0 ; dR_2
        rows[3:6, r_cols(2)] = (B_21_inv) # B_{21}^{-1} ; dR_2

        rows[0:3, t_col(2)] = np.zeros(3) # 0 ; dt_2
        rows[3:6, t_col(2)] = (-B_21_inv @ v2_minus).reshape(3) # -B_{21}^{-1} v_2^- ; dt_2

        rows[0:3, r_cols(ctx.segment_num - 1)] = np.zeros(3) # 0 ; dR_{N-1}
        rows[3:6, r_cols(ctx.segment_num - 1)] = - B_f_inv # - B_{N-1,N}^{-1} ; dR_{N-1}

        rows[0:3, t_col(ctx.segment_num - 1)] = np.zeros(3) # 0 ; dt_{N-1}
        rows[3:6, t_col(ctx.segment_num - 1)] = (B_f_inv @ vN_plus).reshape(3) # B_{N-1,N}^{-1} v_{N}^+ ; dt_{N-1}

        rows[0:3, r_cols(ctx.segment_num)] = - np.eye(3) # -I ; dR_N
        rows[3:6, r_cols(ctx.segment_num)] = (B_f_inv @ A_f) # B_{N-1,N}^{-1} A_{N-1,N} ; dR_N

        rows[0:3, t_col(ctx.segment_num)] = np.zeros(3) # 0 ; dt_N
        rows[3:6, t_col(ctx.segment_num)] = -(a_N_minus - D_NN1 @ (B_NN1_inv @ vN_minus)).reshape(3) # -(a_N^- - D_{N,N-1} B_{N,N-1}^{-1} v_N^-) ; dt_N

        rhs[0:3] = -(r1 - rN_minus)
        rhs[3:6] = -(v1_plus - vN_minus)

        return rows, rhs
