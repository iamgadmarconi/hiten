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
    
    def __init__(self, name: str = "constraint", active: bool = True) -> None:
        """Initialize the base constraint.
        
        Parameters
        ----------
        name : str
            Name of the constraint.
        active : bool
            Whether the constraint is active.
        """
        self.name = name
        self.active = active
    
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


class _ScalarConstraint(_ConstraintBase):
    """Base class for scalar constraint applied at a multiple-shooting node.

    Represents alpha_{kj} = alpha(R_k, bar{V}_k^+, bar{V}_k^-, t_k) with an
    optional analytic gradient in the native variables. When ``jacobian_fn`` is
    ``None``, callers may approximate derivatives via finite differences.
    
    Parameters
    ----------
    name : str
        Name of the constraint.
    patch_index : int
        Node index k (0..segment_num).
    eval_fn : _ConstraintEval
        Function to evaluate the constraint value.
    target : float, optional
        Desired value alpha_{kj} (default: 0.0).
    weight : float, optional
        Weight for the constraint (default: 1.0).
    jacobian_fn : _ConstraintJacobianFn, optional
        Analytic gradient function (default: None).
    fd_step : float, optional
        Finite-difference step override (default: None).
    active : bool, optional
        Enable/disable constraint (default: True).
    """

    def __init__(
        self,
        name: str,
        patch_index: int,
        eval_fn: _ConstraintEval,
        target: float = 0.0,
        weight: float = 1.0,
        jacobian_fn: Optional[_ConstraintJacobianFn] = None,
        fd_step: Optional[float] = None,
        active: bool = True,
    ) -> None:
        """Initialize the constraint."""
        super().__init__(name=name, active=active)
        self.patch_index = patch_index
        self.eval_fn = eval_fn
        self.target = target
        self.weight = weight
        self.jacobian_fn = jacobian_fn
        self.fd_step = fd_step

    def build_rows(self, ctx: "_ConstraintContext") -> tuple[np.ndarray, np.ndarray]:
        if not self.active:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return np.zeros((0, n_cols)), np.zeros(0)

        k = int(self.patch_index)
        if k <= 0 or k >= ctx.segment_num:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return np.zeros((0, n_cols)), np.zeros(0)

        parts = ctx.node_partials.get(k, None)
        if parts is None:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return np.zeros((0, n_cols)), np.zeros(0)

        r_k = np.asarray(ctx.x_patches[k][:3], dtype=float)
        v_plus_k = np.asarray(ctx.x_patches[k][3:6], dtype=float)
        v_minus_k = np.asarray(ctx.xf_patches[k - 1][3:6], dtype=float)
        t_k = float(ctx.t_patches[k])

        alpha = float(self.eval_fn(r_k, v_plus_k, v_minus_k, t_k))
        delta_alpha = float(self.target - alpha)

        if self.jacobian_fn is not None:
            grad = self.jacobian_fn(r_k, v_plus_k, v_minus_k, t_k)
            g_r = np.asarray(grad.d_r, dtype=float).reshape(3)
            g_p = np.asarray(grad.d_v_plus, dtype=float).reshape(3)
            g_m = np.asarray(grad.d_v_minus, dtype=float).reshape(3)
            g_t = float(grad.d_t)
        else:
            g_r, g_p, g_m, g_t = self._fd_grad(r_k, v_plus_k, v_minus_k, t_k)

        block = self._compose_constraint_block_from_native_grads(g_r, g_t, g_p, g_m, parts)

        n_cols = (ctx.segment_num - 2) * 4 + 12
        row = np.zeros((1, n_cols))
        col_start = (k - 1) * 4
        row[0, col_start:col_start + 12] = block

        scale = np.sqrt(self.weight) if self.weight != 1.0 else 1.0
        if scale != 1.0:
            row *= scale
            delta_alpha *= scale

        return row, np.asarray([delta_alpha], dtype=float)

    def _compose_constraint_block_from_native_grads(
        self,
        g_r: np.ndarray,
        g_t: float,
        g_p: np.ndarray,
        g_m: np.ndarray,
        parts: "_NodePartials",
    ) -> np.ndarray:
        """Compose a 1x12 block for a scalar constraint at node k.

        Order: [R_{k-1}(3), t_{k-1}(1), R_k(3), t_k(1), R_{k+1}(3), t_{k+1}(1)].
        """
        B_b_inv = np.linalg.solve(parts.B_backward, np.eye(3))
        B_f_inv = np.linalg.solve(parts.B_forward, np.eye(3))

        row = np.zeros(12)
        row[0:3] = (g_p @ (-B_b_inv @ parts.A_backward))
        row[3] = float(g_p @ (parts.a_k_plus - parts.D_backward @ (B_b_inv @ parts.v_km1_plus)))
        row[4:7] = g_r + (g_p @ B_b_inv) + (g_m @ B_f_inv)
        row[7] = float(g_t + g_p @ (-B_b_inv @ parts.v_k_plus) + g_m @ (-B_f_inv @ parts.v_kp1_minus))
        row[8:11] = (g_m @ (-B_f_inv @ parts.A_forward))
        row[11] = float(g_m @ (parts.a_k_minus - parts.D_forward @ (B_f_inv @ parts.v_k_plus)))
        return row





class PeriodicityConstraint(_ConstraintBase):
    """Preset provider for periodicity (R0=RN and V1^+=V_N^-).
    
    Parameters
    ----------
    name : str, optional
        Name of the constraint (default: "periodicity").
    weight_pos : float, optional
        Weight for position periodicity (default: 1.0).
    weight_vel : float, optional
        Weight for velocity periodicity (default: 1.0).
    active : bool, optional
        Enable/disable constraint (default: True).
    """

    def __init__(
        self,
        name: str = "periodicity",
        weight_pos: float = 1.0,
        weight_vel: float = 1.0,
        active: bool = True,
    ) -> None:
        """Initialize the periodicity constraint."""
        super().__init__(name=name, active=active)
        self.weight_pos = weight_pos
        self.weight_vel = weight_vel

    def build_rows(self, ctx: _ConstraintContext) -> tuple[np.ndarray, np.ndarray]:
        if not self.active or ctx.segment_num < 2:
            n_cols = (ctx.segment_num - 2) * 4 + 12
            return np.zeros((0, n_cols)), np.zeros(0)

        n_cols = (ctx.segment_num - 2) * 4 + 12
        rows = np.zeros((6, n_cols))
        # Build RHS as -current residuals (α* - α). For periodicity α*=0.
        rhs = np.zeros(6)

        def r_cols(j: int) -> slice:
            base = j * 4
            return slice(base, base + 3)

        def t_col(j: int) -> int:
            return j * 4 + 3

        # Position periodicity
        sp = np.sqrt(self.weight_pos) if self.weight_pos != 1.0 else 1.0
        rows[0:3, r_cols(0)] += np.eye(3) * sp
        rows[0:3, r_cols(ctx.segment_num)] -= np.eye(3) * sp
        # Current position residual: R0 - R_N^-
        r0 = np.asarray(ctx.x_patches[0][:3], dtype=float)
        rN_minus = np.asarray(ctx.xf_patches[ctx.segment_num - 1][:3], dtype=float)
        rhs[0:3] = -(r0 - rN_minus) * sp

        # Velocity periodicity
        sv = np.sqrt(self.weight_vel) if self.weight_vel != 1.0 else 1.0
        p1 = ctx.node_partials.get(1)
        pNm1 = ctx.node_partials.get(ctx.segment_num - 1)
        if p1 is None or pNm1 is None:
            return rows, rhs

        B_b_inv = np.linalg.solve(p1.B_backward, np.eye(3))
        rows[3:6, r_cols(0)] += (-B_b_inv @ p1.A_backward) * sv
        rows[3:6, t_col(0)] += (p1.a_k_plus - p1.D_backward @ (B_b_inv @ p1.v_km1_plus)).reshape(3) * sv
        rows[3:6, r_cols(1)] += (B_b_inv) * sv
        rows[3:6, t_col(1)] += (-B_b_inv @ p1.v_k_plus).reshape(3) * sv

        B_f_inv = np.linalg.solve(pNm1.B_forward, np.eye(3))
        rows[3:6, r_cols(ctx.segment_num - 1)] -= (B_f_inv) * sv
        rows[3:6, t_col(ctx.segment_num - 1)] -= (-B_f_inv @ pNm1.v_kp1_minus).reshape(3) * sv
        rows[3:6, r_cols(ctx.segment_num)] -= (-B_f_inv @ pNm1.A_forward) * sv
        rows[3:6, t_col(ctx.segment_num)] -= (pNm1.a_k_minus - pNm1.D_forward @ (B_f_inv @ pNm1.v_k_plus)).reshape(3) * sv
        # Current velocity residual: V1^+ - V_N^-
        v1_plus = np.asarray(ctx.x_patches[1][3:6], dtype=float)
        vN_minus = np.asarray(ctx.xf_patches[ctx.segment_num - 1][3:6], dtype=float)
        rhs[3:6] = -(v1_plus - vN_minus) * sv

        return rows, rhs


class SpecificEnergyConstraint(_ScalarConstraint):
    def __init__(
        self,
        *,
        patch_index: int,
        mu: float,
        target: float,
        velocity_side: str = "plus",
        weight: float = 1.0,
        active: bool = True,
    ) -> None:
        side = "plus" if velocity_side.lower() != "minus" else "minus"

        super().__init__(
            name="specific_energy",
            patch_index=patch_index,
            eval_fn=lambda r_k, v_plus, v_minus, t_k: self._eval_fn(side, mu, r_k, v_plus, v_minus, t_k),
            target=target,
            weight=weight,
            jacobian_fn=lambda r_k, v_plus, v_minus, t_k: self._jac_fn(side, mu, r_k, v_plus, v_minus, t_k),
            fd_step=None,
            active=active,
        )

    def _eval_fn(self, side: str, mu, r_k: np.ndarray, v_plus: np.ndarray, v_minus: np.ndarray, t_k: float) -> float:
        v = v_plus if side == "plus" else v_minus
        r_norm = float(np.linalg.norm(r_k))
        r_norm = r_norm if r_norm > 1e-12 else 1e-12
        return 0.5 * float(v @ v) - float(mu) / r_norm

    def _jac_fn(self, side: str, mu, r_k: np.ndarray, v_plus: np.ndarray, v_minus: np.ndarray, t_k: float) -> _ConstraintGrad:
        r_norm = float(np.linalg.norm(r_k))
        r_norm = r_norm if r_norm > 1e-12 else 1e-12
        d_r = (float(mu) * r_k) / (r_norm ** 3)
        if side == "plus":
            d_v_plus = v_plus
            d_v_minus = np.zeros(3)
        else:
            d_v_plus = np.zeros(3)
            d_v_minus = v_minus
        return _ConstraintGrad(d_r=d_r, d_v_plus=d_v_plus, d_v_minus=d_v_minus, d_t=0.0)


class ApseConstraint(_ScalarConstraint):
    def __init__(
        self,
        *,
        patch_index: int,
        center_state_fn: Callable[[float], tuple[np.ndarray, np.ndarray]],
        velocity_side: str = "plus",
        target: float = 0.0,
        weight: float = 1.0,
        active: bool = True,
    ) -> None:
        side = "plus" if velocity_side.lower() != "minus" else "minus"

        super().__init__(
            name="apse",
            patch_index=patch_index,
            eval_fn=lambda r_k, v_plus, v_minus, t_k: self._eval_fn(side, center_state_fn, r_k, v_plus, v_minus, t_k),
            target=target,
            weight=weight,
            jacobian_fn=lambda r_k, v_plus, v_minus, t_k: self._jac_fn(side, center_state_fn, r_k, v_plus, v_minus, t_k),
            fd_step=None,
            active=active,
        )

    def _eval_fn(self, side: str, center_state_fn: Callable[[float], tuple[np.ndarray, np.ndarray]], r_k: np.ndarray, v_plus: np.ndarray, v_minus: np.ndarray, t_k: float) -> float:
        R_BA, V_BA = center_state_fn(float(t_k))
        vB = v_plus if side == "plus" else v_minus
        r_rel = r_k - np.asarray(R_BA, dtype=float)
        v_rel = vB - np.asarray(V_BA, dtype=float)
        return float(r_rel @ v_rel)

    def _jac_fn(self, side: str, center_state_fn: Callable[[float], tuple[np.ndarray, np.ndarray]], r_k: np.ndarray, v_plus: np.ndarray, v_minus: np.ndarray, t_k: float) -> _ConstraintGrad:
        R_BA, V_BA = center_state_fn(float(t_k))
        vB = v_plus if side == "plus" else v_minus
        r_rel = r_k - np.asarray(R_BA, dtype=float)
        v_rel = vB - np.asarray(V_BA, dtype=float)
        d_r = v_rel
        d_v_plus = v_rel * 0.0 if side == "minus" else r_rel
        d_v_minus = r_rel if side == "minus" else r_rel * 0.0
        return _ConstraintGrad(d_r=d_r, d_v_plus=d_v_plus, d_v_minus=d_v_minus, d_t=0.0)
