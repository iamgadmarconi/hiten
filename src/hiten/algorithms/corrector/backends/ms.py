"""Multiple shooting correction algorithms for orbital mechanics.

This module provides position shooting and velocity correction algorithms
for multiple shooting methods in periodic orbit computation.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.types import (PositionInput, PositionOutput,
                                              StepperFactory, VelocityInput,
                                              VelocityOutput)
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.algorithms.corrector.constraints import _NodePartials, _ConstraintContext
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.algorithms.dynamics.base import _DynamicalSystem


def _get_A_block(stm):
    """Extract A block (position-position) from 6x6 STM."""
    return stm[:3, :3]

def _get_B_block(stm):
    """Extract B block (position-velocity) from 6x6 STM."""
    return stm[:3, 3:6]

def _get_C_block(stm):
    """Extract C block (velocity-position) from 6x6 STM."""
    return stm[3:6, :3]

def _get_D_block(stm):
    """Extract D block (velocity-velocity) from 6x6 STM."""
    return stm[3:6, 3:6]


class _PositionShooting(_CorrectorBackend):

    def __init__(
        self,
        *,
        var_dynsys: "_DynamicalSystem",
        method: str,
        order: int,
        steps: int,
        stepper_factory: StepperFactory | None = None,
    ):
        super().__init__(stepper_factory=stepper_factory)
        self._var_dynsys = var_dynsys
        self._method = method
        self._order = order
        self._steps = steps

    def run(self, request: PositionInput) -> PositionOutput:
        """Run position shooting to correct initial velocity.
        
        Iteratively adjusts the initial velocity to minimize the position
        error at the target time using the B block of the STM.
        
        Parameters
        ----------
        request : PositionInput
            Input containing initial/target states, times, and solver parameters
            
        Returns
        -------
        PositionOutput
            Output containing corrected states, STM, and convergence info
        """
        t_initial = request.t_initial
        x_initial = request.x_initial.copy()
        t_target = request.t_target
        x_target = request.x_target
        segment_num = request.segment_num
        norm_fn = request.norm_fn
        max_attempts = request.max_attempts
        tol = request.tol

        t_span = t_target - t_initial
        sigma = 0.618  # Golden ratio damping for quasi-Newton corrections

        metadata: dict[str, Any] = {
            "segment_num": segment_num,
            "convergence_history": [],
        }

        # Initialize variables for final return (in case of non-convergence)
        x_final = x_initial.copy()
        stm_final = np.eye(6)
        error_norm = float('inf')

        for iteration in range(max_attempts):
            # Propagate with STM
            x_traj, _, stm_final, _ = _compute_stm(
                dynsys=self._var_dynsys,
                x0=x_initial,
                tf=t_span,
                steps=self._steps,
                method=self._method,
                order=self._order,
            )
            x_final = x_traj[-1, :]

            # Compute position error
            error_final = x_target[:3] - x_final[:3]
            error_norm = self._compute_norm(error_final, norm_fn)

            # Compute velocity correction using B block
            B = _get_B_block(stm_final)
            delta_v = self._dVk_minus_one(B, error_final)

            # Apply damped correction (quasi-Newton, not gradient descent)
            x_initial[3:6] = x_initial[3:6] + sigma * delta_v

            metadata["convergence_history"].append(
                {
                    "iteration": iteration,
                    "error_norm": float(error_norm),
                }
            )

            if error_norm < tol:
                logger.debug(
                    "Position shooting converged at iteration %d/%d (segment %d, error=%.2e)",
                    iteration + 1, max_attempts, segment_num, error_norm
                )
                metadata["iterations"] = iteration
                return PositionOutput(
                    x0_corrected=x_initial,
                    xf_corrected=x_final,
                    stm_corrected=stm_final,
                    success=True,
                    metadata=metadata,
                )

        # Failed to converge
        metadata["iterations"] = max_attempts
        logger.warning(
            "Position shooting failed to converge after %d iterations (segment %d, error=%.2e)",
            max_attempts, segment_num, error_norm
        )
        
        return PositionOutput(
            x0_corrected=x_initial,
            xf_corrected=x_final,
            stm_corrected=stm_final,
            success=False,
            metadata=metadata,
        )

    def _dVk_minus_one(self, B_block: np.ndarray, position_error: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(B_block, position_error)
        except np.linalg.LinAlgError:
            Binv = np.linalg.pinv(B_block, rcond=1e-12)
            return Binv @ position_error


class _VelocityCorrection(_CorrectorBackend):
    """Level-2 velocity correction for multiple shooting.
    
    Eliminates velocity discontinuities at patch points by adjusting
    positions and times using the state relationship matrix.
    """

    def __init__(
        self,
        *,
        var_dynsys: "_DynamicalSystem",
        method: str = "adaptive",
        order: int = 8,
        steps: int = 2000,
        stepper_factory: StepperFactory | None = None,
        debug_jacobian: bool = False,
        enable_backtracking: bool = False,
        max_backtrack_steps: int = 4,
    ):
        super().__init__(stepper_factory=stepper_factory)
        self._position_shooter = _PositionShooting(
            var_dynsys=var_dynsys,
            method=method,
            order=order,
            steps=steps,
            stepper_factory=stepper_factory,
        )
        self._var_dynsys = var_dynsys
        self._method = method
        self._order = order
        self._steps = steps
        self._debug_jacobian = debug_jacobian
        self._enable_backtracking = enable_backtracking
        self._max_backtrack_steps = max_backtrack_steps

    def run(self, request: VelocityInput) -> VelocityOutput:
        """Run level-2 velocity correction.
        
        Parameters
        ----------
        request : VelocityInput
            Input containing patch states, times, STMs, etc.
            
        Returns
        -------
        VelocityOutput
            Output containing corrected states, times, and convergence info.
        """
        # Extract inputs
        t_patches = request.t_patches.copy()
        x_patches = [x.copy() for x in request.x_patches]
        dynsys_fn = request.dynsys_fn
        vel_max_attempts = request.vel_max_attempts
        pos_max_attempts = request.pos_max_attempts
        pos_tol = request.pos_tol
        vel_tol = request.vel_tol
        pos_norm_fn = request.pos_norm_fn
        initial_position_fixed = request.initial_position_fixed
        final_position_fixed = request.final_position_fixed
        segment_num = request.segment_num
        constraints = request.constraints

        stms = np.zeros((segment_num, 6, 6))
        xf_patches = np.zeros((segment_num, 6))

        metadata: dict[str, Any] = {
            "convergence_history": [],
        }

        error_norm = float('inf')

        for iteration in range(vel_max_attempts):
            for seg in range(segment_num):
                pos_request = PositionInput(
                    t_initial=t_patches[seg],
                    x_initial=x_patches[seg],
                    t_target=t_patches[seg + 1],
                    x_target=x_patches[seg + 1],
                    segment_num=seg + 1,
                    max_attempts=pos_max_attempts,
                    tol=pos_tol,
                    norm_fn=pos_norm_fn,

                )
                pos_output = self._position_shooter.run(pos_request)
                x_patches[seg] = pos_output.x0_corrected
                xf_patches[seg] = pos_output.xf_corrected
                stms[seg] = pos_output.stm_corrected
                if not pos_output.success:
                    # Keep going; constraints/Level-2 may recover the failing segment
                    logger.warning(
                        "Level-1 failed at segment %d (error persisted); continuing to Level-2 update",
                        seg + 1,
                    )

            # Use paper's orientation: ΔV_k = V_k^+ − V_k^-
            delta_v_list = [
                x_patches[i + 1][3:6] - xf_patches[i][3:6]
                for i in range(segment_num - 1)
            ]
            delta_v_vec = np.concatenate(delta_v_list) if delta_v_list else np.zeros(0)
            
            # Store base residual for diagnostics (before constraint augmentation)
            b_base = delta_v_vec.copy()

            error_norm = float(np.linalg.norm(delta_v_vec))

            metadata["convergence_history"].append({
                "iteration": iteration + 1,
                "velocity_error_norm": error_norm,
            })

            if error_norm < vel_tol:
                logger.info(
                    "Velocity correction converged at iteration %d/%d (error=%.2e)",
                    iteration + 1, vel_max_attempts, error_norm
                )
                metadata["iterations"] = iteration + 1
                return VelocityOutput(
                    x_corrected=x_patches,
                    t_corrected=list(t_patches),
                    success=True,
                    metadata=metadata,
                )

            # Build base state relationship matrix for ΔV continuity
            n_rows = (segment_num - 2) * 3 + 3
            n_cols = (segment_num - 2) * 4 + 12
            base_M = np.zeros((n_rows, n_cols))

            node_cache: list[dict[str, np.ndarray]] = [None] * (segment_num + 1)

            for i in range(1, segment_num):
                block = self._build_relationship_matrix(stms, x_patches, xf_patches, t_patches, i, dynsys_fn)

                row_start = (i - 1) * 3
                col_start = (i - 1) * 4
                base_M[row_start:row_start + 3, col_start:col_start + 12] = block

                Ab, Bb, Db, Af, Bf, Df, v_km1_plus, v_k_minus, v_k_plus, v_kp1_minus, a_k_plus, a_k_minus = self._build_params(stms, i, x_patches, xf_patches, t_patches, dynsys_fn)

                node_cache[i] = {
                    "A_b": Ab, "B_b": Bb, "D_b": Db,
                    "A_f": Af, "B_f": Bf, "D_f": Df,
                    "v_km1_plus": v_km1_plus, "v_k_minus": v_k_minus,
                    "v_k_plus": v_k_plus, "v_kp1_minus": v_kp1_minus,
                    "a_k_plus": a_k_plus, "a_k_minus": a_k_minus,
                }

            # Add terminal node (index = segment_num) with backward blocks (N → N−1)
            if segment_num >= 2:
                stm_last_fwd = stms[segment_num - 1]
                try:
                    stm_last_bwd = np.linalg.inv(stm_last_fwd)
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse as a fallback; extract blocks compatibly
                    stm_last_bwd = np.linalg.pinv(stm_last_fwd, rcond=1e-12)

                AbN = _get_A_block(stm_last_bwd)
                BbN = _get_B_block(stm_last_bwd)
                DbN = _get_D_block(stm_last_bwd)

                # Kinematics at terminal node
                v_km1_plus_N = x_patches[segment_num - 1][3:6]             # v_{N-1}^+
                v_k_minus_N = xf_patches[segment_num - 1][3:6]             # v_N^-
                v_k_plus_N = x_patches[segment_num][3:6]                    # v_N^+
                v_kp1_minus_N = v_k_minus_N                                 # no N+1; duplicate

                a_k_minus_term = dynsys_fn(t_patches[segment_num], xf_patches[segment_num - 1])[3:6]
                a_k_plus_term = dynsys_fn(t_patches[segment_num], x_patches[segment_num])[3:6]

                node_cache[segment_num] = {
                    "A_b": AbN, "B_b": BbN, "D_b": DbN,
                    "A_f": np.zeros((3, 3)), "B_f": np.zeros((3, 3)), "D_f": np.zeros((3, 3)),
                    "v_km1_plus": v_km1_plus_N, "v_k_minus": v_k_minus_N,
                    "v_k_plus": v_k_plus_N, "v_kp1_minus": v_kp1_minus_N,
                    "a_k_plus": a_k_plus_term, "a_k_minus": a_k_minus_term,
                }

            # Build augmented matrix with constraints
            constraint_rows: list[np.ndarray] = []
            constraint_rhs_list: list[np.ndarray] = []

            if constraints:
                node_partials_map: dict[int, _NodePartials] = {}

                for i in range(1, segment_num):

                    cache = node_cache[i]

                    if cache is None:
                        continue
                    node_partials_map[i] = _NodePartials(
                        A_backward=cache["A_b"], B_backward=cache["B_b"], D_backward=cache["D_b"],
                        A_forward=cache["A_f"], B_forward=cache["B_f"], D_forward=cache["D_f"],
                        v_km1_plus=cache["v_km1_plus"], v_k_minus=cache["v_k_minus"],
                        v_k_plus=cache["v_k_plus"], v_kp1_minus=cache["v_kp1_minus"],
                        a_k_plus=cache["a_k_plus"], a_k_minus=cache["a_k_minus"],
                    )

                # Add terminal entry for k = N with backward blocks
                term_cache = node_cache[segment_num]
                node_partials_map[segment_num] = _NodePartials(
                    A_backward=term_cache["A_b"], B_backward=term_cache["B_b"], D_backward=term_cache["D_b"],
                    A_forward=term_cache["A_f"], B_forward=term_cache["B_f"], D_forward=term_cache["D_f"],
                    v_km1_plus=term_cache["v_km1_plus"], v_k_minus=term_cache["v_k_minus"],
                    v_k_plus=term_cache["v_k_plus"], v_kp1_minus=term_cache["v_kp1_minus"],
                    a_k_plus=term_cache["a_k_plus"], a_k_minus=term_cache["a_k_minus"],
                )

                ctx = _ConstraintContext(
                    x_patches=x_patches,
                    xf_patches=xf_patches,
                    t_patches=t_patches,
                    stms=stms,
                    node_partials=node_partials_map,
                    segment_num=segment_num,
                )

                for c in constraints:
                    rows, rhs = c.build_rows(ctx)
                    if rows is None or np.size(rows) == 0:
                        continue
                    constraint_rows.append(np.asarray(rows))
                    constraint_rhs_list.append(np.asarray(rhs, dtype=float).reshape(-1))

            # Apply boundary trimming to columns consistently
            M_tilde = base_M
            if initial_position_fixed:
                M_tilde = M_tilde[:, 4:]
            if final_position_fixed:
                M_tilde = M_tilde[:, :-4]

            b_tilde = delta_v_vec.copy()

            if constraint_rows:
                CR = np.vstack(constraint_rows)
                if initial_position_fixed:
                    CR = CR[:, 4:]
                if final_position_fixed:
                    CR = CR[:, :-4]
                M_tilde = np.vstack([M_tilde, CR])
                b_tilde = np.concatenate([b_tilde, np.concatenate(constraint_rhs_list)])

            if self._debug_jacobian:
                cond_M_tilde = np.linalg.cond(M_tilde)
                singular_values = np.linalg.svd(M_tilde, compute_uv=False)
                s_max = singular_values[0]
                s_min = singular_values[-1]
                s_ratio = s_min / s_max if s_max > 0 else 0.0
                print(f"cond(M_tilde) = {cond_M_tilde:.2e} (shape: {M_tilde.shape})")
                print(f"  s_min = {s_min:.2e}, s_max = {s_max:.2e}")
                print(f"  s_min/s_max = {s_ratio:.2e}")

            correction, *_ = np.linalg.lstsq(M_tilde, b_tilde, rcond=None)

            # Debug Jacobian if enabled
            if self._debug_jacobian:
                self._run_jacobian_diagnostics(
                    M_tilde=M_tilde,
                    b_tilde=b_tilde,
                    b_base=b_base,
                    correction=correction,
                    x_patches=x_patches,
                    t_patches=t_patches,
                    stms=stms,
                    xf_patches=xf_patches,
                    segment_num=segment_num,
                    initial_position_fixed=initial_position_fixed,
                    final_position_fixed=final_position_fixed,
                    pos_max_attempts=pos_max_attempts,
                    pos_tol=pos_tol,
                    pos_norm_fn=pos_norm_fn,
                    dynsys_fn=dynsys_fn,
                )

            # Apply backtracking line search if enabled
            step_scale = 1.0
            if self._enable_backtracking:
                step_scale = self._backtracking_line_search(
                    correction=correction,
                    x_patches=x_patches,
                    t_patches=t_patches,
                    b_base=b_base,
                    error_norm_current=error_norm,
                    segment_num=segment_num,
                    initial_position_fixed=initial_position_fixed,
                    final_position_fixed=final_position_fixed,
                    pos_max_attempts=pos_max_attempts,
                    pos_tol=pos_tol,
                    pos_norm_fn=pos_norm_fn,
                )
                print(f"Backtracking: selected step scale = {step_scale:.4f}")
                print()

            update_segments = list(range(segment_num + 1))
            index_offset = 0

            if initial_position_fixed:
                update_segments.remove(0)
                index_offset = -1
            if final_position_fixed and segment_num in update_segments:
                update_segments.remove(segment_num)

            # Apply correction with step scaling
            for i in update_segments:
                dR, dt = self._extract_patch_correction(correction, i + index_offset)
                x_patches[i][:3] += step_scale * dR
                t_patches[i] += step_scale * dt

            if segment_num not in update_segments:
                _, dtN = self._extract_patch_correction(correction, segment_num + index_offset)
                t_patches[segment_num] += step_scale * dtN

        # Failed to converge
        metadata["iterations"] = vel_max_attempts
        logger.warning(
            "Velocity correction failed to converge after %d iterations (error=%.2e)",
            vel_max_attempts, error_norm
        )

        return VelocityOutput(
            x_corrected=x_patches,
            t_corrected=list(t_patches),
            success=False,
            metadata=metadata,
        )

    def _compute_residual_vector(
        self,
        x_patches: list[np.ndarray],
        t_patches: np.ndarray,
        segment_num: int,
        pos_max_attempts: int,
        pos_tol: float,
        pos_norm_fn,
    ) -> np.ndarray:
        """Compute the residual vector (velocity discontinuities) via Level-1 sweep.
        
        Parameters
        ----------
        x_patches : list
            Current patch states
        t_patches : ndarray
            Current patch times
        segment_num : int
            Number of segments
        pos_max_attempts : int
            Max iterations for position shooting
        pos_tol : float
            Position shooting tolerance
        pos_norm_fn : callable
            Norm function for position shooting
            
        Returns
        -------
        b : ndarray
            Velocity discontinuity vector (stacked ΔV at interior junctions)
        """
        xf_patches = np.zeros((segment_num, 6))
        x0_corrected_list = [x.copy() for x in x_patches]

        for seg in range(segment_num):
            pos_request = PositionInput(
                t_initial=t_patches[seg],
                x_initial=x_patches[seg].copy(),
                t_target=t_patches[seg + 1],
                x_target=x_patches[seg + 1],
                segment_num=seg + 1,
                max_attempts=pos_max_attempts,
                tol=pos_tol,
                norm_fn=pos_norm_fn,
            )
            pos_output = self._position_shooter.run(pos_request)
            # Capture corrected initial and terminal states
            x0_corrected_list[seg] = pos_output.x0_corrected
            xf_patches[seg] = pos_output.xf_corrected
        
        # Compute velocity discontinuities
        # Paper's orientation: ΔV_k = V_k^+ − V_k^-
        delta_v_list = [
            x0_corrected_list[i + 1][3:6] - xf_patches[i][3:6]
            for i in range(segment_num - 1)
        ]
        return np.concatenate(delta_v_list) if delta_v_list else np.zeros(0)

    def _run_jacobian_diagnostics(
        self,
        M_tilde: np.ndarray,
        b_tilde: np.ndarray,
        b_base: np.ndarray,
        correction: np.ndarray,
        x_patches: list[np.ndarray],
        t_patches: np.ndarray,
        stms: np.ndarray,
        xf_patches: np.ndarray,
        segment_num: int,
        initial_position_fixed: bool,
        final_position_fixed: bool,
        pos_max_attempts: int,
        pos_tol: float,
        pos_norm_fn,
        dynsys_fn,
    ):
        """Run comprehensive Jacobian diagnostics.
        
        Note: We use b_base (velocity discontinuities only) for comparisons,
        not b_tilde (which includes constraint RHS), since _compute_residual_vector
        only computes velocity discontinuities.
        """

        if not initial_position_fixed and not final_position_fixed and segment_num >= 3:
            print("\nFocused per-interface checks (3×12 blocks):")
            ks = [1, max(1, segment_num // 2), max(1, segment_num - 1)]
            # de-duplicate while keeping order
            seen = set()
            k_list = []
            for k in ks:
                if k not in seen:
                    seen.add(k)
                    k_list.append(k)
            for k in k_list:
                try:
                    self._check_interface_jacobian_block(
                        k=k,
                        stms=stms,
                        x_patches=x_patches,
                        xf_patches=xf_patches,
                        t_patches=t_patches,
                        segment_num=segment_num,
                        b_base=b_base,
                        pos_max_attempts=pos_max_attempts,
                        pos_tol=pos_tol,
                        pos_norm_fn=pos_norm_fn,
                        dynsys_fn=dynsys_fn,
                    )
                except Exception as exc:
                    print(f"  [k={k}] checker failed: {exc!r}")

        print("=" * 60)
        print()

    def _backtracking_line_search(
        self,
        correction: np.ndarray,
        x_patches: list[np.ndarray],
        t_patches: np.ndarray,
        b_base: np.ndarray,
        error_norm_current: float,
        segment_num: int,
        initial_position_fixed: bool,
        final_position_fixed: bool,
        pos_max_attempts: int,
        pos_tol: float,
        pos_norm_fn,
    ) -> float:
        """Perform backtracking line search to find safe step size.
        
        Parameters
        ----------
        b_base : ndarray
            Current base residual (velocity discontinuities)
        error_norm_current : float
            Current error norm (should equal ||b_base||)
        
        Returns
        -------
        step_scale : float
            Scale factor in (0, 1] for the correction step
        """

        best_scale = 1.0
        best_error = float('inf')
        
        for step_num in range(self._max_backtrack_steps):
            gamma = 2.0 ** (-step_num)  # 1, 1/2, 1/4, 1/8, ...
            
            # Apply scaled correction
            x_test = [x.copy() for x in x_patches]
            t_test = t_patches.copy()
            
            update_segments = list(range(segment_num + 1))
            index_offset = 0
            if initial_position_fixed:
                update_segments.remove(0)
                index_offset = -1
            if final_position_fixed and segment_num in update_segments:
                update_segments.remove(segment_num)
            
            for i in update_segments:
                dR, dt = self._extract_patch_correction(correction, i + index_offset)
                x_test[i][:3] += gamma * dR
                t_test[i] += gamma * dt
            
            if segment_num not in update_segments:
                _, dtN = self._extract_patch_correction(correction, segment_num + index_offset)
                t_test[segment_num] += gamma * dtN
            
            # Evaluate error
            b_new = self._compute_residual_vector(
                x_test, t_test, segment_num,
                pos_max_attempts, pos_tol, pos_norm_fn
            )
            error_new = float(np.linalg.norm(b_new))
                        
            if error_new < best_error:
                best_error = error_new
                best_scale = gamma
            
            # Accept if we get sufficient decrease
            if error_new < 0.99 * error_norm_current:
                return gamma
        
        return best_scale

    def _build_params(self, stms, iteration, x_patches, xf_patches, t_patches, dynsys_fn):

        stm21 = stms[iteration - 1]   # (k-1) -> k (forward)
        stm32 = stms[iteration]       # k -> (k+1) (forward)

        # Left (paper's k-1,k) blocks from the backward STM: k -> (k-1)
        try:
            stm12 = np.linalg.inv(stm21)
        except np.linalg.LinAlgError:
            stm12 = np.linalg.pinv(stm21, rcond=1e-12)

        Bkkm1 = _get_B_block(stm21)

        Ab = _get_A_block(stm12)  # A_{k-1,k}
        Bb = _get_B_block(stm12)  # B_{k-1,k}
        Cb = _get_C_block(stm12)  # C_{k-1,k}
        Db = _get_D_block(stm12)  # D_{k-1,k}
        
        # Right (paper's k+1,k) blocks from the forward STM: k -> (k+1)
        Af = _get_A_block(stm32)  # A_{k+1,k}
        Bf = _get_B_block(stm32)  # B_{k+1,k}
        Cf = _get_C_block(stm32)  # C_{k+1,k}
        Df = _get_D_block(stm32)  # D_{k+1,k}

        def _inv(matrix):
            try:
                return np.linalg.solve(matrix, np.eye(3))
            except np.linalg.LinAlgError:
                return np.linalg.pinv(matrix, rcond=1e-12)

        assert np.allclose(Bkkm1, _inv((Cb - Db @ (_inv(Bb) @ Ab))))

        # Extract velocities at the three nodes (consistent with ΔV_k = V_k^+ − V_k^-)
        v_km1_plus = x_patches[iteration - 1][3:6]   # v_{k-1}^+
        v_k_minus = xf_patches[iteration - 1][3:6]   # v_k^- (end of segment k-1)
        v_k_plus = x_patches[iteration][3:6]         # v_k^+ (start of segment k)
        v_kp1_minus = xf_patches[iteration][3:6]     # v_{k+1}^- (end of segment k)

        # Compute accelerations at node k
        a_k_minus = dynsys_fn(t_patches[iteration], xf_patches[iteration - 1])[3:6]  # a_k^-
        a_k_plus = dynsys_fn(t_patches[iteration], x_patches[iteration])[3:6]        # a_k^+

        return Ab, Bb, Db, Af, Bf, Df, v_km1_plus, v_k_minus, v_k_plus, v_kp1_minus, a_k_plus, a_k_minus


    def _build_relationship_matrix(self, stms, x_patches, xf_patches, t_patches, iteration, dynsys_fn):
        Ab, Bb, Db, Af, Bf, Df, v_km1_plus, v_k_minus, v_k_plus, v_kp1_minus, a_k_plus, a_k_minus = self._build_params(stms, iteration, x_patches, xf_patches, t_patches, dynsys_fn)

        block = self._build_M_matrix(
            B_backward=Bb,
            A_backward=Ab,
            D_backward=Db,
            B_forward=Bf,
            A_forward=Af,
            D_forward=Df,
            v_km1_plus=v_km1_plus,
            v_k_minus=v_k_minus,
            v_k_plus=v_k_plus,
            v_kp1_minus=v_kp1_minus,
            a_k_plus=a_k_plus,
            a_k_minus=a_k_minus,
        )

        return block

    def _B_inv(self, B_block):
        try:
            return np.linalg.solve(B_block, np.eye(3))
        except np.linalg.LinAlgError:
            return np.linalg.pinv(B_block, rcond=1e-12)

    def _dVk_dRk1(self, B_block):
        return self._B_inv(B_block)

    def _dVk_dRk(self, B_block, A_block):
        B_inv = self._B_inv(B_block)
        return - B_inv @ A_block

    def _dVk_dtk1(self, B_block, v_k):
        B_inv = self._B_inv(B_block)
        return - B_inv @ v_k

    def _dVk_dtk(self, a_k, D_block, B_block, v_k):
        B_inv = self._B_inv(B_block)
        return a_k - D_block @ B_inv @ v_k

    def _build_M_matrix(
        self,
        B_backward,
        A_backward,
        D_backward,
        B_forward,
        A_forward,
        D_forward,
        v_km1_plus,
        v_k_minus,
        v_k_plus,
        v_kp1_minus,
        a_k_plus,
        a_k_minus,
    ):
        # Backward: k-1, k
        # Forward: k+1, k
        B_bwd_inv = self._B_inv(B_backward) # B_{k-1,k}^{-1}
        B_fwd_inv = self._B_inv(B_forward) # B_{k+1,k}^{-1}

        col_Rkm1 = -B_bwd_inv # - B_{k-1,k}^{-1} ; dR_{k-1}
        col_tkm1 = (B_bwd_inv @ v_km1_plus).reshape(-1, 1) # B_{k-1,k}^{-1} v_{k-1}^+ ; dt_{k-1}

        col_Rk = B_bwd_inv @ A_backward - B_fwd_inv @ A_forward # B_{k-1,k}^{-1} A_{k-1,k} - B_{k+1,k}^{-1} A_{k+1,k} ; dR_{k}
        col_tk = ((a_k_plus - a_k_minus) - B_bwd_inv @ (A_backward @ v_k_minus) + B_fwd_inv @ (A_forward @ v_k_plus)).reshape(-1, 1) # a_{k}^+ - a_{k}^- - B_{k-1,k}^{-1} A_{k-1,k} v_{k}^- + B_{k+1,k}^{-1} A_{k+1,k} v_{k}^+ ; dt_{k}

        col_Rkp1 = B_fwd_inv # B_{k+1,k}^{-1} ; dR_{k+1}
        col_tkp1 = (-B_fwd_inv @ v_kp1_minus).reshape(-1, 1) # - B_{k+1,k}^{-1} v_{k+1}^- ; dt_{k+1}

        M = np.hstack([
            col_Rkm1,
            col_tkm1,
            col_Rk,
            col_tk,
            col_Rkp1,
            col_tkp1,
        ])

        return M


    def _extract_patch_correction(
        self,
        correction_vector: np.ndarray,
        patch_index: int,
    ) -> tuple[np.ndarray, float]:
        """Extract position and time corrections for a single patch from solution vector.
        
        Parameters
        ----------
        correction_vector : ndarray
            Full correction vector from solving the multiple shooting system
        patch_index : int
            Index of the patch to extract corrections for
            
        Returns
        -------
        dR : ndarray (3,)
            Position correction for the patch
        dt : float
            Time correction for the patch
        """
        base = patch_index * 4
        dR = correction_vector[base:base + 3]
        dt = correction_vector[base + 3]
        return dR, dt

    def _check_interface_jacobian_block(
        self,
        *,
        k: int,
        stms: np.ndarray,
        x_patches: list[np.ndarray],
        xf_patches: np.ndarray,
        t_patches: np.ndarray,
        segment_num: int,
        b_base: np.ndarray,
        pos_max_attempts: int,
        pos_tol: float,
        pos_norm_fn,
        dynsys_fn,
    ) -> None:
        """Compare analytic vs FD 3x12 block at interface k.

        Assumes no boundary trimming. Prints per-column relative errors and cosine similarity.
        """
        if not (1 <= k <= segment_num - 1):
            return

        block_analytic = self._build_relationship_matrix(
            stms, x_patches, xf_patches, t_patches, k, dynsys_fn
        )

        n_cols_total = (segment_num - 2) * 4 + 12
        col_start = (k - 1) * 4
        row_start = (k - 1) * 3
        eps = 1e-6

        block_fd = np.zeros((3, 12))
        for local_j in range(12):
            e = np.zeros(n_cols_total)
            e[col_start + local_j] = eps

            x_pert = [x.copy() for x in x_patches]
            t_pert = t_patches.copy()

            for patch_idx in range(segment_num + 1):
                dR, dt = self._extract_patch_correction(e, patch_idx)
                x_pert[patch_idx][:3] += dR
                t_pert[patch_idx] += dt

            b_new = self._compute_residual_vector(
                x_pert, t_pert, segment_num,
                pos_max_attempts, pos_tol, pos_norm_fn,
            )

            db = (b_new - b_base) / eps
            block_fd[:, local_j] = db[row_start:row_start + 3]

        block_signflip = -block_analytic

        def _per_col_stats(Ja: np.ndarray, Jn: np.ndarray):
            rels = []
            coss = []
            for j in range(Ja.shape[1]):
                a = Ja[:, j]
                n = Jn[:, j]
                na = float(np.linalg.norm(a))
                nn = float(np.linalg.norm(n))
                rel = float(np.linalg.norm(a - n) / (na if na > 1e-15 else max(nn, 1e-15)))
                cos = float((a @ n) / (na * nn)) if na > 1e-15 and nn > 1e-15 else 1.0
                rels.append(rel)
                coss.append(cos)
            return rels, coss

        rel_A, cos_A = _per_col_stats(block_analytic, block_fd)
        rel_S, cos_S = _per_col_stats(block_signflip, block_fd)

        def _fmt(vals):
            return ", ".join(f"{v:.2e}" for v in vals)

        print(f"Interface k={k}: per-column FD comparison (3x12 block)")
        print("  Analytic vs FD (this code):")
        print(f"    rel_err: [{_fmt(rel_A)}]")
        print(f"    cos_sim: [{_fmt(cos_A)}]")
        print("  Sign-flipped analytic vs FD:")
        print(f"    rel_err: [{_fmt(rel_S)}]")
        print(f"    cos_sim: [{_fmt(cos_S)}]")

        mean_rel_A = float(np.mean(rel_A))
        mean_rel_S = float(np.mean(rel_S))
        if mean_rel_S + 1e-3 < mean_rel_A:
            print("  → Hint: residual sign/orientation likely reversed at this interface.")
        elif mean_rel_A < 1e-2:
            print("  → OK: block matches FD (≤1e-2).")
        else:
            print("  → Mismatch persists: check column ordering and forward/backward STM usage.")

        # Additional targeted checks for center block orientation (R_k and t_k)
        try:
            # Numeric center group from FD
            fd_Rk = block_fd[:, 4:7]  # 3x3
            fd_tk = block_fd[:, 7]    # 3-vector

            # Build both variants for the left contribution
            # Current (uses backward-left blocks: A_{k-1,k}, B_{k-1,k})
            Ab, Bb, Db, Af, Bf, Df, v_km1_plus, v_k_minus, v_k_plus, v_kp1_minus, a_k_plus, a_k_minus = self._build_params(
                stms, k, x_patches, xf_patches, t_patches, dynsys_fn
            )
            B_bwd_inv = self._B_inv(Bb)
            B_fwd_inv = self._B_inv(Bf)
            Rk_current = B_bwd_inv @ Ab - B_fwd_inv @ Af
            tk_current = (a_k_plus - a_k_minus) - (B_bwd_inv @ (Ab @ v_k_minus)) + (B_fwd_inv @ (Af @ v_k_plus))

            # Eq.(19) identity check for left blocks: B_{k,k-1} = (C_{k-1,k} - D_{k-1,k} B_{k-1,k}^{-1} A_{k-1,k})^{-1}
            # Forward-left blocks (k,k-1)
            stm_left_fwd = stms[k - 1]
            A_lf = _get_A_block(stm_left_fwd)
            B_lf = _get_B_block(stm_left_fwd)
            C_lf = _get_C_block(stm_left_fwd)
            D_lf = _get_D_block(stm_left_fwd)
            B_lf_inv = self._B_inv(B_lf)

            # Backward-left blocks (k-1,k) from inverse STM
            try:
                stm_left_bwd = np.linalg.inv(stm_left_fwd)
            except np.linalg.LinAlgError:
                stm_left_bwd = np.linalg.pinv(stm_left_fwd, rcond=1e-12)
            Ab_b = _get_A_block(stm_left_bwd)
            Bb_b = _get_B_block(stm_left_bwd)
            Cb_b = _get_C_block(stm_left_bwd)
            Db_b = _get_D_block(stm_left_bwd)

            def _fro_rel(a, b):
                na = float(np.linalg.norm(a))
                return float(np.linalg.norm(a - b) / (na if na > 1e-15 else 1.0))

            # Build B via Eq.(19) from backward blocks
            try:
                B_from_19 = np.linalg.inv(Cb_b - Db_b @ (self._B_inv(Bb_b) @ Ab_b))
            except np.linalg.LinAlgError:
                B_from_19 = np.linalg.pinv(Cb_b - Db_b @ (self._B_inv(Bb_b) @ Ab_b), rcond=1e-12)
            rel_B19 = _fro_rel(B_lf, B_from_19)
            print(f"  Left Eq.(19) check: rel||B_{'{'}k,k-1{'}'} − (C_b − D_b B_b^{-1} A_b)^{-1}|| = {rel_B19:.2e}")

            # Alternative using forward-left blocks (A_{k,k-1}, B_{k,k-1}, D_{k,k-1})
            Rk_alt = - B_lf_inv @ A_lf - B_fwd_inv @ Af
            tk_alt_BA = (a_k_plus - a_k_minus) - (B_lf_inv @ (A_lf @ v_k_minus)) + (B_fwd_inv @ (Af @ v_k_plus))
            tk_alt_D = (a_k_plus - a_k_minus) + (D_lf @ (B_lf_inv @ v_k_minus)) - (Df @ (B_fwd_inv @ v_k_plus))

            def _rel(a, b):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                return float(np.linalg.norm(a - b) / (nb if nb > 1e-15 else 1.0))
            def _cos_mat(a, b):
                a_f = a.reshape(-1)
                b_f = b.reshape(-1)
                na = float(np.linalg.norm(a_f))
                nb = float(np.linalg.norm(b_f))
                return float((a_f @ b_f) / (na * nb)) if na > 1e-15 and nb > 1e-15 else 1.0

            rel_Rk_current = _rel(Rk_current, fd_Rk)
            rel_Rk_alt = _rel(Rk_alt, fd_Rk)
            rel_tk_current = _rel(tk_current, fd_tk)
            rel_tk_alt_BA = _rel(tk_alt_BA, fd_tk)
            rel_tk_alt_D = _rel(tk_alt_D, fd_tk)

            print(f"  Center R_k block: rel(current)={rel_Rk_current:.2e}, rel(alt Bfwd)={rel_Rk_alt:.2e}")
            print(f"  Center t_k col:   rel(current B^{-1}A)={rel_tk_current:.2e}, rel(alt B^{-1}A left-fwd)={rel_tk_alt_BA:.2e}, rel(D-form)={rel_tk_alt_D:.2e}")

            # Acceleration-side variants (which side to use for ā_k^± at t_k)
            a_right = a_k_plus  # evaluated at x_patches[k]
            a_left = a_k_minus  # evaluated at xf_patches[k-1]

            # B^{-1}A form variants
            tk_hybrid_BA = (a_right - a_left) - (B_bwd_inv @ (Ab @ v_k_minus)) + (B_fwd_inv @ (Af @ v_k_plus))
            tk_right_BA = (a_right - a_right) - (B_bwd_inv @ (Ab @ v_k_minus)) + (B_fwd_inv @ (Af @ v_k_plus))
            tk_left_BA = (a_left - a_left) - (B_bwd_inv @ (Ab @ v_k_minus)) + (B_fwd_inv @ (Af @ v_k_plus))

            # D-form variants (use forward-left/right D blocks)
            stm_left_fwd = stms[k - 1]
            D_lf = _get_D_block(stm_left_fwd)
            B_lf = _get_B_block(stm_left_fwd)
            B_lf_inv = self._B_inv(B_lf)
            tk_hybrid_D = (a_right - a_left) + (D_lf @ (B_lf_inv @ v_k_minus)) - (Df @ (B_fwd_inv @ v_k_plus))
            tk_right_D = (a_right - a_right) + (D_lf @ (B_lf_inv @ v_k_minus)) - (Df @ (B_fwd_inv @ v_k_plus))
            tk_left_D = (a_left - a_left) + (D_lf @ (B_lf_inv @ v_k_minus)) - (Df @ (B_fwd_inv @ v_k_plus))

            rel_tk_hybrid_BA = _rel(tk_hybrid_BA, fd_tk)
            rel_tk_right_BA = _rel(tk_right_BA, fd_tk)
            rel_tk_left_BA = _rel(tk_left_BA, fd_tk)
            rel_tk_hybrid_D = _rel(tk_hybrid_D, fd_tk)
            rel_tk_right_D = _rel(tk_right_D, fd_tk)
            rel_tk_left_D = _rel(tk_left_D, fd_tk)

            print(
                f"  Center t_k accel variants (B^{-1}A): hybrid={rel_tk_hybrid_BA:.2e}, right-only={rel_tk_right_BA:.2e}, left-only={rel_tk_left_BA:.2e}"
            )
            print(
                f"  Center t_k accel variants (D-form):  hybrid={rel_tk_hybrid_D:.2e}, right-only={rel_tk_right_D:.2e}, left-only={rel_tk_left_D:.2e}"
            )

            # Center R_k per-side magnitudes and cosine vs FD
            L = B_bwd_inv @ Ab
            R = - B_fwd_inv @ Af
            sum_LR = L + R
            nL, nR, nSum, nFD = (float(np.linalg.norm(L)), float(np.linalg.norm(R)), float(np.linalg.norm(sum_LR)), float(np.linalg.norm(fd_Rk)))
            cos_LR = _cos_mat(sum_LR, fd_Rk)
            print(f"  Center R_k mags: ||L||={nL:.2e}, ||R||={nR:.2e}, ||L+R||={nSum:.2e}, ||FD||={nFD:.2e}, cos(L+R,FD)={cos_LR:+.2f}")
        except Exception as _exc_center:
            print(f"  [center-block check skipped: {_exc_center!r}]")

        # Right-block mapping/sign diagnostics (R_{k+1}, t_{k+1})
        try:
            fd_Rkp1 = block_fd[:, 8:11]
            fd_tkp1 = block_fd[:, 11]
            fd_Rk = block_fd[:, 4:7]
            fd_Rkm1 = block_fd[:, 0:3]
            fd_tk = block_fd[:, 7]
            fd_tkm1 = block_fd[:, 3]

            Ab, Bb, Db, Af, Bf, Df, v_km1_plus, v_k_minus, v_k_plus, v_kp1_minus, a_k_plus, a_k_minus = self._build_params(
                stms, k, x_patches, xf_patches, t_patches, dynsys_fn
            )
            B_fwd_inv = self._B_inv(Bf)
            Rkp1_theory = B_fwd_inv
            tkp1_theory = -(B_fwd_inv @ v_kp1_minus)
            tkp1_pos = +(B_fwd_inv @ v_kp1_minus)

            def _rel(a, b):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                return float(np.linalg.norm(a - b) / (nb if nb > 1e-15 else 1.0))
            def _cos(a, b):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                return float((a.reshape(-1) @ b.reshape(-1)) / (na * nb)) if na > 1e-15 and nb > 1e-15 else 1.0

            # R_{k+1} placement
            rel_R_right = _rel(Rkp1_theory, fd_Rkp1)
            rel_R_center = _rel(Rkp1_theory, fd_Rk)
            rel_R_left = _rel(Rkp1_theory, fd_Rkm1)
            # Also report cosine for R block (Frobenius)
            def _cos_mat(a, b):
                a_f = a.reshape(-1)
                b_f = b.reshape(-1)
                na = float(np.linalg.norm(a_f))
                nb = float(np.linalg.norm(b_f))
                return float((a_f @ b_f) / (na * nb)) if na > 1e-15 and nb > 1e-15 else 1.0
            cos_R_right = _cos_mat(Rkp1_theory, fd_Rkp1)
            print(f"  Right R_{'{'}k+1{'}'} block rel: right={rel_R_right:.2e} (cos={cos_R_right:+.2f}), center={rel_R_center:.2e}, left={rel_R_left:.2e}")

            # t_{k+1} sign/placement
            rel_t_right = _rel(tkp1_theory, fd_tkp1)
            rel_t_right_pos = _rel(tkp1_pos, fd_tkp1)
            rel_t_center = _rel(tkp1_theory, fd_tk)
            rel_t_left = _rel(tkp1_theory, fd_tkm1)
            cos_t_right = _cos(tkp1_theory, fd_tkp1)
            print(f"  Right t_{'{'}k+1{'}'} col rel: right={rel_t_right:.2e} (cos={cos_t_right:+.2f}), right(+sign)={rel_t_right_pos:.2e}, center={rel_t_center:.2e}, left={rel_t_left:.2e}")
        except Exception as _exc_right:
            print(f"  [right-block check skipped: {_exc_right!r}]")