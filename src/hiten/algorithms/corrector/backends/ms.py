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
    ):
        super().__init__(stepper_factory=None)
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
    ):
        super().__init__(stepper_factory=None)
        self._position_shooter = _PositionShooting(
            var_dynsys=var_dynsys,
            method=method,
            order=order,
            steps=steps,
        )
        self._var_dynsys = var_dynsys
        self._method = method
        self._order = order
        self._steps = steps

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

            b_tilde = - delta_v_vec.copy()

            if constraint_rows:
                CR = np.vstack(constraint_rows)
                if initial_position_fixed:
                    CR = CR[:, 4:]
                if final_position_fixed:
                    CR = CR[:, :-4]
                M_tilde = np.vstack([M_tilde, CR])
                b_tilde = np.concatenate([b_tilde, np.concatenate(constraint_rhs_list)])

            correction, *_ = np.linalg.lstsq(M_tilde, b_tilde, rcond=None)

            update_segments = list(range(segment_num + 1))
            index_offset = 0

            if initial_position_fixed:
                update_segments.remove(0)
                index_offset = -1
            if final_position_fixed and segment_num in update_segments:
                update_segments.remove(segment_num)

            # Apply correction
            for i in update_segments:
                dR, dt = self._extract_patch_correction(correction, i + index_offset)
                x_patches[i][:3] += dR
                t_patches[i] += dt

            if segment_num not in update_segments:
                _, dtN = self._extract_patch_correction(correction, segment_num + index_offset)
                t_patches[segment_num] += dtN

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
