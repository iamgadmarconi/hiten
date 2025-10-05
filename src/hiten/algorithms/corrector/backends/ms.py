from re import I
from typing import TYPE_CHECKING, Any

import numpy as np

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.types import (
    PositionInput,
    PositionOutput,
    StepperFactory,
    VelocityInput,
    VelocityOutput,
)
from hiten.algorithms.dynamics.rtbp import _compute_stm

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
        """Run position shooting to correct initial velocity."""
        t_initial = request.t_initial
        x_initial = request.x_initial.copy()
        t_target = request.t_target
        x_target = request.x_target
        segment_num = request.segment_num
        norm_fn = request.norm_fn
        max_attempts = request.max_attempts
        tol = request.tol

        # Use default L2 norm if none provided
        if norm_fn is None:
            norm_fn = lambda r: float(np.linalg.norm(r))

        t_span = t_target - t_initial
        sigma = 0.618

        metadata = {
            "segment_num": segment_num,
            "iterations": 0,
            "convergence_history": [],
        }

        success = False

        for iteration in range(max_attempts):

            x_traj, _, stm_final, _ = _compute_stm(
                dynsys=self._var_dynsys,
                x0=x_initial,
                tf=t_span,
                steps=self._steps,
                method=self._method,
                order=self._order,
            )
            x_final = x_traj[-1, :]

            error_final = x_target[:3] - x_final[:3]
            error_norm = self._compute_norm(error_final, norm_fn)

            B = _get_B_block(stm_final)
            delta_v = self._dVk_minus_one(B, error_final)

            x_initial[3:6] = x_initial[3:6] + sigma * delta_v

            metadata["convergence_history"].append(
                {
                    "iteration": iteration,
                    "error_norm": float(error_norm),
                }
            )

            if error_norm < tol:
                print(f"POSITION SHOOTING: success at iteration {iteration + 1} of {max_attempts} | Residual error: {error_norm}")
                success = True
                break

        metadata["iterations"] = max_attempts

        return PositionOutput(
            x0_corrected=x_initial,
            xf_corrected=x_final,
            stm_corrected=stm_final,
            success=success,
            metadata=metadata,
        )

    def _dVk_minus_one(self, B_block: np.ndarray, position_error: np.ndarray) -> np.ndarray:
        return np.linalg.solve(B_block, position_error)


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
        position_tol = request.position_tol
        max_attempts = request.max_attempts
        tol = request.tol
        norm_fn = request.norm_fn
        initial_position_fixed = request.initial_position_fixed
        final_position_fixed = request.final_position_fixed
        segment_num = request.segment_num

        # Initialize empty lists for STMs and final states (populated in level-1)
        stms = np.zeros((segment_num, 6, 6))
        xf_patches = np.zeros((segment_num, 6))

        sigma = 1.0  # Step damping factor

        metadata = {
            "iterations": 0,
            "convergence_history": [],
        }

        success = False

        for iteration in range(max_attempts):
            for seg in range(segment_num):
                pos_request = PositionInput(
                    t_initial=t_patches[seg],
                    x_initial=x_patches[seg],
                    t_target=t_patches[seg + 1],
                    x_target=x_patches[seg + 1],
                    segment_num=seg + 1,
                    max_attempts=50,
                    tol=position_tol,
                    norm_fn=norm_fn,
                )
                pos_output = self._position_shooter.run(pos_request)
                x_patches[seg] = pos_output.x0_corrected
                xf_patches[seg] = pos_output.xf_corrected
                stms[seg] = pos_output.stm_corrected

            # Compute velocity discontinuities between patches
            delta_v_list = [
                xf_patches[i][3:6] - x_patches[i + 1][3:6]
                for i in range(segment_num - 1)
            ]
            delta_v_vec = np.concatenate(delta_v_list) if delta_v_list else np.zeros(0)

            error_norm = float(np.linalg.norm(delta_v_vec))

            metadata["convergence_history"].append({
                "iteration": iteration + 1,
                "velocity_error_norm": error_norm,
            })

            if error_norm < tol:
                success = True
                break

            # Build state relationship matrix
            n_rows = (segment_num - 2) * 3 + 3
            n_cols = (segment_num - 2) * 4 + 12
            state_rel_matrix = np.zeros((n_rows, n_cols))

            for i in range(1, segment_num):
                # Get STMs: stm21 (from node iteration-1 to iteration), stm32 (from node iteration to iteration+1)
                block = self._build_relationship_matrix(stms, x_patches, xf_patches, t_patches, i, dynsys_fn)

                row_start = (i - 1) * 3
                col_start = (i - 1) * 4
                state_rel_matrix[row_start:row_start + 3, col_start:col_start + 12] = block

            # Handle fixed boundaries
            if initial_position_fixed:
                state_rel_matrix = state_rel_matrix[:, 4:]
            if final_position_fixed:
                state_rel_matrix = state_rel_matrix[:, :-4]

            # Solve for corrections using pseudo-inverse
            correction = np.linalg.pinv(state_rel_matrix) @ delta_v_vec

            # Apply corrections to positions and epochs
            update_segments = list(range(segment_num + 1))
            index_offset = 0

            if initial_position_fixed:
                update_segments.remove(0)
                index_offset = -1
            if final_position_fixed and segment_num in update_segments:
                update_segments.remove(segment_num)

            for i in update_segments:
                # Extract corrections for this patch
                dR, dt = self._extract_patch_correction(correction, i + index_offset)
                # Apply corrections with damping
                x_patches[i][:3] += sigma * dR
                t_patches[i] += sigma * dt

        metadata["iterations"] = i + 1 if success else max_attempts

        return VelocityOutput(
            x_corrected=x_patches,
            t_corrected=list(t_patches),
            success=success,
            metadata=metadata,
        )

    def _build_relationship_matrix(self, stms, x_patches, xf_patches, t_patches, iteration, dynsys_fn):
        """Build the relationship matrix M for velocity discontinuity at patch k.
        
        Uses the new _build_M_matrix method with helper functions for STM block extraction.
        
        Parameters
        ----------
        stms : ndarray
            Array of STMs for each segment
        x_patches : list
            Initial states at each patch point
        xf_patches : ndarray
            Final states at each patch point (after propagation)
        t_patches : ndarray
            Times at each patch point
        iteration : int
            Current patch point index (k)
        dynsys_fn : callable
            Dynamical system function
            
        Returns
        -------
        block : ndarray (3, 12)
            Relationship matrix for this junction
        """
        # Get STMs
        # stm21: (k-1) → k, then invert to get stm12: k → (k-1) 
        # stm32: k → (k+1)
        stm21 = stms[iteration - 1]
        stm12 = np.linalg.inv(stm21)  # Inverse to get k → (k-1)
        stm32 = stms[iteration]

        # Extract STM blocks using helper functions
        A_backward = _get_A_block(stm12)  # Use stm12, not stm21!
        B_backward = _get_B_block(stm12)
        D_backward = _get_D_block(stm12)
        
        A_forward = _get_A_block(stm32)
        B_forward = _get_B_block(stm32)
        D_forward = _get_D_block(stm32)

        # Extract velocities at the three nodes
        v_km1_plus = x_patches[iteration - 1][3:6]   # v_{k-1}^+
        v_k_plus = x_patches[iteration][3:6]         # v_k^+
        v_kp1_minus = xf_patches[iteration][3:6]     # v_{k+1}^-

        # Compute accelerations at node k
        a_k_minus = dynsys_fn(t_patches[iteration], xf_patches[iteration - 1])[3:6]  # a_k^-
        a_k_plus = dynsys_fn(t_patches[iteration], x_patches[iteration])[3:6]        # a_k^+

        # Build M matrix using the new method
        block = self._build_M_matrix(
            B_backward=B_backward,
            A_backward=A_backward,
            D_backward=D_backward,
            B_forward=B_forward,
            A_forward=A_forward,
            D_forward=D_forward,
            v_km1_plus=v_km1_plus,
            v_k_plus=v_k_plus,
            v_kp1_minus=v_kp1_minus,
            a_k_plus=a_k_plus,
            a_k_minus=a_k_minus,
        )

        return block

    def _dVk_dRk1(self, B_block):
        return np.linalg.solve(B_block, np.eye(3))

    def _dVk_dRk(self, B_block, A_block):
        B_inv = np.linalg.solve(B_block, np.eye(3))
        return - B_inv @ A_block

    def _dVk_dtk1(self, B_block, v_k):
        B_inv = np.linalg.solve(B_block, np.eye(3))
        return - B_inv @ v_k

    def _dVk_dtk(self, a_k, D_block, B_block, v_k):
        B_inv = np.linalg.solve(B_block, np.eye(3))
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
        v_k_plus,
        v_kp1_minus,
        a_k_plus,
        a_k_minus,
    ):
        r"""Build the M matrix from equation (32).
        
        Computes the Jacobian of velocity discontinuity with respect to 
        control variables at three consecutive patch points k-1, k, k+1.
        
        .. math::
            \delta \Delta \bar{V}_k = M \cdot 
            [\delta \bar{R}_{k-1}, \delta t_{k-1}, \delta \bar{R}_k, \delta t_k, 
             \delta \bar{R}_{k+1}, \delta t_{k+1}]^T
        
        Parameters
        ----------
        B_backward : ndarray (3, 3)
            B block from STM of segment (k-1 → k)
        A_backward : ndarray (3, 3)
            A block from STM of segment (k-1 → k)
        D_backward : ndarray (3, 3)
            D block from STM of segment (k-1 → k)
        B_forward : ndarray (3, 3)
            B block from STM of segment (k → k+1)
        A_forward : ndarray (3, 3)
            A block from STM of segment (k → k+1)
        D_forward : ndarray (3, 3)
            D block from STM of segment (k → k+1)
        v_km1_plus : ndarray (3,)
            Velocity at patch k-1 after leaving (initial velocity)
        v_k_plus : ndarray (3,)
            Velocity at patch k after leaving (initial velocity)
        v_kp1_minus : ndarray (3,)
            Velocity at patch k+1 before arriving (final velocity)
        a_k_plus : ndarray (3,)
            Acceleration at patch k (outgoing state)
        a_k_minus : ndarray (3,)
            Acceleration at patch k (incoming state)
            
        Returns
        -------
        M : ndarray (3, 12)
            Jacobian matrix relating velocity discontinuity to control variables
        """
        # Compute partials for forward segment (k → k+1) - affects ΔV_k^-
        dVk_minus_dRk = self._dVk_dRk1(B_forward)
        dVk_minus_dtk = self._dVk_dtk1(B_forward, v_kp1_minus)
        dVk_minus_dRkp1 = self._dVk_dRk(B_forward, A_forward)
        dVk_minus_dtkp1 = self._dVk_dtk(a_k_minus, D_forward, B_forward, v_k_plus)
        
        # Compute partials for backward segment (k-1 → k) - affects ΔV_k^+
        dVk_plus_dRkm1 = self._dVk_dRk(B_backward, A_backward)
        dVk_plus_dtkm1 = self._dVk_dtk(a_k_plus, D_backward, B_backward, v_km1_plus)
        dVk_plus_dRk = self._dVk_dRk1(B_backward)
        dVk_plus_dtk = self._dVk_dtk1(B_backward, v_k_plus)
        
        # Build M matrix: ∂(ΔV_k^+ - ΔV_k^-)/∂[R_{k-1}, t_{k-1}, R_k, t_k, R_{k+1}, t_{k+1}]
        M = np.hstack([
            dVk_plus_dRkm1,                           # ∂ΔV_k^+/∂R_{k-1} (3x3)
            dVk_plus_dtkm1.reshape(-1, 1),           # ∂ΔV_k^+/∂t_{k-1} (3x1)
            dVk_plus_dRk - dVk_minus_dRk,            # ∂(ΔV_k^+ - ΔV_k^-)/∂R_k (3x3)
            (dVk_plus_dtk - dVk_minus_dtk).reshape(-1, 1),  # ∂(ΔV_k^+ - ΔV_k^-)/∂t_k (3x1)
            -dVk_minus_dRkp1,                         # -∂ΔV_k^-/∂R_{k+1} (3x3)
            -dVk_minus_dtkp1.reshape(-1, 1),         # -∂ΔV_k^-/∂t_{k+1} (3x1)
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
        