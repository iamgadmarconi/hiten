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

            B = stm_final[:3, 3:6]
            delta_v = np.linalg.solve(B, error_final)
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
                base = (i + index_offset) * 4
                x_patches[i][:3] += sigma * correction[base:base + 3]
                t_patches[i] += sigma * correction[base + 3]

        metadata["iterations"] = i + 1 if success else max_attempts

        return VelocityOutput(
            x_corrected=x_patches,
            t_corrected=list(t_patches),
            success=success,
            metadata=metadata,
        )

    def _build_relationship_matrix(self, stms, x_patches, xf_patches, t_patches, iteration, dynsys_fn):
        stm21 = stms[iteration - 1]
        stm12 = np.linalg.inv(stm21)
        stm32 = stms[iteration]

        # Extract velocities at the three nodes
        v1plus = x_patches[iteration - 1][3:6]
        v2minus = xf_patches[iteration - 1][3:6]
        v2plus = x_patches[iteration][3:6]
        v3minus = xf_patches[iteration][3:6]

        # Compute accelerations at node iteration
        a2minus = dynsys_fn(t_patches[iteration], xf_patches[iteration - 1])[3:6]
        a2plus = dynsys_fn(t_patches[iteration], x_patches[iteration])[3:6]

        # Extract STM blocks
        A12 = stm12[:3, :3]
        B12 = stm12[:3, 3:6]
        A32 = stm32[:3, :3]
        B32 = stm32[:3, 3:6]

        # Compute B inverses using solve for numerical stability
        B12_inv = np.linalg.solve(B12, np.eye(3))
        B32_inv = np.linalg.solve(B32, np.eye(3))

        # Build the 3x12 block for this junction
        block = np.hstack([
            -B12_inv,                                              # cols 1-3
            (B12_inv @ v1plus).reshape(-1, 1),                    # col 4
            -B32_inv @ A32 + B12_inv @ A12,                       # cols 5-7
            ((a2plus - a2minus) +                                  # col 8
            B32_inv @ A32 @ v2plus -
            B12_inv @ A12 @ v2minus).reshape(-1, 1),
            B32_inv,
            (-B32_inv @ v3minus).reshape(-1, 1),                  # col 12
        ])

        return block