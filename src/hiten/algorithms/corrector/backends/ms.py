"""Multiple shooting correction algorithms for orbital mechanics.

This module provides position shooting and velocity correction algorithms
for multiple shooting methods in periodic orbit computation.
"""

from typing import TYPE_CHECKING, Any, List

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

def _inv_block(block):
    size = block.shape[0]
    try:
        return np.linalg.solve(block, np.eye(size))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(block, rcond=1e-12)

def _build_node_partials(stm_km1_k: np.ndarray, stm_k_kp1: np.ndarray, segment_km1_k: List[np.ndarray], segment_k_kp1: List[np.ndarray], dynamics_fn: callable) -> _NodePartials:
    """Build node partials from STM.
    
    Parameters
    ----------

    stm_km1_k : np.ndarray
        STM from node k-1 to node k
    stm_k_kp1 : np.ndarray
        STM from node k to node k+1
    segment_km1_k : list(np.ndarray)
        List of states making up the segment from node k-1 to node k
    segment_k_kp1 : list(np.ndarray)
        List of states making up the segment from node k to node k+1

    Returns
    -------
    :class:`~hiten.algorithms.corrector.constraints._NodePartials` : _NodePartials
        Node partials for the segment
    """

    stm_km1_k = stm_km1_k # stm_(k, k-1) -> stm from node k-1 to node k
    stm_k_kp1 = stm_k_kp1 # stm_(k, k+1) -> stm from node k+1 to node k
    stm_k_km1 = _inv_block(stm_km1_k) # stm_(k-1, k) -> stm from node k to node k-1
    stm_kp1_k = _inv_block(stm_k_kp1) # stm_(k+1, k) -> stm from node k+1 to node k

    A_k_km1 = _get_A_block(stm_k_km1) # A_(k-1,k)
    B_k_km1 = _get_B_block(stm_k_km1) # B_(k-1,k)
    C_k_km1 = _get_C_block(stm_k_km1) # C_(k-1,k)
    D_k_km1 = _get_D_block(stm_k_km1) # D_(k-1,k)

    A_km1_k = _get_A_block(stm_km1_k) # A_(k,k-1)
    B_km1_k = _get_B_block(stm_km1_k) # B_(k,k-1)
    C_km1_k = _get_C_block(stm_km1_k) # C_(k,k-1)
    D_km1_k = _get_D_block(stm_km1_k) # D_(k,k-1)

    A_k_kp1 = _get_A_block(stm_k_kp1) # A_(k+1,k)
    B_k_kp1 = _get_B_block(stm_k_kp1) # B_(k+1,k)
    C_k_kp1 = _get_C_block(stm_k_kp1) # C_(k+1,k)   
    D_k_kp1 = _get_D_block(stm_k_kp1) # D_(k+1,k)   
    
    A_kp1_k = _get_A_block(stm_kp1_k) # A_(k,k+1)
    B_kp1_k = _get_B_block(stm_kp1_k) # B_(k,k+1)
    C_kp1_k = _get_C_block(stm_kp1_k) # C_(k,k+1)
    D_kp1_k = _get_D_block(stm_kp1_k) # D_(k,k+1)

    R_km1 = segment_km1_k[0][:3] # R_(k-1)
    R_kp1 = segment_k_kp1[-1][:3] # R_(k+1)

    V_km1_plus = segment_km1_k[0][3:6] # V_(k-1)^+
    V_k_minus = segment_km1_k[-1][3:6] # V_(k)^-
    V_k_plus = segment_k_kp1[0][3:6] # V_(k)^+
    V_kp1_minus = segment_k_kp1[-1][3:6] # V_(k+1)^-

    a_k_minus = dynamics_fn(None, segment_km1_k[-1][0:6])[3:6] # a_(k)^-
    a_k_plus = dynamics_fn(None, segment_k_kp1[0][0:6])[3:6] # a_(k)^+

    return _NodePartials(
        A_k_km1=A_k_km1,
        B_k_km1=B_k_km1,
        C_k_km1=C_k_km1,
        D_k_km1=D_k_km1,

        A_km1_k=A_km1_k,
        B_km1_k=B_km1_k,
        C_km1_k=C_km1_k,
        D_km1_k=D_km1_k,

        A_k_kp1=A_k_kp1,
        B_k_kp1=B_k_kp1,
        C_k_kp1=C_k_kp1,
        D_k_kp1=D_k_kp1,

        A_kp1_k=A_kp1_k,
        B_kp1_k=B_kp1_k,
        C_kp1_k=C_kp1_k,
        D_kp1_k=D_kp1_k,
    
        R_km1=R_km1,
        R_kp1=R_kp1,
    
        V_km1_plus=V_km1_plus,
        V_k_minus=V_k_minus,
        V_k_plus=V_k_plus,
        V_kp1_minus=V_kp1_minus,

        a_k_minus=a_k_minus,
        a_k_plus=a_k_plus,
    )


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
        
        DAMPING_FACTOR = 0.618

        t_km1 = request.t_initial
        X_km1 = request.x_initial
        t_target = request.t_target
        X_target = request.x_target
        dynamics_fn = request.dynamics_fn
        segment_num = request.segment_num
        max_attempts = request.max_attempts
        tol = request.tol
        norm_fn = request.norm_fn

        metadata = {
            "segment_number": segment_num,
            "convergence_history": [],
        }

        for i in range(max_attempts):
            X_km1_k, _, stm_km1_k, _ = _compute_stm(
                dynsys=self._var_dynsys,
                x0=X_km1,
                tf=t_target-t_km1,
                steps=self._steps,
                forward=1,
                method=self._method,
                order=self._order,
            )

            X_k_minus = X_km1_k[-1, :] # last state of segment k-1

            X_k_kp1, stm_k_kp1 = np.zeros_like(X_km1_k), np.zeros_like(stm_km1_k)

            node_partials = _build_node_partials(
                stm_km1_k=stm_km1_k,
                stm_k_kp1=stm_k_kp1,
                segment_km1_k=X_km1_k,
                segment_k_kp1=X_k_kp1,
                dynamics_fn=dynamics_fn
            )
            error = X_target[0:3] - X_km1_k[-1][0:3] # position error

            dV_km1 = _inv_block(node_partials.B_km1_k) @ (error) # velocity correction
            X_km1[3:6] += DAMPING_FACTOR * dV_km1 # update velocity of initial state of segment k-1
            norm = norm_fn(error)

            metadata["convergence_history"].append(
                {
                    "iteration": i,
                    "error": norm,
                    "dV_km1": dV_km1,
                }
            )

            if norm < tol:
                metadata["iterations"] = i
                return PositionOutput(
                    x0_corrected=X_km1,
                    xf_corrected=X_k_minus,
                    stm_corrected=stm_km1_k,
                    success=True,
                    metadata=metadata
                )

        else:
            metadata["iterations"] = max_attempts
            return PositionOutput(
                x0_corrected=X_km1,
                xf_corrected=X_k_minus,
                stm_corrected=stm_km1_k,
                success=False,
                metadata=metadata
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
        
        t_kp = request.t_patches
        X_kp = request.x_patches
        X_km = np.zeros_like(X_kp)
        t_km = np.zeros_like(t_kp)

        dynamics_fn = request.dynsys_fn

        vel_max_attempts = request.vel_max_attempts
        pos_max_attempts = request.pos_max_attempts
        pos_tol = request.pos_tol
        vel_tol = request.vel_tol
        pos_norm_fn = request.pos_norm_fn
        vel_norm_fn = request.vel_norm_fn

        initial_position_fixed = request.initial_position_fixed
        final_position_fixed = request.final_position_fixed

        constraints = request.constraints

        initial_state, initial_time = X_kp[0], t_kp[0]
        final_state, final_time = X_kp[-1], t_kp[-1]

        n_nodes = len(X_kp)
        n_segments = n_nodes - 1

        stms_km1_k = np.zeros((n_segments, 6, 6))

        for i in range(vel_max_attempts):

            for segment_num in range(n_segments):

                position_output = self._position_shooter.run(
                    PositionInput(
                        t_initial=t_kp[segment_num],
                        x_initial=X_kp[segment_num],
                        t_target=t_kp[segment_num + 1],
                        x_target=X_kp[segment_num + 1],
                        dynamics_fn=dynamics_fn,
                        segment_num=segment_num,
                        max_attempts=pos_max_attempts,
                        tol=pos_tol,
                        norm_fn=pos_norm_fn
                    )
                )
                
                X_kp[segment_num] = position_output.x0_corrected
                X_km[segment_num] = position_output.xf_corrected
                stms_km1_k[segment_num] = position_output.stm_corrected

            delta_V_array = [X_km[i][3:6] - X_kp[i+1][3:6] for i in range(n_segments - 1)]
            delta_v_vec = np.concatenate(delta_V_array)

            norm = vel_norm_fn(delta_V_array)

            if norm < vel_tol:
                return VelocityOutput(
                    x_corrected=X_kp,
                    t_corrected=t_kp,
                    success=True,
                    metadata={}
                )
            
            M = np.zeros(((n_segments-2)*3+3, (n_segments-2)*4+12))

            for i in range(1, n_segments):

                X_km1_plus = X_kp[i-1]
                X_k_minus = X_km[i-1]

                X_k_plus = X_kp[i]
                X_kp1_minus = X_km[i]

                stm_km1_k = stms_km1_k[i-1]
                stm_k_kp1 = stms_km1_k[i]
                
                X_km1_k = np.array([X_km1_plus, X_k_minus])
                X_k_kp1 = np.array([X_k_plus, X_kp1_minus])

                node_partials = _build_node_partials(
                    stm_km1_k=stm_km1_k,
                    stm_k_kp1=stm_k_kp1,
                    segment_km1_k=X_km1_k,
                    segment_k_kp1=X_k_kp1,
                    dynamics_fn = dynamics_fn,
                )

                srm_block = self._build_srm(node_partials)

                M[(i-1)*3:(i-1)*3+3, (i-1)*4:(i-1)*4+12] = srm_block
            
            if initial_position_fixed:
                M = M[:, 4:]
            if final_position_fixed:
                M = M[:, :-2]

            correction, *_ = np.linalg.lstsq(M, delta_v_vec, rcond=None)

            update_segments = list(range(n_nodes))
            index_offset = 0

            if initial_position_fixed:
                update_segments.remove(0)
                index_offset = -1
            if final_position_fixed and (n_nodes - 1) in update_segments:
                update_segments.remove(n_nodes - 1)

            # Apply correction
            for i in update_segments:
                dR, dt = self._extract_patch_correction(correction, i + index_offset)
                X_kp[i][:3] += dR
                t_kp[i] += dt

            if (n_nodes - 1) not in update_segments:
                _, dtN = self._extract_patch_correction(correction, (n_nodes - 1) + index_offset)
                t_kp[n_nodes - 1] += dtN

        return VelocityOutput(
            x_corrected=X_kp,
            t_corrected=list(t_kp),
            success=False,
            metadata={},
        )



    def _build_srm(self, node_partials: _NodePartials):
        """Build the state relationship matrix (SRM) for the multiple shooting problem."""
        
        dDVk_dRkm1 = - _inv_block(node_partials.B_k_km1)
        dDVk_dRk = _inv_block(node_partials.B_k_km1) @ node_partials.A_k_km1 - _inv_block(node_partials.B_k_kp1) @ node_partials.A_k_kp1
        dDVk_dRkp1 = _inv_block(node_partials.B_k_kp1)

        dDVk_dtkm1 = (_inv_block(node_partials.B_k_km1) @ node_partials.V_km1_plus).reshape(-1, 1)
        dDVk_dtk = (node_partials.a_k_plus - node_partials.a_k_minus + node_partials.D_km1_k @ _inv_block(node_partials.B_km1_k) @ node_partials.V_k_minus - node_partials.D_kp1_k @ _inv_block(node_partials.B_kp1_k) @ node_partials.V_k_plus).reshape(-1, 1)
        dDVk_dtkp1 = (- _inv_block(node_partials.B_k_kp1) @ node_partials.V_kp1_minus).reshape(-1, 1)

        return np.hstack((dDVk_dRkm1, dDVk_dtkm1, dDVk_dRk, dDVk_dtk, dDVk_dRkp1, dDVk_dtkp1))

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