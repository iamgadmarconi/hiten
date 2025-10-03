"""Domain-agnostic operators for correction and continuation algorithms.

This module defines Protocol-based operator interfaces that abstract domain-specific
operations (propagation, STM computation, event detection, correction) from the
algorithm backends. Interfaces adapt domain objects to these operators, keeping
backends domain-agnostic.

The operator pattern allows:
- Backend remains pure numerical algorithm (arrays + operators)
- Interface owns all domain binding (CR3BP, events, integrators, STM, domain instantiation)
- Easy testing and swapping of implementations
"""

from typing import TYPE_CHECKING, Callable, Optional, Protocol, Sequence

import numpy as np

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit



class _SingleShootingOperators(Protocol):
    """Operations required for single-shooting correction.
    
    This protocol defines the minimal set of operations a backend needs
    to perform single-shooting Newton iteration without knowing about
    CR3BP, integrators, or domain-specific event detection.
    
    Notes
    -----
    All operations work with numpy arrays. The interface adapts domain
    objects (PeriodicOrbit, dynamical systems, event functions) into
    these array-based operations.
    """

    @property
    def control_indices(self) -> Sequence[int]:
        """Indices of state components that vary during correction."""
        ...

    @property
    def residual_indices(self) -> Sequence[int]:
        """Indices of state components with boundary conditions."""
        ...

    @property
    def target(self) -> np.ndarray:
        """Target values for boundary condition residuals."""
        ...

    @property
    def extra_jacobian(self) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Optional extra Jacobian term (e.g., for period correction).
        
        Signature: extra_jac(x_event, Phi_full) -> J_extra
        where J_extra has shape (n_residual, n_control).
        """
        ...

    def reconstruct_full_state(self, base_state: np.ndarray, control_params: np.ndarray) -> np.ndarray:
        """Reconstruct full state from control parameters.
        
        Parameters
        ----------
        base_state : np.ndarray
            Template state with uncontrolled components.
        control_params : np.ndarray
            Control parameter values, shape (n_control,).
        
        Returns
        -------
        np.ndarray
            Full state vector with control_params inserted at control_indices.
        """
        ...

    def propagate_to_event(self, x0: np.ndarray) -> tuple[float, np.ndarray]:
        """Propagate state until boundary event occurs.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state (full state).
        
        Returns
        -------
        t_event : float
            Time at which event occurred.
        x_event : np.ndarray
            State at event time.
        """
        ...

    def compute_stm_to_event(self, x0: np.ndarray, t_event: float) -> np.ndarray:
        """Compute state transition matrix from x0 to event time.
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state (full state).
        t_event : float
            Integration time span.
        
        Returns
        -------
        Phi : np.ndarray
            State transition matrix, shape (n_state, n_state).
        """
        ...


class _MultipleShootingOperators(Protocol):
    """Operations required for multiple-shooting correction.
    
    This protocol extends the single-shooting operators with patch-specific
    operations for continuity constraints and block-structured Jacobian assembly.
    
    Notes
    -----
    Multiple shooting divides the trajectory into N patches. Each patch has:
    - Initial state (at patch boundary i)
    - Time span (dt from patch i to i+1)
    - Template (full state from initial guess, for uncontrolled components)
    """

    @property
    def control_indices(self) -> Sequence[int]:
        """Indices of state components that vary during correction."""
        ...

    @property
    def continuity_indices(self) -> Sequence[int]:
        """Indices of state components enforced continuous at patch junctions."""
        ...

    @property
    def boundary_indices(self) -> Sequence[int]:
        """Indices of state components with boundary conditions at final patch."""
        ...

    @property
    def target(self) -> np.ndarray:
        """Target values for boundary condition residuals."""
        ...

    @property
    def patch_times(self) -> np.ndarray:
        """Time values at patch boundaries, shape (n_patches + 1,)."""
        ...

    @property
    def patch_templates(self) -> Sequence[np.ndarray]:
        """Full-state templates at each patch (for uncontrolled components).
        
        Length: n_patches
        Each template has shape (n_state,)
        """
        ...

    @property
    def extra_jacobian(self) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        """Optional extra Jacobian term (e.g., for period correction).
        
        Signature: extra_jac(x_boundary, Phi_full) -> J_extra
        where J_extra has shape (n_boundary, n_control).
        """
        ...

    def reconstruct_full_state(self, template: np.ndarray, control_params: np.ndarray) -> np.ndarray:
        """Reconstruct full state from template and control parameters.
        
        Parameters
        ----------
        template : np.ndarray
            Template state with uncontrolled components from initial guess.
        control_params : np.ndarray
            Control parameter values, shape (n_control,).
        
        Returns
        -------
        np.ndarray
            Full state vector with control_params inserted at control_indices.
        """
        ...

    def propagate_segment(self, x0: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state for fixed time span (patch segment).
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state (full state) at patch boundary.
        dt : float
            Time span to propagate.
        
        Returns
        -------
        x_final : np.ndarray
            State after propagating for dt.
        """
        ...

    def propagate_to_event(self, x_final_patch: np.ndarray) -> tuple[float, np.ndarray]:
        """Propagate final patch state until boundary event occurs.
        
        Parameters
        ----------
        x_final_patch : np.ndarray
            Initial state at final patch boundary.
        
        Returns
        -------
        t_event : float
            Time at which event occurred (relative to patch start).
        x_event : np.ndarray
            State at event time.
        """
        ...

    def compute_stm_segment(self, x0: np.ndarray, dt: float) -> np.ndarray:
        """Compute STM for fixed time span (patch segment).
        
        Parameters
        ----------
        x0 : np.ndarray
            Initial state (full state) at patch boundary.
        dt : float
            Time span.
        
        Returns
        -------
        Phi : np.ndarray
            State transition matrix, shape (n_state, n_state).
        """
        ...

    def compute_stm_to_event(self, x_final_patch: np.ndarray, t_event: float) -> np.ndarray:
        """Compute STM from final patch to event time.
        
        Parameters
        ----------
        x_final_patch : np.ndarray
            Initial state at final patch boundary.
        t_event : float
            Integration time span to event.
        
        Returns
        -------
        Phi : np.ndarray
            State transition matrix, shape (n_state, n_state).
        """
        ...


class _OrbitCorrectionOperatorBase:
    """Lightweight base with shared integration/STM/event helpers.

    Centralizes common utilities used by single- and multiple-shooting
    operator implementations while keeping backends typed against Protocols.
    """

    def __init__(
        self,
        *,
        domain_obj: "PeriodicOrbit",
        method: str,
        order: int,
        steps: int,
        forward: int,
        event_func: Optional[Callable] = None,
    ) -> None:
        self._domain_obj = domain_obj
        self._method = method
        self._order = order
        self._steps = steps
        self._forward = forward
        self._event_func = event_func

    # Helpers
    def _propagate_fixed(self, x0: np.ndarray, dt: float) -> np.ndarray:
        sol = _propagate_dynsys(
            dynsys=self._domain_obj.dynamics.dynsys,
            state0=x0,
            t0=0,
            tf=dt,
            method=self._method,
            order=self._order,
            steps=self._steps,
            forward=1,
        )
        return sol.states[-1, :]

    def _compute_stm(self, x0: np.ndarray, dt: float) -> np.ndarray:
        _, _, Phi, _ = _compute_stm(
            self._domain_obj.dynamics.var_dynsys,
            x0,
            dt,
            steps=self._steps,
            method=self._method,
            order=self._order,
        )
        return Phi

    def _propagate_to_event(self, x0: np.ndarray) -> tuple[float, np.ndarray]:
        if self._event_func is None:
            raise RuntimeError("No event function configured for this operator")
        return self._event_func(
            dynsys=self._domain_obj.dynamics.dynsys,
            x0=x0,
            forward=self._forward,
        )


class _SingleShootingOrbitOperators(_OrbitCorrectionOperatorBase, _SingleShootingOperators):
    """Concrete implementation of _SingleShootingOperators for periodic orbits.
    
    This class adapts PeriodicOrbit domain objects to the abstract operator
    protocol, handling CR3BP-specific propagation, STM computation, and event
    detection.
    
    Parameters
    ----------
    domain_obj : PeriodicOrbit
        The periodic orbit domain object.
    control_indices : Sequence[int]
        Indices of state components that vary during correction.
    residual_indices : Sequence[int]
        Indices of state components with boundary conditions.
    target : np.ndarray
        Target values for boundary conditions.
    extra_jacobian : Callable or None
        Optional extra Jacobian term (e.g., for period correction).
    event_func : Callable
        Event function for boundary detection.
    forward : int
        Integration direction (1 for forward, -1 for backward).
    method : str
        Integration method.
    order : int
        Integration order.
    steps : int
        Number of integration steps.
    """

    def __init__(
        self,
        *,
        domain_obj: "PeriodicOrbit",
        control_indices: Sequence[int],
        residual_indices: Sequence[int],
        target: Sequence[float],
        extra_jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
        event_func: Callable,
        forward: int,
        method: str,
        order: int,
        steps: int,
    ):
        super().__init__(
            domain_obj=domain_obj,
            method=method,
            order=order,
            steps=steps,
            forward=forward,
            event_func=event_func,
        )
        self._control_indices = tuple(control_indices)
        self._residual_indices = tuple(residual_indices)
        self._target = np.asarray(target, dtype=float)
        self._extra_jacobian = extra_jacobian
        self._base_state = domain_obj.initial_state.copy()

    @property
    def control_indices(self) -> Sequence[int]:
        return self._control_indices

    @property
    def residual_indices(self) -> Sequence[int]:
        return self._residual_indices

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def extra_jacobian(self) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        return self._extra_jacobian

    def reconstruct_full_state(self, base_state: np.ndarray, control_params: np.ndarray) -> np.ndarray:
        """Reconstruct full state from control parameters."""
        x_full = base_state.copy()
        x_full[list(self._control_indices)] = control_params
        return x_full

    def propagate_to_event(self, x0: np.ndarray) -> tuple[float, np.ndarray]:
        """Propagate state until boundary event occurs."""
        return self._propagate_to_event(x0)

    def compute_stm_to_event(self, x0: np.ndarray, t_event: float) -> np.ndarray:
        """Compute state transition matrix from x0 to event time."""
        return self._compute_stm(x0, t_event)
    
    def build_residual_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Build residual function for single-shooting correction.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Residual function: params -> residual vector.
        """
        base_state = self._base_state
        
        def residual_fn(params: np.ndarray) -> np.ndarray:
            """Compute residual from control parameters."""
            x_full = self.reconstruct_full_state(base_state, params)
            _, x_event = self.propagate_to_event(x_full)
            residual_vals = x_event[list(self._residual_indices)]
            return residual_vals - self._target
        
        return residual_fn
    
    def build_jacobian_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Build Jacobian function for single-shooting correction.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Jacobian function: params -> Jacobian matrix.
        """
        base_state = self._base_state
        
        def jacobian_fn(params: np.ndarray) -> np.ndarray:
            """Compute Jacobian from control parameters."""
            x_full = self.reconstruct_full_state(base_state, params)
            t_event, x_event = self.propagate_to_event(x_full)
            Phi_full = self.compute_stm_to_event(x_full, t_event)
            
            # Extract relevant block
            jac = Phi_full[np.ix_(list(self._residual_indices), list(self._control_indices))]
            
            # Apply extra Jacobian term if present
            if self._extra_jacobian is not None:
                jac -= self._extra_jacobian(x_event, Phi_full)
            
            return jac
        
        return jacobian_fn


class _MultipleShootingOrbitOperatorsImpl(_OrbitCorrectionOperatorBase, _MultipleShootingOperators):
    """Concrete implementation of _MultipleShootingOperators for periodic orbits.
    
    This class adapts PeriodicOrbit domain objects to the multiple-shooting
    operator protocol, handling patch-based propagation, continuity constraints,
    and block-structured Jacobian operations.
    
    Parameters
    ----------
    domain_obj : PeriodicOrbit
        The periodic orbit domain object.
    control_indices : Sequence[int]
        Indices of state components that vary during correction.
    continuity_indices : Sequence[int]
        Indices of state components enforced continuous at patch junctions.
    boundary_indices : Sequence[int]
        Indices of state components with boundary conditions at final patch.
    target : np.ndarray
        Target values for boundary conditions.
    patch_times : np.ndarray
        Time values at patch boundaries.
    patch_templates : Sequence[np.ndarray]
        Full-state templates at each patch.
    extra_jacobian : Callable or None
        Optional extra Jacobian term.
    event_func : Callable
        Event function for final boundary detection.
    forward : int
        Integration direction.
    method : str
        Integration method.
    order : int
        Integration order.
    steps : int
        Number of integration steps per patch.
    """

    def __init__(
        self,
        *,
        domain_obj: "PeriodicOrbit",
        control_indices: Sequence[int],
        continuity_indices: Sequence[int],
        boundary_indices: Sequence[int],
        target: Sequence[float],
        patch_times: np.ndarray,
        patch_templates: Sequence[np.ndarray],
        extra_jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]],
        event_func: Callable,
        forward: int,
        method: str,
        order: int,
        steps: int,
    ):
        super().__init__(
            domain_obj=domain_obj,
            method=method,
            order=order,
            steps=steps,
            forward=forward,
            event_func=event_func,
        )
        self._control_indices = tuple(control_indices)
        self._continuity_indices = tuple(continuity_indices)
        self._boundary_indices = tuple(boundary_indices)
        self._target = np.asarray(target, dtype=float)
        self._patch_times = np.asarray(patch_times, dtype=float)
        self._patch_templates = [tpl.copy() for tpl in patch_templates]
        self._extra_jacobian = extra_jacobian

    @property
    def control_indices(self) -> Sequence[int]:
        return self._control_indices

    @property
    def continuity_indices(self) -> Sequence[int]:
        return self._continuity_indices

    @property
    def boundary_indices(self) -> Sequence[int]:
        return self._boundary_indices

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def patch_times(self) -> np.ndarray:
        return self._patch_times

    @property
    def patch_templates(self) -> Sequence[np.ndarray]:
        return self._patch_templates

    @property
    def extra_jacobian(self) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        return self._extra_jacobian

    def reconstruct_full_state(self, template: np.ndarray, control_params: np.ndarray) -> np.ndarray:
        """Reconstruct full state from template and control parameters."""
        x_full = template.copy()
        x_full[list(self._control_indices)] = control_params
        return x_full

    def propagate_segment(self, x0: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state for fixed time span (patch segment)."""
        return self._propagate_fixed(x0, dt)

    def propagate_to_event(self, x_final_patch: np.ndarray) -> tuple[float, np.ndarray]:
        """Propagate final patch state until boundary event occurs."""
        return self._propagate_to_event(x_final_patch)

    def compute_stm_segment(self, x0: np.ndarray, dt: float) -> np.ndarray:
        """Compute STM for fixed time span (patch segment)."""
        return self._compute_stm(x0, dt)

    def compute_stm_to_event(self, x_final_patch: np.ndarray, t_event: float) -> np.ndarray:
        """Compute STM from final patch to event time."""
        return self._compute_stm(x_final_patch, t_event)
    
    def build_residual_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Build residual function for multiple-shooting correction.
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Residual function: params -> residual vector.
        """
        n_patches = len(self._patch_templates)
        n_control = len(self._control_indices)
        continuity_indices = list(self._continuity_indices)
        boundary_indices = list(self._boundary_indices)
        
        def residual_fn(params: np.ndarray) -> np.ndarray:
            """Compute residual from all patch control parameters."""
            params_2d = params.reshape((n_patches, n_control))
            residuals = []
            
            # Propagate all patches
            x_propagated_list = []
            for i in range(n_patches):
                x_patch = self.reconstruct_full_state(self._patch_templates[i], params_2d[i])
                
                if i < n_patches - 1:
                    # Interior patches: propagate fixed time
                    dt = self._patch_times[i + 1] - self._patch_times[i]
                    x_next = self.propagate_segment(x_patch, dt)
                else:
                    # Final patch: propagate to event
                    _, x_next = self.propagate_to_event(x_patch)
                
                x_propagated_list.append(x_next)
            
            # Continuity residuals
            for i in range(n_patches - 1):
                x_next_minus = x_propagated_list[i][continuity_indices]
                x_next_plus = params_2d[i + 1, :len(continuity_indices)]  # Assumes control_indices match continuity
                continuity_error = x_next_minus - x_next_plus
                residuals.append(continuity_error)
            
            # Boundary residual
            x_final = x_propagated_list[-1]
            boundary_error = x_final[boundary_indices] - self._target
            residuals.append(boundary_error)
            
            return np.concatenate(residuals)
        
        return residual_fn
    
    def build_jacobian_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        """Build block-structured Jacobian for multiple-shooting correction.
        
        The Jacobian has block structure:
        
            [Phi_0  -I   0   0  ]  <- continuity at patch 1
            [ 0   Phi_1  -I  0  ]  <- continuity at patch 2
            [ 0    0   Phi_2 -I ]  <- continuity at patch 3
            [dBC  dBC  dBC  Phi_f]  <- boundary at final event
        
        Returns
        -------
        Callable[[np.ndarray], np.ndarray]
            Jacobian function: params -> block-structured Jacobian matrix.
        """
        n_patches = len(self._patch_templates)
        n_control = len(self._control_indices)
        n_continuity = len(self._continuity_indices)
        n_boundary = len(self._boundary_indices)
        continuity_indices = list(self._continuity_indices)
        boundary_indices = list(self._boundary_indices)
        control_indices = list(self._control_indices)
        
        # Total residual dimension
        n_residual = (n_patches - 1) * n_continuity + n_boundary
        n_params = n_patches * n_control
        
        def jacobian_fn(params: np.ndarray) -> np.ndarray:
            """Compute block-structured Jacobian."""
            params_2d = params.reshape((n_patches, n_control))
            J = np.zeros((n_residual, n_params))
            
            # Compute STMs for all patches
            Phi_list = []
            x_propagated_list = []
            
            for i in range(n_patches):
                x_patch = self.reconstruct_full_state(self._patch_templates[i], params_2d[i])
                
                if i < n_patches - 1:
                    # Interior patches
                    dt = self._patch_times[i + 1] - self._patch_times[i]
                    Phi = self.compute_stm_segment(x_patch, dt)
                    x_next = self.propagate_segment(x_patch, dt)
                else:
                    # Final patch to event
                    t_event, x_next = self.propagate_to_event(x_patch)
                    Phi = self.compute_stm_to_event(x_patch, t_event)
                
                Phi_list.append(Phi)
                x_propagated_list.append(x_next)
            
            # Fill continuity blocks
            row_offset = 0
            for i in range(n_patches - 1):
                # Block Phi_i in position (i, i)
                col_start = i * n_control
                Phi_block = Phi_list[i][np.ix_(continuity_indices, control_indices)]
                J[row_offset:row_offset + n_continuity, col_start:col_start + n_control] = Phi_block
                
                # Block -I in position (i, i+1)
                col_start_next = (i + 1) * n_control
                J[row_offset:row_offset + n_continuity, col_start_next:col_start_next + n_continuity] = -np.eye(n_continuity)
                
                row_offset += n_continuity
            
            # Fill boundary row
            Phi_final = Phi_list[-1]
            for i in range(n_patches):
                col_start = i * n_control
                
                if i < n_patches - 1:
                    # Chain rule: dBC/dx_final * dxfinal/dx_i
                    # For interior patches, dxfinal/dx_i comes from multiplying STMs
                    Phi_chain = Phi_final
                    for j in range(n_patches - 1, i, -1):
                        Phi_chain = Phi_chain @ Phi_list[j]
                    
                    J_block = Phi_chain[np.ix_(boundary_indices, control_indices)]
                else:
                    # Final patch
                    J_block = Phi_final[np.ix_(boundary_indices, control_indices)]
                
                J[row_offset:row_offset + n_boundary, col_start:col_start + n_control] = J_block
            
            # Apply extra Jacobian term if present
            if self._extra_jacobian is not None:
                x_final = x_propagated_list[-1]
                extra_jac = self._extra_jacobian(x_final, Phi_final)
                # Subtract from boundary row(s)
                J[row_offset:row_offset + n_boundary, :] -= extra_jac
            
            return J
        
        return jacobian_fn
