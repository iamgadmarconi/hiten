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

from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

import numpy as np

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import _compute_stm
from hiten.utils.log_config import logger

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


class _MultipleShootingOrbitOperators(_OrbitCorrectionOperatorBase, _MultipleShootingOperators):
    """Multiple-shooting operators with clean residual/Jacobian evaluation."""
    pass