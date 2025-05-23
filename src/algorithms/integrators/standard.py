from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from utils.log_config import logger

from ..dynamics import crtbp_accel


def propagate_orbit(
    initial_state: Sequence[float],
    mu: float,
    tspan: Union[Sequence[float], Tuple[float, float]],
    events: Optional[Union[Callable, List[Callable]]] = None,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    method: str = 'DOP853',
    dense_output: bool = True,
    max_step: float = np.inf
) -> OdeResult:
    """
    Propagate an orbit in the CR3BP.
    
    This function numerically integrates the equations of motion for the CR3BP
    to propagate an orbit from the given initial state over the specified time span.
    
    Parameters
    ----------
    initial_state : Sequence[float]
        Initial state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    tspan : Sequence[float] or Tuple[float, float]
        Time span for integration [t_start, t_end] or array of specific times
    events : Callable or List[Callable], optional
        Events to detect during integration (see scipy.integrate.solve_ivp)
    rtol : float, optional
        Relative tolerance for the integrator
    atol : float, optional
        Absolute tolerance for the integrator
    method : str, optional
        Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA')
    dense_output : bool, optional
        Whether to compute a continuous solution
    max_step : float, optional
        Maximum allowed step size for the integrator
    
    Returns
    -------
    sol : OdeResult
        Solution object from scipy.integrate.solve_ivp containing:
        - t: array of time points
        - y: array of solution values (shape: (6, len(t)))
        - status: Integration status code
        - message: Description of the status
        - success: Boolean indicating successful integration
        - Additional integration metadata (nfev, njev, nlu)
    
    Notes
    -----
    This function uses scipy's solve_ivp under the hood, which provides
    adaptive step size integration and dense output capabilities.
    
    Default integration tolerances are set to high precision (rtol=1e-12, 
    atol=1e-12) with the DOP853 method, which is generally well-suited for
    astrodynamics problems requiring high accuracy.
    """
    # Type check and convert initial_state if needed
    initial_state_np = _validate_initial_state(initial_state)

    logger.debug(f"Initial state: {np.array2string(initial_state_np, precision=8, suppress_small=True)}")
    logger.debug(f"Time span: {tspan}")
    
    # Ensure tspan is properly formatted
    tspan_np = np.asarray(tspan, dtype=np.float64)
    if len(tspan_np) < 2:
        msg = f"Time span must have at least 2 elements, but got {len(tspan_np)}"
        logger.error(msg)
        raise ValueError(msg)
    
    logger.debug(f"Starting CR3BP orbit propagation.")
    logger.debug(f"Time span: [{tspan_np[0]}, {tspan_np[-1]}], mu={mu}")
    logger.debug(f"Integration method: {method}, rtol={rtol}, atol={atol}")
    if events is not None:
        logger.debug(f"Events detection enabled")
    
    # Create a wrapper for the acceleration function to match scipy's interface
    def f(t: float, y: np.ndarray) -> np.ndarray:
        return crtbp_accel(y, mu)
    
    # Perform the integration with error handling
    try:
        logger.debug("Calling scipy.integrate.solve_ivp...")
        sol = solve_ivp(
            f, 
            [tspan_np[0], tspan_np[-1]], 
            initial_state_np, 
            t_eval=tspan_np, 
            events=events,
            rtol=rtol, 
            atol=atol, 
            method=method, 
            dense_output=dense_output,
            max_step=max_step
        )
        
        logger.debug(f"Integration finished. Status: {sol.status} ('{sol.message}'), nfev: {sol.nfev}, njev: {getattr(sol, 'njev', 'N/A')}, nlu: {getattr(sol, 'nlu', 'N/A')}")
        
        if not sol.success:
            logger.warning(f"Integration did not complete successfully: {sol.message}")
            
        return sol
    except Exception as e:
        logger.exception(f"An error occurred during solve_ivp execution: {e}")
        raise  # Re-raise the exception after logging


def propagate_crtbp(
    state0: Sequence[float],
    t0: float,
    tf: float,
    mu: float,
    forward: int = 1,
    steps: int = 1000,
    **solve_kwargs: Dict[str, Any]
) -> OdeResult:
    """
    Propagate a state in the CR3BP from initial to final time.
    
    This function numerically integrates the CR3BP equations of motion,
    providing a trajectory from an initial state at time t0 to a final
    time tf. It handles both forward and backward propagation and is
    designed to replicate MATLAB-style integration conventions.
    
    Parameters
    ----------
    state0 : array_like
        Initial state vector [x, y, z, vx, vy, vz]
    t0 : float
        Initial time
    tf : float
        Final time
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    steps : int, optional
        Number of time steps for output. Default is 1000.
    **solve_kwargs
        Additional keyword arguments passed to scipy.integrate.solve_ivp
    
    Returns
    -------
    sol : OdeResult
        Solution object from scipy.integrate.solve_ivp containing:
        - t: array of time points
        - y: array of solution values (shape: (6, len(t)))
        - Additional integration metadata
    
    Notes
    -----
    This function adopts MATLAB's 'FORWARD' convention where:
    1. Integration always occurs over a positive time span [|t0|, |tf|]
    2. The derivative is multiplied by 'forward' to control direction
    3. The output time array is scaled by 'forward' to reflect actual times
    
    Default integration tolerances are set to high precision (rtol=3e-14, 
    atol=1e-14) but can be overridden through solve_kwargs.
    """
    # Type check and convert initial_state if needed
    state0_np = _validate_initial_state(state0)

    logger.debug(f"Initial state: {np.array2string(state0_np, precision=8, suppress_small=True)}")
    logger.debug(f"Time span: [{t0}, {tf}] (raw), mu={mu}, steps={steps}")

    direction = "forward" if forward == 1 else "backward"
    logger.debug(f"Starting CR3BP {direction} propagation.")

    # 1) Always make the integration span positive, even if tf is negative
    t0_abs = abs(t0)
    tf_abs = abs(tf)
    t_span = [t0_abs, tf_abs]
    t_eval = np.linspace(t0_abs, tf_abs, steps)
    logger.debug(f" Integration span (abs): {t_span}")
    
    # 2) ODE function includes the forward sign, exactly like 'xdot = FORWARD * xdot'
    def ode_func(t, y):
        return forward * crtbp_accel(y, mu)

    # 4) Default tolerances, or user can override via solve_kwargs
    rtol = solve_kwargs.setdefault('rtol', 3e-14)
    atol = solve_kwargs.setdefault('atol', 1e-14)
    logger.debug(f" Using integration tolerances: rtol={rtol}, atol={atol}")
    logger.debug(f" Additional solve_ivp args: { {k: v for k, v in solve_kwargs.items() if k not in ['rtol', 'atol']} }")

    # 5) Integrate
    logger.debug("Calling scipy.integrate.solve_ivp...")
    sol = solve_ivp(
        ode_func,
        t_span,
        state0_np,
        t_eval=t_eval,
        **solve_kwargs
    )
    logger.debug(f"Integration finished. Status: {sol.status} ('{sol.message}'), nfev: {sol.nfev}, njev: {sol.njev}, nlu: {sol.nlu}")

    if not sol.success:
        logger.error(f"Integration failed: {sol.message}")
        # Optionally, raise an exception here if failure should halt execution
        # raise RuntimeError(f"CR3BP propagation failed: {sol.message}")

    # 6) Finally, flip the reported times so that if forward = -1,
    #    the time array goes from 0 down to -T (like MATLAB's t=FORWARD*t)
    sol.t = forward * sol.t
    logger.debug(f"Final time array adjusted for direction: [{sol.t[0]:.4f}, ..., {sol.t[-1]:.4f}] ({len(sol.t)} points)")

    return sol

def _validate_initial_state(state):
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (6,):
        msg = f"Initial state vector must have 6 elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np