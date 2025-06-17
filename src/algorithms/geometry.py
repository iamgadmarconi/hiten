from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar

from algorithms.dynamics.rtbp import create_rtbp_system
from algorithms.integrators.base import Solution
from algorithms.integrators.rk import RungeKutta
from utils.log_config import logger


def _find_y_zero_crossing(x0: NDArray[np.float64], mu: float, forward: int = 1, **solver_kwargs: Any) -> Tuple[float, NDArray[np.float64]]:
    """
    Find the time and state at which an orbit next crosses the y=0 plane.
    
    This function propagates a trajectory from an initial state and determines
    when it next crosses the y=0 plane (i.e., the x-z plane). This is particularly
    useful for periodic orbit computations where the orbit is symmetric about the
    x-z plane.
    
    Parameters
    ----------
    x0 : npt.NDArray[np.float64]
        Initial state vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    **solver_kwargs : Any
        Additional keyword arguments passed to the propagator.
    
    Returns
    -------
    t1_z : float
        Time at which the orbit crosses the y=0 plane
    x1_z : npt.NDArray[np.float64]
        State vector [x, y, z, vx, vy, vz] at the crossing
    
    Notes
    -----
    The function uses a two-step approach:
    1. First integrating to an approximate time where the crossing is expected (π/2 - 0.15)
    2. Then using a root-finding method to precisely locate the crossing time
    
    This hybrid approach is more efficient than using a very fine integration
    time step, especially for orbits with long periods.
    """
    logger.debug(f"Entering _find_y_zero_crossing with x0={x0}, mu={mu}, forward={forward}")
    t0_z = np.pi/2 - 0.15
    logger.debug(f"Initial time guess t0_z = {t0_z}")

    # 1) Integrate from t=0 up to t0_z.
    logger.debug(f"Propagating from t=0 to t={t0_z}")
    sol = propagate_crtbp(x0, 0.0, t0_z, mu, forward=forward, **solver_kwargs)
    xx = sol.states
    x0_z: NDArray[np.float64] = xx[-1]
    logger.debug(f"State after initial propagation x0_z = {x0_z}")

    # 2) Define a local function that depends on time t.
    def y_component_wrapper(t: float) -> float:
        """
        Wrapper function that returns the y-coordinate of the orbit at time t.
        
        This function is used as the target for root-finding, since we want to
        find where y=0 (i.e., when the orbit crosses the x-z plane).
        
        Parameters
        ----------
        t : float
            Time to evaluate the orbit
        
        Returns
        -------
        float
            The y-coordinate of the orbit at time t
        """
        return _y_component(t, t0_z, x0_z, mu, forward=forward, steps=500)

    # 3) Find the time at which y=0 by bracketing the root.
    logger.debug(f"Finding root bracket starting from t0_z = {t0_z}")
    t1_z = _find_bracket(y_component_wrapper, t0_z)
    logger.debug(f"Found crossing time t1_z = {t1_z}")

    # 4) Integrate from t0_z to t1_z to get the final state.
    logger.debug(f"Propagating from t={t0_z} to t={t1_z} to get final state")
    sol = propagate_crtbp(x0_z, t0_z, t1_z, mu, forward=forward, **solver_kwargs)
    xx_final = sol.states
    x1_z: NDArray[np.float64] = xx_final[-1]
    logger.debug(f"Final state at crossing x1_z = {x1_z}")

    return t1_z, x1_z

def _find_bracket(f: Callable[[float], float], x0: float, max_expand: int = 500) -> float:
    """
    Find a bracketing interval for a root and solve using Brent's method.
    
    This function attempts to locate an interval containing a root of the function f
    by expanding outward from an initial guess x0. Once a sign change is detected,
    it applies Brent's method to find the precise root location. The approach is
    similar to MATLAB's fzero function.
    
    Parameters
    ----------
    f : Callable[[float], float]
        The function for which we want to find a root f(x)=0
    x0 : float
        Initial guess for the root location
    max_expand : int, optional
        Maximum number of expansion iterations to try.
        Default is 500.
    
    Returns
    -------
    float
        The location of the root, as determined by Brent's method
    
    Notes
    -----
    The function works by:
    1. Starting with a small step size (1e-10)
    2. Testing points on both sides of x0 (x0±dx)
    3. If a sign change is detected, applying Brent's method to find the root
    4. If no sign change is found, increasing the step size by sqrt(2) and trying again
    
    This approach is effective for finding roots of smooth functions where a
    reasonable initial guess is available, particularly for orbital period and
    crossing time calculations.
    """
    logger.debug(f"Entering _find_bracket with initial guess x0={x0}")
    f0 = f(x0)
    if abs(f0) < 1e-14:
        logger.debug(f"Initial guess x0={x0} is already close to root (f(x0)={f0}). Returning x0.")
        return x0

    dx = 1e-10 # * abs(x0) if x0 != 0 else 1e-10
    logger.debug(f"Starting bracket search with dx={dx}")

    for i in range(max_expand):
        # Try the positive direction: x0 + dx
        x_right = x0 + dx
        f_right = f(x_right)
        logger.debug(f"Iteration {i+1}: Testing right: x={x_right}, f(x)={f_right}, dx={dx}")
        if np.sign(f_right) != np.sign(f0):
            a, b = (x0, x_right) if x0 < x_right else (x_right, x0)
            logger.debug(f"Found bracket on right: ({a}, {b}). Solving with brentq.")
            root = root_scalar(f, bracket=(a, b), method='brentq', xtol=1e-12).root
            logger.debug(f"Found root: {root}")
            return root

        # Try the negative direction: x0 - dx
        x_left = x0 - dx
        f_left = f(x_left)
        logger.debug(f"Iteration {i+1}: Testing left:  x={x_left}, f(x)={f_left}, dx={dx}")
        if np.sign(f_left) != np.sign(f0):
            a, b = (x_left, x0) if x_left < x0 else (x0, x_left)
            logger.debug(f"Found bracket on left: ({a}, {b}). Solving with brentq.")
            root = root_scalar(f, bracket=(a, b), method='brentq', xtol=1e-12).root
            logger.debug(f"Found root: {root}")
            return root

        # Expand step size by multiplying by sqrt(2)
        dx *= np.sqrt(2)

    logger.warning(f"Failed to find a bracketing interval within {max_expand} expansions from x0={x0}")
    # Consider raising an error or returning a specific value if bracket not found
    raise RuntimeError(f"Could not find bracket for root near {x0} within {max_expand} iterations")


def _y_component(t1: float, t0_z: float, x0_z: NDArray[np.float64], mu: float, forward: int = 1, steps: int = 3000, tol: float = 1e-10) -> float:
    """
    Compute the y-component of an orbit at a specified time.
    
    This function propagates an orbit from a reference state and time to a
    target time, and returns the y-component of the resulting state. It is
    designed to be used for finding orbital plane crossings.
    
    Parameters
    ----------
    t1 : float
        Target time at which to evaluate the y-component
    t0_z : float
        Reference time corresponding to the reference state x0_z
    x0_z : npt.NDArray[np.float64]
        Reference state vector [x, y, z, vx, vy, vz] at time t0_z
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : {1, -1}, optional
        Direction of time integration:
        * 1: forward in time (default)
        * -1: backward in time
    steps : int, optional
        Number of integration steps to use. Default is 3000.
    tol : float, optional
        Tolerance for the numerical integrator. Default is 1e-10.
    
    Returns
    -------
    float
        The y-component of the orbit state at time t1
    
    Notes
    -----
    This function is primarily used within root-finding algorithms to locate
    precise times when the orbit crosses the y=0 plane. It avoids unnecessary
    computation when t1 is very close to t0_z by simply returning the y-component
    of the reference state in that case.
    """
    logger.debug(f"Entering _y_component: t1={t1}, t0_z={t0_z}, x0_z={x0_z}, mu={mu}, forward={forward}")
    # If t1 == t0_z, no integration is done.  Just take the initial condition.
    if np.isclose(t1, t0_z, rtol=3e-10, atol=1e-10):
        x1_zgl = x0_z
        logger.debug(f"t1 ({t1}) is close to t0_z ({t0_z}). Returning y-component from x0_z: {x1_zgl[1]}")
    else:
        logger.debug(f"Propagating from t={t0_z} to t={t1}")
        sol = propagate_crtbp(x0_z, t0_z, t1, mu, forward=forward, steps=steps, rtol=3*tol, atol=tol)
        xx = sol.states
        # The final state is the last row of xx
        x1_zgl: NDArray[np.float64] = xx[-1, :]
        logger.debug(f"Propagation finished. Final state x1_zgl = {x1_zgl}. Returning y-component: {x1_zgl[1]}")

    return float(x1_zgl[1]) # Explicitly cast to float


def propagate_crtbp(
    state0: Sequence[float],
    t0: float,
    tf: float,
    mu: float,
    forward: int = 1,
    steps: int = 1000,
    **solve_kwargs: Dict[str, Any]
) -> Solution:
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

    system = create_rtbp_system(mu)

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
        return forward * system.rhs(t, y)

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

    # 6) Finally, flip the reported times so that if forward = -1,
    #    the time array goes from 0 down to -T (like MATLAB's t=FORWARD*t)
    t = forward * sol.t
    logger.debug(f"Final time array adjusted for direction: [{t[0]:.4f}, ..., {t[-1]:.4f}] ({len(t)} points)")

    return Solution(t, sol.y.T)

def _validate_initial_state(state):
    state_np = np.asarray(state, dtype=np.float64)
    if state_np.shape != (6,):
        msg = f"Initial state vector must have 6 elements, but got shape {state_np.shape}"
        logger.error(msg)
        raise ValueError(msg)
    return state_np