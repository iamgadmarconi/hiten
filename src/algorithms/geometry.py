import numpy as np
import mpmath as mp
from numpy.typing import NDArray
from scipy.optimize import root_scalar
from typing import Callable, Any, Tuple

from system.libration import LibrationPoint, CollinearPoint, L1Point, L2Point, L3Point
from algorithms.propagators import propagate_crtbp
from log_config import logger


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
    tolzero = 1.e-10
    # Initial guess for the time
    t0_z = np.pi/2 - 0.15
    logger.debug(f"Initial time guess t0_z = {t0_z}")

    # 1) Integrate from t=0 up to t0_z.
    logger.debug(f"Propagating from t=0 to t={t0_z}")
    sol = propagate_crtbp(x0, 0.0, t0_z, mu, forward=forward, **solver_kwargs)
    xx = sol.y.T  # assume sol.y is (state_dim, time_points)
    x0_z: NDArray[np.float64] = xx[-1]  # final state after integration
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
    xx_final = sol.y.T
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
        xx = sol.y.T
        # The final state is the last row of xx
        x1_zgl: NDArray[np.float64] = xx[-1, :]
        logger.debug(f"Propagation finished. Final state x1_zgl = {x1_zgl}. Returning y-component: {x1_zgl[1]}")

    return float(x1_zgl[1]) # Explicitly cast to float


def _gamma_l1(mu: float, x: float):
    """
    Compute the gamma function for L1 libration point.
    """
    term1 = x**5
    term2 = -(3-mu) * x**4
    term3 = (3-2*mu) * x**3
    term4 = -mu*x**2
    term5 = 2*mu*x
    term6 = -mu

    return term1 + term2 + term3 + term4 + term5 + term6

def _gamma_l2(mu: float, x: float):
    """
    Compute the gamma function for L2 libration point.
    """
    term1 = x**5
    term2 = (3-mu) * x**4
    term3 = (3-2*mu) * x**3
    term4 = -mu*x**2
    term5 = -2*mu*x
    term6 = -mu

    return term1 + term2 + term3 + term4 + term5 + term6

def _gamma_l3(mu: float, x: float):
    """
    Compute the gamma function for L3 libration point.
    """
    term1 = x**5
    term2 = (2+mu) * x**4
    term3 = (1+2*mu) * x**3
    term4 = -(1-mu) * x**2
    term5 = -2*(1-mu)*x
    term6 = -(1-mu)

    return term1 + term2 + term3 + term4 + term5 + term6

def _gamma_L(mu: float, libration_point: LibrationPoint, precision: int = 50):
    """
    Calculate the ratio of libration point distance from the closest primary 
    with high precision using a hybrid approach.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system
    libration_point : LibrationPoint
        The libration point to calculate (L1, L2, or L3)
    precision : int, optional
        Number of decimal places for high precision calculation. Default is 50.
        
    Returns
    -------
    float
        The gamma ratio for the specified libration point with high precision.
    """
    if not isinstance(libration_point, CollinearPoint):
        msg = f"Expected CollinearPoint, got {type(libration_point)}."
        logger.error(msg)
        raise TypeError(msg)
    
    # Step 1: Get initial approximation using np.roots()
    mu2 = 1 - mu
    
    if isinstance(libration_point, L1Point):
        # Define polynomial coefficients for L1
        poly = [1, -1*(3-mu), (3-2*mu), -mu, 2*mu, -mu]
        logger.debug(f"Finding initial estimate for L1 using np.roots()")
    elif isinstance(libration_point, L2Point):
        # Define polynomial coefficients for L2
        poly = [1, (3-mu), (3-2*mu), -mu, -2*mu, -mu]
        logger.debug(f"Finding initial estimate for L2 using np.roots()")
    elif isinstance(libration_point, L3Point):
        # Define polynomial coefficients for L3
        poly = [1, (2+mu), (1+2*mu), -mu2, -2*mu2, -mu2]
        logger.debug(f"Finding initial estimate for L3 using np.roots()")
    else:
        msg = f"Expected L1Point, L2Point, or L3Point, got {type(libration_point)}."
        logger.error(msg)
        raise TypeError(msg)
    
    # Find all roots
    roots = np.roots(poly)
    
    # Find the real root (there should be only one)
    x0 = None
    for r in roots:
        if np.isreal(r):
            x0 = float(r.real)
            break
    
    if x0 is None:
        # Fallback to traditional initial approximations if np.roots() fails
        if isinstance(libration_point, L1Point) or isinstance(libration_point, L2Point):
            x0 = (mu / 3)**(1/3)
        else:  # L3Point
            x0 = 1 - 7 / 12 * mu
    
    logger.debug(f"Initial estimate for {type(libration_point).__name__}: x0 = {x0}")
    
    # Step 2: Refine using high precision mp.findroot()
    with mp.workdps(precision):
        if isinstance(libration_point, L1Point):
            func = lambda x_val: _gamma_l1(mu, x_val)
        elif isinstance(libration_point, L2Point):
            func = lambda x_val: _gamma_l2(mu, x_val)
        else:  # L3Point
            func = lambda x_val: _gamma_l3(mu, x_val)
        
        x = mp.findroot(func, x0)
        x = float(x)
    
    logger.info(f"{type(libration_point).__name__} position calculated: x = {x}")
    return x
