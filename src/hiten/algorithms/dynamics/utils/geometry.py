r"""
dynamics.hiten.utils.geometry
=======================

Utility routines for geometric post-processing in the spatial circular
restricted three-body problem (CRTBP).

The functions in this module locate coordinate-plane crossings, build
Poincaré surfaces of section and resample numerical trajectories
produced by :pyfunc:`hiten.algorithms.dynamics.rtbp._propagate_dynsys`.

All routines assume the canonical rotating frame of the CRTBP, where the
primary bodies are fixed at :math:`(-\mu, 0, 0)` and
:math:`(1-\mu, 0, 0)` and time is non-dimensionalised so that the mean
motion equals one.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.dynamics.rtbp import _propagate_dynsys
from hiten.utils.log_config import logger


def _find_y_zero_crossing(dynsys: _DynamicalSystem, x0: NDArray[np.float64], forward: int = 1) -> Tuple[float, NDArray[np.float64]]:
    r"""
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
    time step, especially for hiten.system.orbits with long periods.
    """
    logger.debug(f"Entering _find_y_zero_crossing with x0={x0}, forward={forward}")
    t0_z = np.pi/2 - 0.15
    logger.debug(f"Initial time guess t0_z = {t0_z}")

    # 1) Integrate from t=0 up to t0_z.
    logger.debug(f"Propagating from t=0 to t={t0_z}")
    sol = _propagate_dynsys(dynsys, x0, 0.0, t0_z, forward=forward)
    xx = sol.states
    x0_z: NDArray[np.float64] = xx[-1]
    logger.debug(f"State after initial propagation x0_z = {x0_z}")

    # 2) Define a local function that depends on time t.
    def y_component_wrapper(t: float) -> float:
        r"""
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
        return _y_component(dynsys, t, t0_z, x0_z, forward=forward, steps=500)

    # 3) Find the time at which y=0 by bracketing the root.
    logger.debug(f"Finding root bracket starting from t0_z = {t0_z}")
    t1_z = _find_bracket(y_component_wrapper, t0_z)
    logger.debug(f"Found crossing time t1_z = {t1_z}")

    # 4) Integrate from t0_z to t1_z to get the final state.
    logger.debug(f"Propagating from t={t0_z} to t={t1_z} to get final state")
    sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1_z, forward=forward)
    xx_final = sol.states
    x1_z: NDArray[np.float64] = xx_final[-1]
    logger.debug(f"Final state at crossing x1_z = {x1_z}")

    return t1_z, x1_z


def _find_x_zero_crossing(dynsys: _DynamicalSystem, x0: NDArray[np.float64], forward: int = 1) -> Tuple[float, NDArray[np.float64]]:
    r"""
    Find the time and state at which an orbit next crosses the x=0 plane.
    
    This function propagates a trajectory from an initial state and determines
    when it next crosses the x=0 plane (i.e., the y-z plane). This is particularly
    useful for periodic orbit computations where the orbit is symmetric about the
    y-z plane.
    
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
    
    Returns
    -------
    t1_z : float
        Time at which the orbit crosses the x=0 plane
    x1_z : npt.NDArray[np.float64]
        State vector [x, y, z, vx, vy, vz] at the crossing
    
    Notes
    -----
    The function uses a two-step approach:
    1. First integrating to an approximate time where the crossing is expected (π/2 - 0.15)
    2. Then using a root-finding method to precisely locate the crossing time
    
    This hybrid approach is more efficient than using a very fine integration
    time step, especially for hiten.system.orbits with long periods.
    """
    logger.debug(f"Entering _find_x_zero_crossing with x0={x0}, forward={forward}")
    t0_z = np.pi/2 - 0.15
    logger.debug(f"Initial time guess t0_z = {t0_z}")

    # 1) Integrate from t=0 up to t0_z.
    logger.debug(f"Propagating from t=0 to t={t0_z}")
    sol = _propagate_dynsys(dynsys, x0, 0.0, t0_z, forward=forward)
    xx = sol.states
    x0_z: NDArray[np.float64] = xx[-1]
    logger.debug(f"State after initial propagation x0_z = {x0_z}")

    # 2) Define a local function that depends on time t.
    def x_component_wrapper(t: float) -> float:
        r"""
        Wrapper function that returns the x-coordinate of the orbit at time t.
        
        This function is used as the target for root-finding, since we want to
        find where x=0 (i.e., when the orbit crosses the y-z plane).
        
        Parameters
        ----------
        t : float
            Time to evaluate the orbit
        
        Returns
        -------
        float
            The x-coordinate of the orbit at time t
        """
        return _x_component(dynsys, t, t0_z, x0_z, forward=forward, steps=500)

    # 3) Find the time at which x=0 by bracketing the root.
    logger.debug(f"Finding root bracket starting from t0_z = {t0_z}")
    t1_z = _find_bracket(x_component_wrapper, t0_z)
    logger.debug(f"Found crossing time t1_z = {t1_z}")

    # 4) Integrate from t0_z to t1_z to get the final state.
    logger.debug(f"Propagating from t={t0_z} to t={t1_z} to get final state")
    sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1_z, forward=forward)
    xx_final = sol.states
    x1_z: NDArray[np.float64] = xx_final[-1]
    logger.debug(f"Final state at crossing x1_z = {x1_z}")

    return t1_z, x1_z


def _find_z_zero_crossing(dynsys: _DynamicalSystem, x0: NDArray[np.float64], forward: int = 1) -> Tuple[float, NDArray[np.float64]]:
    r"""
    Find the time and state at which an orbit next crosses the z=0 plane.
    
    This function propagates a trajectory from an initial state and determines
    when it next crosses the z=0 plane (i.e., the x-y plane). This is particularly
    useful for periodic orbit computations where the orbit is symmetric about the
    x-y plane.
    
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
    
    Returns
    -------
    t1_z : float
        Time at which the orbit crosses the z=0 plane
    x1_z : npt.NDArray[np.float64]
        State vector [x, y, z, vx, vy, vz] at the crossing
    
    Notes
    -----
    The function uses a two-step approach:
    1. First integrating to an approximate time where the crossing is expected (π/2 - 0.15)
    2. Then using a root-finding method to precisely locate the crossing time
    
    This hybrid approach is more efficient than using a very fine integration
    time step, especially for hiten.system.orbits with long periods.
    """
    logger.debug(f"Entering _find_z_zero_crossing with x0={x0}, forward={forward}")
    t0_z = np.pi/2 - 0.15
    logger.debug(f"Initial time guess t0_z = {t0_z}")

    # 1) Integrate from t=0 up to t0_z.
    logger.debug(f"Propagating from t=0 to t={t0_z}")
    sol = _propagate_dynsys(dynsys, x0, 0.0, t0_z, forward=forward)
    xx = sol.states
    x0_z: NDArray[np.float64] = xx[-1]
    logger.debug(f"State after initial propagation x0_z = {x0_z}")

    # 2) Define a local function that depends on time t.
    def z_component_wrapper(t: float) -> float:
        r"""
        Wrapper function that returns the z-coordinate of the orbit at time t.
        
        This function is used as the target for root-finding, since we want to
        find where z=0 (i.e., when the orbit crosses the x-y plane).
        
        Parameters
        ----------
        t : float
            Time to evaluate the orbit
        
        Returns
        -------
        float
            The z-coordinate of the orbit at time t
        """
        return _z_component(dynsys, t, t0_z, x0_z, forward=forward, steps=500)

    # 3) Find the time at which z=0 by bracketing the root.
    logger.debug(f"Finding root bracket starting from t0_z = {t0_z}")
    t1_z = _find_bracket(z_component_wrapper, t0_z)
    logger.debug(f"Found crossing time t1_z = {t1_z}")

    # 4) Integrate from t0_z to t1_z to get the final state.
    logger.debug(f"Propagating from t={t0_z} to t={t1_z} to get final state")
    sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1_z, forward=forward)
    xx_final = sol.states
    x1_z: NDArray[np.float64] = xx_final[-1]
    logger.debug(f"Final state at crossing x1_z = {x1_z}")

    return t1_z, x1_z

def _find_bracket(f: Callable[[float], float], x0: float, max_expand: int = 500) -> float:
    r"""
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


def _y_component(dynsys: _DynamicalSystem, t1: float, t0_z: float, x0_z: NDArray[np.float64], forward: int = 1, steps: int = 3000) -> float:
    r"""
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
    logger.debug(f"Entering _y_component: t1={t1}, t0_z={t0_z}, x0_z={x0_z}, forward={forward}")
    # If t1 == t0_z, no integration is done.  Just take the initial condition.
    if np.isclose(t1, t0_z, rtol=3e-10, atol=1e-10):
        x1_zgl = x0_z
        logger.debug(f"t1 ({t1}) is close to t0_z ({t0_z}). Returning y-component from x0_z: {x1_zgl[1]}")
    else:
        logger.debug(f"Propagating from t={t0_z} to t={t1}")
        sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1, forward=forward, steps=steps)
        xx = sol.states
        # The final state is the last row of xx
        x1_zgl: NDArray[np.float64] = xx[-1, :]
        logger.debug(f"Propagation finished. Final state x1_zgl = {x1_zgl}. Returning y-component: {x1_zgl[1]}")

    return float(x1_zgl[1]) # Explicitly cast to float


def _x_component(dynsys: _DynamicalSystem, t1: float, t0_z: float, x0_z: NDArray[np.float64], forward: int = 1, steps: int = 3000) -> float:
    r"""
    Compute the x-component of an orbit at a specified time.
    
    This function propagates an orbit from a reference state and time to a
    target time, and returns the x-component of the resulting state. It is
    designed to be used for finding orbital plane crossings.
    
    Parameters
    ----------
    t1 : float
        Target time at which to evaluate the x-component
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
        The x-component of the orbit state at time t1
    
    Notes
    -----
    This function is primarily used within root-finding algorithms to locate
    precise times when the orbit crosses the x=0 plane. It avoids unnecessary
    computation when t1 is very close to t0_z by simply returning the x-component
    of the reference state in that case.
    """
    logger.debug(f"Entering _x_component: t1={t1}, t0_z={t0_z}, x0_z={x0_z}, forward={forward}")
    # If t1 == t0_z, no integration is done.  Just take the initial condition.
    if np.isclose(t1, t0_z, rtol=3e-10, atol=1e-10):
        x1_zgl = x0_z
        logger.debug(f"t1 ({t1}) is close to t0_z ({t0_z}). Returning x-component from x0_z: {x1_zgl[0]}")
    else:
        logger.debug(f"Propagating from t={t0_z} to t={t1}")
        sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1, forward=forward, steps=steps)
        xx = sol.states
        # The final state is the last row of xx
        x1_zgl: NDArray[np.float64] = xx[-1, :]
        logger.debug(f"Propagation finished. Final state x1_zgl = {x1_zgl}. Returning x-component: {x1_zgl[0]}")

    return float(x1_zgl[0]) # Explicitly cast to float


def _z_component(dynsys: _DynamicalSystem, t1: float, t0_z: float, x0_z: NDArray[np.float64], forward: int = 1, steps: int = 3000, tol: float = 1e-10) -> float:
    r"""
    Compute the z-component of an orbit at a specified time.
    
    This function propagates an orbit from a reference state and time to a
    target time, and returns the z-component of the resulting state. It is
    designed to be used for finding orbital plane crossings.
    
    Parameters
    ----------
    t1 : float
        Target time at which to evaluate the z-component
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
        The z-component of the orbit state at time t1
    
    Notes
    -----
    This function is primarily used within root-finding algorithms to locate
    precise times when the orbit crosses the z=0 plane. It avoids unnecessary
    computation when t1 is very close to t0_z by simply returning the z-component
    of the reference state in that case.
    """
    logger.debug(f"Entering _z_component: t1={t1}, t0_z={t0_z}, x0_z={x0_z}, forward={forward}")
    # If t1 == t0_z, no integration is done.  Just take the initial condition.
    if np.isclose(t1, t0_z, rtol=3e-10, atol=1e-10):
        x1_zgl = x0_z
        logger.debug(f"t1 ({t1}) is close to t0_z ({t0_z}). Returning z-component from x0_z: {x1_zgl[2]}")
    else:
        logger.debug(f"Propagating from t={t0_z} to t={t1}")
        sol = _propagate_dynsys(dynsys, x0_z, t0_z, t1, forward=forward, steps=steps)
        xx = sol.states
        # The final state is the last row of xx
        x1_zgl: NDArray[np.float64] = xx[-1, :]
        logger.debug(f"Propagation finished. Final state x1_zgl = {x1_zgl}. Returning z-component: {x1_zgl[2]}")

    return float(x1_zgl[2]) # Explicitly cast to float


def surface_of_section(X, T, mu, M=1, C=1):
    r"""
    Compute the surface-of-section for the CR3BP at specified plane crossings.
    
    This function identifies and computes the points where a trajectory crosses
    a specified plane in the phase space, creating a Poincaré section that is
    useful for analyzing the structure of the dynamics.
    
    Parameters
    ----------
    X : ndarray
        State trajectory with shape (n_points, state_dim), where each row is a
        state vector (positions and velocities), with columns representing
        [x, y, z, vx, vy, vz]
    T : ndarray
        Time stamps corresponding to the points in the state trajectory, with shape (n_points,)
    mu : float
        CR3BP mass parameter (mass ratio of smaller body to total mass)
    M : {0, 1, 2}, optional
        Determines which plane to use for the section:
        * 0: x = 0 (center-of-mass plane)
        * 1: x = -mu (larger primary plane) (default)
        * 2: x = 1-mu (smaller primary plane)
    C : {-1, 0, 1}, optional
        Crossing condition on y-coordinate:
        * 1: accept crossings with y >= 0 (default)
        * -1: accept crossings with y <= 0
        * 0: accept both y >= 0 and y <= 0
    
    Returns
    -------
    Xy0 : ndarray
        Array of state vectors at the crossing points, with shape (n_crossings, state_dim)
    Ty0 : ndarray
        Array of times corresponding to the crossing points, with shape (n_crossings,)
    
    Notes
    -----
    The function detects sign changes in the shifted x-coordinate to identify
    crossings. For M=2, it uses higher-resolution interpolation to more precisely
    locate the crossing points.
    
    Crossings are only kept if they satisfy the condition C*y >= 0, allowing
    selection of crossings in specific regions of phase space.
    """
    RES = 50  # Resolution for interpolation when M=2

    try:
        # Input validation
        if not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 6:
            logger.error(f"Invalid trajectory data: shape {X.shape if hasattr(X, 'shape') else 'unknown'}")
            return np.array([]), np.array([])
            
        if not isinstance(T, np.ndarray) or T.ndim != 1 or T.size != X.shape[0]:
            logger.error(f"Invalid time data: shape {T.shape if hasattr(T, 'shape') else 'unknown'}")
            return np.array([]), np.array([])
        
        if M not in [0, 1, 2]:
            logger.error(f"Invalid plane selector M={M}, must be 0, 1, or 2")
            return np.array([]), np.array([])
            
        if C not in [-1, 0, 1]:
            logger.error(f"Invalid crossing condition C={C}, must be -1, 0, or 1")
            return np.array([]), np.array([])

        # Determine the shift d based on M
        if M == 1:
            d = -mu
        elif M == 2:
            d = 1 - mu
        elif M == 0:
            d = 0
        
        # Copy to avoid modifying the original data
        X_copy = np.array(X, copy=True)
        T_copy = np.array(T)
        n_rows, n_cols = X_copy.shape
        
        # Shift the x-coordinate by subtracting d
        X_copy[:, 0] = X_copy[:, 0] - d
    
        # Prepare lists to hold crossing states and times
        Xy0_list = []
        Ty0_list = []
        
        if M == 1 or M == 0:
            # For M == 0 or M == 1, use the original data points
            for k in range(n_rows - 1):
                # Check if there is a sign change in the x-coordinate
                if X_copy[k, 0] * X_copy[k+1, 0] <= 0:  # Sign change or zero crossing
                    # Check the condition on y (C*y >= 0)
                    if C == 0 or np.sign(C * X_copy[k, 1]) >= 0:
                        # Choose the point with x closer to zero (to the plane)
                        K = k if abs(X_copy[k, 0]) < abs(X_copy[k+1, 0]) else k+1
                        Xy0_list.append(X[K, :])  # Use original X, not X_copy
                        Ty0_list.append(T[K])
        
        elif M == 2:
            # For M == 2, refine the crossing using interpolation
            for k in range(n_rows - 1):
                # Check if there is a sign change in the x-coordinate
                if X_copy[k, 0] * X_copy[k+1, 0] <= 0:  # Sign change or zero crossing
                    # Interpolate between the two points with increased resolution
                    dt_segment = abs(T[k+1] - T[k]) / RES
                    
                    # Make sure we have enough points for interpolation
                    if dt_segment > 0:
                        try:
                            # Use trajectory interpolation
                            XX, TT = _interpolate(X[k:k+2, :], T[k:k+2], dt_segment)
                            
                            # Also compute the shifted X values
                            XX_shifted = XX.copy()
                            XX_shifted[:, 0] = XX[:, 0] - d
                            
                            # Look through the interpolated points for the crossing
                            found_valid_crossing = False
                            for kk in range(len(TT) - 1):
                                if XX_shifted[kk, 0] * XX_shifted[kk+1, 0] <= 0:
                                    if C == 0 or np.sign(C * XX_shifted[kk, 1]) >= 0:
                                        # Choose the interpolated point closer to the plane
                                        K = kk if abs(XX_shifted[kk, 0]) < abs(XX_shifted[kk+1, 0]) else kk+1
                                        Xy0_list.append(XX[K, :])
                                        Ty0_list.append(TT[K])
                                        found_valid_crossing = True
                            
                            if not found_valid_crossing:
                                logger.debug(f"No valid crossing found after interpolation at t={T[k]:.3f}")
                        except Exception as e:
                            # logger.warning(f"Interpolation failed at t={T[k]:.3f}: {str(e)}")
                            # Fallback to original point
                            K = k if abs(X_copy[k, 0]) < abs(X_copy[k+1, 0]) else k+1
                            if C == 0 or np.sign(C * X_copy[K, 1]) >= 0:
                                Xy0_list.append(X[K, :])
                                Ty0_list.append(T[K])
        
        # Convert lists to arrays
        Xy0 = np.array(Xy0_list)
        Ty0 = np.array(Ty0_list)
        
        logger.debug(f"Found {len(Xy0)} crossings for M={M}, C={C}")
        return Xy0, Ty0
    
    except Exception as e:
        logger.error(f"Error in surface_of_section: {str(e)}", exc_info=True)
        return np.array([]), np.array([]) 


def _interpolate(x, t=None, dt=None):
    r"""
    Function with dual behavior:
    1. When called with 3 arguments like _interpolate(x1, x2, s), performs simple linear interpolation
       between two points x1 and x2 with parameter s.
    2. When called with trajectory data _interpolate(x, t, dt), resamples a trajectory using cubic splines.
    
    Parameters
    ----------
    x : ndarray
        Either:
        * First argument: Data to interpolate for trajectory resampling
        * First point x1 for simple interpolation
    t : ndarray or float, optional
        Either:
        * Time points for trajectory resampling
        * Second point x2 for simple interpolation
    dt : float or int, optional
        Either:
        * Time step or number of points for trajectory resampling
        * Interpolation parameter s for simple interpolation
    
    Returns
    -------
    X : ndarray or tuple
        Either:
        * Interpolated point between x1 and x2 (for simple interpolation)
        * Tuple (X, T) of resampled trajectory and time vector (for trajectory resampling)
    
    Notes
    -----
    This function determines which behavior to use based on the number and types
    of arguments provided. For backward compatibility, it supports both the original
    trajectory resampling behavior and the simple point interpolation used in
    surface_of_section calculations.
    """
    # Special case: When called with 3 arguments from surface_of_section
    # Using pattern: _interpolate(X1, X2, s)
    # where s is a scalar in [0, 1]
    if dt is not None and np.isscalar(dt) and (0 <= dt <= 1):
        # This is simple linear interpolation
        # x = x1, t = x2, dt = s
        s = dt
        x1 = x
        x2 = t
        
        # Ensure s is in [0, 1]
        s = max(0, min(1, s))
        
        # Simple linear interpolation
        return x1 + s * (x2 - x1)
        
    # Original trajectory resampling case
    t = np.asarray(t) if t is not None else None
    x = np.asarray(x)
    
    # Default dt if not provided
    if dt is None:
        dt = 0.05 * 2 * np.pi
    
    # Handle special cases for t
    if t is None or len(t) < 2:
        return x  # Can't interpolate

    # If dt > 10, then treat dt as number of points (N) and recalc dt
    if dt > 10:
        N = int(dt)
        dt = (np.max(t) - np.min(t)) / (N - 1)
    
    # Adjust time vector if it spans negative and positive values
    NEG = 1 if (np.min(t) < 0 and np.max(t) > 0) else 0
    tt = np.abs(t - NEG * np.min(t))
    
    # Create new evenly spaced time vector for the interpolation domain
    TT = np.arange(tt[0], tt[-1] + dt/10, dt)
    # Recover the correct "arrow of time"
    T = np.sign(t[-1]) * TT + NEG * np.min(t)
    
    # Interpolate each column using cubic spline interpolation
    if x.ndim == 1:
        # For a single-dimensional x, treat as a single column
        cs = CubicSpline(tt, x)
        X = cs(TT)
    else:
        m, n = x.shape
        X = np.zeros((len(TT), n))
        for i in range(n):
            cs = CubicSpline(tt, x[:, i])
            X[:, i] = cs(TT)
    
    return X, T
