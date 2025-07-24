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
import numpy as np
from scipy.interpolate import CubicSpline

from hiten.utils.log_config import logger


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
