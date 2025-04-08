"""
Dynamical equations for the Circular Restricted Three-Body Problem (CR3BP).

This module provides the core differential equations and their derivatives for 
the CR3BP, including:

1. Acceleration equations for state propagation
2. Jacobian matrices for stability analysis
3. Variational equations for state transition matrix computation

These components are essential for numerical integration, stability analysis,
and differential correction in the CR3BP. The implementation uses Numba for
performance optimization, making these computations suitable for intensive
numerical simulations.
"""

import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def crtbp_accel(state, mu):
    """
    Calculate the state derivative (acceleration) for the CR3BP.
    
    This function computes the time derivative of the state vector in the 
    Circular Restricted Three-Body Problem. It returns the velocity and 
    acceleration components that define the equations of motion in the 
    rotating reference frame.
    
    Parameters
    ----------
    state : array_like
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        Time derivative of the state vector [vx, vy, vz, ax, ay, az]
    
    Notes
    -----
    The equations of motion include gravitational forces from both primaries
    and the Coriolis and centrifugal forces from the rotating reference frame.
    This function is optimized using Numba for high-performance computations.
    """
    x, y, z, vx, vy, vz = state

    # Distances to each primary
    r1 = np.sqrt((x + mu)**2 + y**2 + z**2)      # from m1 at (-mu, 0, 0)
    r2 = np.sqrt((x - (1 - mu))**2 + y**2 + z**2) # from m2 at (1-mu, 0, 0)

    # Accelerations
    ax = 2*vy + x - (1 - mu)*(x + mu) / r1**3 - mu*(x - 1 + mu) / r2**3
    ay = -2*vx + y - (1 - mu)*y / r1**3          - mu*y / r2**3
    az = -(1 - mu)*z / r1**3 - mu*z / r2**3

    return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

@numba.njit(fastmath=True, cache=True)
def jacobian_crtbp(x, y, z, mu):
    """
    Compute the Jacobian matrix for the CR3BP equations of motion.
    
    This function calculates the 6x6 Jacobian matrix (state transition matrix 
    derivative) for the 3D Circular Restricted Three-Body Problem in the 
    rotating reference frame. It's used in stability analysis and for computing
    the variational equations.
    
    Parameters
    ----------
    x : float
        x-coordinate in the rotating frame
    y : float
        y-coordinate in the rotating frame
    z : float
        z-coordinate in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    ndarray
        6x6 Jacobian matrix structured as:
        [ 0    0    0    1     0    0 ]
        [ 0    0    0    0     1    0 ]
        [ 0    0    0    0     0    1 ]
        [ omgxx omgxy omgxz  0     2    0 ]
        [ omgxy omgyy omgyz -2     0    0 ]
        [ omgxz omgyz omgzz  0     0    0 ]
    
    Notes
    -----
    The indices of the matrix correspond to: x=0, y=1, z=2, vx=3, vy=4, vz=5.
    The implementation matches the partial derivatives formulation common in
    astrodynamics literature.
    """

    # As in var3D.m:
    #   mu2 = 1 - mu (big mass fraction)
    mu2 = 1.0 - mu

    # Distances squared to the two primaries
    # r^2 = (x+mu)^2 + y^2 + z^2       (distance^2 to M1, which is at (-mu, 0, 0))
    # R^2 = (x - mu2)^2 + y^2 + z^2    (distance^2 to M2, which is at (1-mu, 0, 0))
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    r5 = r2**2.5
    R3 = R2**1.5
    R5 = R2**2.5

    # From var3D.m, the partial derivatives "omgxx," "omgyy," ...
    omgxx = 1.0 \
        + mu2/r5 * 3.0*(x + mu)**2 \
        + mu  /R5 * 3.0*(x - mu2)**2 \
        - (mu2/r3 + mu/R3)

    omgyy = 1.0 \
        + mu2/r5 * 3.0*(y**2) \
        + mu  /R5 * 3.0*(y**2) \
        - (mu2/r3 + mu/R3)

    omgzz = 0.0 \
        + mu2/r5 * 3.0*(z**2) \
        + mu  /R5 * 3.0*(z**2) \
        - (mu2/r3 + mu/R3)

    omgxy = 3.0*y * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgxz = 3.0*z * ( mu2*(x + mu)/r5 + mu*(x - mu2)/R5 )
    omgyz = 3.0*y*z*( mu2/r5 + mu/R5 )

    # Build the 6x6 matrix F
    F = np.zeros((6, 6), dtype=np.float64)

    # Identity block for velocity wrt position
    F[0, 3] = 1.0  # dx/dvx
    F[1, 4] = 1.0  # dy/dvy
    F[2, 5] = 1.0  # dz/dvz

    # The second derivatives block
    F[3, 0] = omgxx
    F[3, 1] = omgxy
    F[3, 2] = omgxz

    F[4, 0] = omgxy
    F[4, 1] = omgyy
    F[4, 2] = omgyz

    F[5, 0] = omgxz
    F[5, 1] = omgyz
    F[5, 2] = omgzz

    # Coriolis terms
    F[3, 4] = 2.0
    F[4, 3] = -2.0

    return F

@numba.njit(fastmath=True, cache=True)
def variational_equations(t, PHI_vec, mu, forward=1):
    """
    Compute the variational equations for the CR3BP.
    
    This function implements the 3D variational equations for the CR3BP,
    calculating the time derivatives of both the state transition matrix (STM)
    and the state vector simultaneously. It's used for sensitivity analysis,
    differential correction, and stability analysis.
    
    Parameters
    ----------
    t : float
        Current time (not used, but required for ODE integrators)
    PHI_vec : ndarray
        Combined 42-element vector containing:
        - First 36 elements: flattened 6x6 state transition matrix (STM)
        - Last 6 elements: state vector [x, y, z, vx, vy, vz]
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    forward : int, optional
        Direction of integration (1 for forward, -1 for backward). Default is 1.
    
    Returns
    -------
    ndarray
        42-element vector containing:
        - First 36 elements: time derivative of flattened STM (dΦ/dt)
        - Last 6 elements: state derivatives [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
    
    Notes
    -----
    The variational equations integrate the STM Φ according to dΦ/dt = F·Φ,
    where F is the Jacobian of the system. This allows tracking how small 
    perturbations in initial conditions propagate through the system, which is
    essential for targeting and differential correction algorithms.
    """
    # 1) Unpack the STM (first 36) and the state (last 6)
    phi_flat = PHI_vec[:36]
    x_vec    = PHI_vec[36:]  # [x, y, z, vx, vy, vz]

    # Reshape the STM to 6x6
    Phi = phi_flat.reshape((6, 6))

    # Unpack the state
    x, y, z, vx, vy, vz = x_vec

    # 2) Build the 6x6 matrix F from the partial derivatives
    F = jacobian_crtbp(x, y, z, mu)

    # 3) dPhi/dt = F * Phi  (manually done to keep numba happy)
    phidot = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            s = 0.0 
            for k in range(6):
                s += F[i, k] * Phi[k, j]
            phidot[i, j] = s

    # 4) State derivatives, same formula as var3D.m
    #    xdot(4) = x(1) - mu2*( (x+mu)/r3 ) - mu*( (x-mu2)/R3 ) + 2*vy, etc.
    mu2 = 1.0 - mu
    r2 = (x + mu)**2 + y**2 + z**2
    R2 = (x - mu2)**2 + y**2 + z**2
    r3 = r2**1.5
    R3 = R2**1.5

    ax = ( x 
           - mu2*( (x+mu)/r3 ) 
           -  mu*( (x - mu2)/R3 ) 
           + 2.0*vy )
    ay = ( y
           - mu2*( y / r3 )
           -  mu*( y / R3 )
           - 2.0*vx )
    az = ( - mu2*( z / r3 ) 
           - mu  *( z / R3 ) )

    # 5) Build derivative of the 42-vector
    dPHI_vec = np.zeros_like(PHI_vec)

    # First 36 = flattened phidot
    dPHI_vec[:36] = phidot.ravel()

    # Last 6 = [vx, vy, vz, ax, ay, az], each multiplied by 'forward'
    dPHI_vec[36] = forward * vx
    dPHI_vec[37] = forward * vy
    dPHI_vec[38] = forward * vz
    dPHI_vec[39] = forward * ax
    dPHI_vec[40] = forward * ay
    dPHI_vec[41] = forward * az

    return dPHI_vec