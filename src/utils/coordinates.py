import numpy as np


def rotating_to_inertial(state, t, mu):
    """
    Convert state from rotating to inertial frame.
    
    Parameters
    ----------
    state : array-like
        The state vector [x, y, z, vx, vy, vz] in rotating frame.
    t : float
        The time value (used for rotation angle).
    mu : float
        The mass parameter of the system.
        
    Returns
    -------
    numpy.ndarray
        The state vector in inertial frame.
    """
    # Extract position and velocity components
    x, y, z, vx, vy, vz = state
    
    # Rotation matrix (R) for position conversion
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    
    # Position in inertial frame
    pos_rot = np.array([x, y, z])
    pos_inertial = R @ pos_rot
    
    # For velocity, need to account for both rotation of coordinates and angular velocity
    # Angular velocity term
    omega_cross_r = np.array([
        -y,
        x,
        0
    ])
    
    # Velocity in rotating frame
    vel_rot = np.array([vx, vy, vz])
    
    # Velocity in inertial frame = R·(v_rot + ω×r)
    vel_inertial = R @ (vel_rot + omega_cross_r)
    
    # Combine position and velocity
    return np.concatenate([pos_inertial, vel_inertial])
