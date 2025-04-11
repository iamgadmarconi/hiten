import numpy as np
from system.libration import CollinearPoint, L1Point, L2Point, L3Point


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


def inertial_to_rotating(state, t, mu):
    """
    Convert state from inertial to rotating frame.
    
    Parameters
    ----------
    state : array-like
        The state vector [x, y, z, vx, vy, vz] in inertial frame.
    t : float
        The time value (used for rotation angle).
    mu : float
        The mass parameter of the system.
        
    Returns
    -------
    numpy.ndarray
        The state vector in rotating frame.
    """
    # Extract position and velocity components
    x, y, z, vx, vy, vz = state
    
    # Rotation matrix (R) for position conversion
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    R = np.array([
        [cos_t, sin_t, 0],
        [-sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    
    # Position in inertial frame
    pos_inertial = np.array([x, y, z])
    pos_rotating = R @ pos_inertial
    
    # For velocity, need to account for both rotation of coordinates and angular velocity
    # Angular velocity term
    omega_cross_r = np.array([
        -y,
        x,
        0
    ])
    
    # Velocity in inertial frame
    vel_inertial = np.array([vx, vy, vz])
    
    # Velocity in rotating frame = R^T·(v_inertial - ω×r)
    vel_rotating = R.T @ (vel_inertial - omega_cross_r)
    
    # Combine position and velocity
    return np.concatenate([pos_rotating, vel_rotating])


def standard_to_centered(state, mu, libration_point: CollinearPoint):
    """
    Transform from standard CR3BP coordinates to centered coordinates.
    
    Args:
        state: Array with position and velocity [x, y, z, vx, vy, vz] in standard coordinates
        mu: Mass parameter
        libration_point: CollinearPoint instance indicating which libration point to use.
        
    Returns:
        State in centered coordinates [X, Y, Z, VX, VY, VZ]
    """
    x, y, z, vx, vy, vz = state

    # Get gamma directly from the libration point instance
    gamma = libration_point.gamma

    # Calculate 'a' and 'sign' parameters based on libration point type
    if isinstance(libration_point, L1Point):
        a = -1 + gamma
        sign = 1  # "upper sign" (+)
    elif isinstance(libration_point, L2Point):
        a = -1 - gamma
        sign = -1  # "lower sign" (-)
    elif isinstance(libration_point, L3Point):
        a = gamma # Note: This 'a' definition seems unusual for L3, review needed?
                # Original code implicitly used this logic, preserving it.
        sign = -1  # Sign convention for L3 transformation
    else:
        # This should not happen if the identifier was valid
        raise TypeError(f"Expected L1, L2, or L3 point, got {type(libration_point)}")
    
    # Apply coordinate transformation
    X = sign * gamma * x + mu + a
    Y = sign * gamma * y
    Z = gamma * z
    
    # Apply velocity transformation (scale by gamma)
    VX = sign * gamma * vx
    VY = sign * gamma * vy
    VZ = gamma * vz
    
    return np.array([X, Y, Z, VX, VY, VZ])


def centered_to_standard(state, mu, libration_point: CollinearPoint):
    """
    Transform from centered coordinates back to standard CR3BP coordinates.
    
    Args:
        state: Array with position and velocity [X, Y, Z, VX, VY, VZ] in centered coordinates
        mu: Mass parameter
        libration_point: CollinearPoint instance indicating which libration point to use.
        
    Returns:
        State in standard coordinates [x, y, z, vx, vy, vz]
    """
    X, Y, Z, VX, VY, VZ = state

    # Get gamma directly from the libration point instance
    gamma = libration_point.gamma
    
    # Calculate 'a' and 'sign' parameters based on libration point type
    if isinstance(libration_point, L1Point):
        a = -1 + gamma
        sign = 1  # "upper sign" (+)
    elif isinstance(libration_point, L2Point):
        a = -1 - gamma
        sign = -1  # "lower sign" (-)
    elif isinstance(libration_point, L3Point):
        a = gamma # Preserving original logic, review if needed
        sign = -1  # Sign convention for L3 transformation
    else:
        raise TypeError(f"Expected L1, L2, or L3 point, got {type(libration_point)}")
    
    # Apply inverse coordinate transformation
    x = (X - mu - a) / (sign * gamma)
    y = Y / (sign * gamma)
    z = Z / gamma
    
    # Apply inverse velocity transformation
    vx = VX / (sign * gamma)
    vy = VY / (sign * gamma)
    vz = VZ / gamma
    
    return np.array([x, y, z, vx, vy, vz])


