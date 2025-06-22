import numpy as np

from hiten.utils.constants import Constants


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
        The mass parameter of the hiten.system.
        
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
        The mass parameter of the hiten.system.
        
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

def _get_mass_parameter(primary_mass, secondary_mass):
    """
    Calculate the mass parameter μ for the CR3BP.
    
    The mass parameter μ is defined as the ratio of the secondary mass
    to the total system mass: μ = m₂/(m₁ + m₂).
    
    Parameters
    ----------
    primary_mass : float
        Mass of the primary body (m₁) in kilograms
    secondary_mass : float
        Mass of the secondary body (m₂) in kilograms
    
    Returns
    -------
    float
        Mass parameter μ (dimensionless)
    """
    return secondary_mass / (primary_mass + secondary_mass)

def _get_angular_velocity(primary_mass, secondary_mass, distance):
    """
    Calculate the mean motion (angular velocity) of the CR3BP.
    
    Computes the angular velocity at which the two primary bodies
    orbit around their common barycenter in a circular orbit.
    
    Parameters
    ----------
    primary_mass : float
        Mass of the primary body in kilograms
    secondary_mass : float
        Mass of the secondary body in kilograms
    distance : float
        Distance between the two bodies in meters
    
    Returns
    -------
    float
        Angular velocity in radians per second
        
    Notes
    -----
    This is calculated using Kepler's Third Law: ω² = G(m₁+m₂)/r³
    where G is the gravitational constant, m₁ and m₂ are the masses,
    and r is the distance between the bodies.
    """
    return np.sqrt(Constants.G * (primary_mass + secondary_mass) / distance**3)

def to_crtbp_units(state_si, m1, m2, distance):
    """
    Convert an SI-state vector into the dimensionless state used by crtbp_accel.
    
    Parameters
    ----------
    state_si  : array-like of shape (6,)
        [x, y, z, vx, vy, vz] in meters and meters/sec, all in Earth-centered coordinates
        if you're modeling Earth-Moon, for example.
    m1        : float
        Mass of primary (e.g., Earth) in kilograms.
    m2        : float
        Mass of secondary (e.g., Moon) in kilograms.
    distance  : float
        Distance between the two main bodies (e.g., Earth-Moon) in meters.
        
    Returns
    -------
    state_dimless : np.ndarray of shape (6,)
        The dimensionless state vector suitable for crtbp_accel.
    mu            : float
        Dimensionless mass parameter = m2 / (m1 + m2).
    """
    # Mean motion (rad/s) => in CRTBP, we want n = 1, so we scale by this factor.
    n = _get_angular_velocity(m1, m2, distance)

    # Compute the dimensionless mass parameter
    mu = _get_mass_parameter(m1, m2)

    # Position scaled by the chosen distance
    x_star = state_si[0] / distance
    y_star = state_si[1] / distance
    z_star = state_si[2] / distance

    # Velocity scaled by distance * n
    vx_star = state_si[3] / (distance * n)
    vy_star = state_si[4] / (distance * n)
    vz_star = state_si[5] / (distance * n)

    state_dimless = np.array([x_star, y_star, z_star, vx_star, vy_star, vz_star], dtype=np.float64)
    return state_dimless

def to_si_units(state_dimless, m1, m2, distance):
    """
    Convert a dimensionless state vector into the SI-state vector used by crtbp_accel.

    Parameters
    ----------
    state_dimless : np.ndarray of shape (6,)
        The dimensionless state vector suitable for crtbp_accel.
    m1        : float
        Mass of primary (e.g., Earth) in kilograms.
    m2        : float
        Mass of secondary (e.g., Moon) in kilograms.
    distance  : float
        Distance between the two main bodies (e.g., Earth-Moon) in meters.

    Returns
    -------
    state_si : np.ndarray of shape (6,)
        The SI-state vector suitable for crtbp_accel.
    """
    n = _get_angular_velocity(m1, m2, distance)

    x = state_dimless[0] * distance
    y = state_dimless[1] * distance
    z = state_dimless[2] * distance

    vx = state_dimless[3] * distance * n
    vy = state_dimless[4] * distance * n
    vz = state_dimless[5] * distance * n

    return np.array([x, y, z, vx, vy, vz], dtype=np.float64)

def dimless_time(T, m1, m2, distance):
    """
    Convert time from SI units (seconds) to dimensionless CR3BP time units.
    
    Parameters
    ----------
    T : float
        Time in seconds
    m1 : float
        Mass of primary body in kilograms
    m2 : float
        Mass of secondary body in kilograms
    distance : float
        Distance between the two bodies in meters
        
    Returns
    -------
    float
        Time in dimensionless CR3BP units
        
    Notes
    -----
    In the CR3BP, the time unit is chosen such that the mean motion
    is equal to 1, which means one dimensionless time unit corresponds
    to 1/n seconds, where n is the angular velocity in rad/s.
    """
    n = _get_angular_velocity(m1, m2, distance)
    return T * n


def si_time(T_dimless, m1, m2, distance):
    """
    Convert time from dimensionless CR3BP time units to SI units (seconds).
    
    Parameters
    ----------
    T_dimless : float
        Time in dimensionless CR3BP units
    m1 : float
        Mass of primary body in kilograms
    m2 : float
        Mass of secondary body in kilograms
    distance : float
        Distance between the two bodies in meters
        
    Returns
    -------
    float
        Time in seconds
        
    Notes
    -----
    This is the inverse operation of dimless_time().
    """
    n = _get_angular_velocity(m1, m2, distance)
    return T_dimless / n


def get_distance(state_1_nondim, state_0_nondim, system_distance):
    """
    Calculate physical distance between two bodies in meters.
    
    Parameters
    ----------
    state_1_nondim : np.ndarray[6,]
        First body's dimensionless state vector
    state_0_nondim : np.ndarray[6,]
        Second body's dimensionless state vector  
    system_distance : float
        Actual distance between primary bodies in meters (conversion factor)
        
    Returns
    -------
    float
        Physical distance between bodies in meters
    """
    # Get position components (first 3 elements) from dimensionless states
    pos_diff = state_1_nondim[:3] - state_0_nondim[:3]
    
    # Calculate dimensionless distance (normalized by system_distance)
    dimless_dist = np.linalg.norm(pos_diff)
    
    # Convert to physical distance in meters
    return dimless_dist * system_distance
