"""
Energy computation functions for the Circular Restricted Three-Body Problem (CR3BP).

This module provides functions for calculating and analyzing energies and
related quantities in the CR3BP, including:
- Computing the energy (Hamiltonian) of a state
- Converting between energy and Jacobi constant
- Determining energy bounds for different regimes of motion
- Computing the potential and effective potential
"""

from typing import Sequence, Tuple

import numpy as np

from utils.log_config import logger


def crtbp_energy(state: Sequence[float], mu: float) -> float:
    """
    Compute the energy (Hamiltonian) of a state in the CR3BP.
    
    This function calculates the total energy of a given state in the CR3BP,
    which is a conserved quantity in the rotating frame and is related to
    the Jacobi constant by C = -2E.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The energy value (scalar)
    
    Notes
    -----
    The energy in the rotating frame consists of the kinetic energy plus
    the effective potential, which includes the gravitational potential and
    the centrifugal potential. This is a conserved quantity along any trajectory
    in the CR3BP.
    """
    logger.debug(f"Computing energy for state={state}, mu={mu}")
    
    x, y, z, vx, vy, vz = state
    mu1 = 1.0 - mu
    mu2 = mu
    
    r1 = np.sqrt((x + mu2)**2 + y**2 + z**2)
    r2 = np.sqrt((x - mu1)**2 + y**2 + z**2)
    
    # Log a warning if we're close to a singularity
    min_distance = 1e-10
    if r1 < min_distance or r2 < min_distance:
        logger.warning(f"Very close to a primary body: r1={r1}, r2={r2}")
    
    kin = 0.5 * (vx*vx + vy*vy + vz*vz)
    pot = -(mu1 / r1) - (mu2 / r2) - 0.5*(x*x + y*y + z*z) - 0.5*mu1*mu2
    
    result = kin + pot
    logger.debug(f"Energy calculated: {result}")
    return result

def hill_region(
    mu: float, 
    C: float, 
    x_range: Tuple[float, float] = (-1.5, 1.5), 
    y_range: Tuple[float, float] = (-1.5, 1.5), 
    n_grid: int = 400
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Hill region (zero-velocity curves) for a given Jacobi constant.
    
    This function calculates the regions in the x-y plane that are accessible
    to an orbit with a specific Jacobi constant. The boundaries of these regions
    are the zero-velocity curves where the kinetic energy is exactly zero.
    
    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    C : float
        Jacobi constant value
    x_range : tuple, optional
        Range of x values to compute (xmin, xmax). Default is (-1.5, 1.5).
    y_range : tuple, optional
        Range of y values to compute (ymin, ymax). Default is (-1.5, 1.5).
    n_grid : int, optional
        Number of grid points in each dimension. Default is 400.
    
    Returns
    -------
    X : ndarray
        2D array of x-coordinates
    Y : ndarray
        2D array of y-coordinates
    Z : ndarray
        2D array of values where Z > 0 indicates forbidden regions and
        Z ≤ 0 indicates allowed regions
    
    Notes
    -----
    The Hill region calculation is a powerful tool for visualizing the
    accessible regions of phase space. Points where Z > 0 are inaccessible
    (forbidden) to an orbit with the given Jacobi constant, while points
    where Z ≤ 0 are accessible.
    """
    logger.info(f"Computing Hill region for mu={mu}, C={C}, grid={n_grid}x{n_grid}")
    logger.debug(f"x_range={x_range}, y_range={y_range}")
    
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)

    r1 = np.sqrt((X + mu)**2 + Y**2)
    r2 = np.sqrt((X - 1 + mu)**2 + Y**2)

    Omega = (1 - mu) / r1 + mu / r2 + 0.5 * (X**2 + Y**2)

    Z = Omega - C/2
    
    logger.debug(f"Hill region computation complete. Z shape: {Z.shape}")
    return X, Y, Z

def energy_to_jacobi(energy: float) -> float:
    """
    Convert energy to Jacobi constant.
    
    The Jacobi constant C is related to the energy E by C = -2E.
    
    Parameters
    ----------
    energy : float
        Energy value
    
    Returns
    -------
    float
        Corresponding Jacobi constant
    """
    jacobi = -2 * energy
    logger.debug(f"Converted energy {energy} to Jacobi constant {jacobi}")
    return jacobi


def jacobi_to_energy(jacobi: float) -> float:
    """
    Convert Jacobi constant to energy.
    
    The energy E is related to the Jacobi constant C by E = -C/2.
    
    Parameters
    ----------
    jacobi : float
        Jacobi constant value
    
    Returns
    -------
    float
        Corresponding energy value
    """
    energy = -jacobi / 2
    logger.debug(f"Converted Jacobi constant {jacobi} to energy {energy}")
    return energy


def kinetic_energy(state: Sequence[float]) -> float:
    """
    Compute the kinetic energy of a state.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    
    Returns
    -------
    float
        Kinetic energy value
    """
    x, y, z, vx, vy, vz = state
    
    result = 0.5 * (vx**2 + vy**2 + vz**2)
    logger.debug(f"Kinetic energy for state={state}: {result}")
    return result


def effective_potential(state: Sequence[float], mu: float) -> float:
    """
    Compute the effective potential of a state in the CR3BP.
    
    The effective potential includes both the gravitational potential
    and the centrifugal potential in the rotating frame.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The effective potential value
    
    Notes
    -----
    The effective potential is the sum of the gravitational potential
    and the centrifugal potential. It determines the shape of the
    zero-velocity curves and the location of the libration points.
    """
    logger.debug(f"Computing effective potential for state={state}, mu={mu}")
    
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = primary_distance(state, mu)
    r2 = secondary_distance(state, mu)
    
    min_distance = 1e-10
    if r1 < min_distance or r2 < min_distance:
        logger.warning(f"Very close to a primary body: r1={r1}, r2={r2}")
    
    U = gravitational_potential(state, mu)
    U_eff = -0.5 * (x**2 + y**2 + z**2) + U
    logger.debug(f"Effective potential calculated: {U_eff}")
    
    return U_eff


def pseudo_potential_at_point(x: float, y: float, mu: float) -> float:
    """
    Compute the pseudo-potential (Omega) at a point (x,y) in the rotating frame.
    
    The pseudo-potential Ω is the function whose negative gradient gives
    the forces in the rotating frame, excluding the Coriolis force.
    Note that Ω = -U_eff (ignoring constant terms and z-dimension).
    
    Parameters
    ----------
    x : float
        x-coordinate in the rotating frame
    y : float
        y-coordinate in the rotating frame
    mu : float
        Mass parameter of the CR3BP
    
    Returns
    -------
    float
        The pseudo-potential (Omega) value
    
    Notes
    -----
    This function calculates the commonly used pseudo-potential Ω rather than U_eff
    (which is computed by effective_potential() and used in the Hamiltonian).
    The sign convention is Ω = 0.5*(x^2+y^2) + (1-mu)/r1 + mu/r2, which is
    essentially the negative of the effective potential used in the Hamiltonian.
    """
    logger.debug(f"Computing pseudo-potential at point x={x}, y={y}, mu={mu}")
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def gravitational_potential(state: Sequence[float], mu: float) -> float:
    """
    Compute the gravitational potential of a state in the CR3BP.
    
    This function calculates the gravitational potential energy due
    to the two primary bodies in the CR3BP.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        The gravitational potential value
    """
    logger.debug(f"Computing gravitational potential for state={state}, mu={mu}")
    
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    mu_2 = mu
    r1 = primary_distance(state, mu)
    r2 = secondary_distance(state, mu)
    U = -mu_1 / r1 - mu_2 / r2 - 0.5 * mu_1 * mu_2
    return U


def primary_distance(state: Sequence[float], mu: float) -> float:
    """
    Compute the distance from a state to the primary body.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        Distance to the primary body
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing primary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_2 = mu
    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    return r1


def secondary_distance(state: Sequence[float], mu: float) -> float:
    """
    Compute the distance from a state to the secondary body.
    
    Parameters
    ----------
    state : Sequence[float]
        State vector [x, y, z, vx, vy, vz] in the rotating frame
    mu : float
        Mass parameter of the CR3BP system (ratio of smaller to total mass)
    
    Returns
    -------
    float
        Distance to the secondary body
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing secondary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)
    return r2 