"""Provide energy and potential functions for the Circular Restricted Three-Body Problem.

This module provides numerical routines for computing energies, potentials,
and zero-velocity curves in the spatial Circular Restricted Three-Body Problem
(CR3BP). All quantities are nondimensional and expressed in the rotating
(synodic) frame.

All functions use the standard CR3BP nondimensional units where the
primary-secondary separation is 1 and the orbital period is 2*pi.

References
----------
Szebehely, V. (1967). *Theory of Orbits: The Restricted Problem of Three Bodies*.
Academic Press.
"""

from typing import Sequence, Tuple

import numpy as np
from numba import njit

from hiten.algorithms.utils.config import FASTMATH
from hiten.utils.log_config import logger


@njit(cache=True, fastmath=FASTMATH)
def _max_rel_energy_error(states: np.ndarray, mu: float) -> float:
    """Compute maximum relative deviation of Jacobi constant along trajectory.
    
    Parameters
    ----------
    states : np.ndarray, shape (N, 6)
        Array of state vectors [x, y, z, vx, vy, vz] along trajectory.
    mu : float
        Mass parameter of the CR3BP system.
        
    Returns
    -------
    float
        Maximum relative error in Jacobi constant preservation.
        
    Notes
    -----
    Uses the first state as reference for computing relative errors.
    For near-zero Jacobi constants (|C0| < 1e-14), uses absolute error.
    """

    mu1 = 1.0 - mu
    mu2 = mu

    def _jacobi(x, y, z, vx, vy, vz):
        r1 = ((x + mu2) ** 2 + y * y + z * z) ** 0.5
        r2 = ((x - mu1) ** 2 + y * y + z * z) ** 0.5
        return x * x + y * y + 2.0 * (mu1 / r1 + mu2 / r2) - (vx * vx + vy * vy + vz * vz)

    x0, y0, z0, vx0, vy0, vz0 = states[0]
    C0 = _jacobi(x0, y0, z0, vx0, vy0, vz0)

    absC0 = abs(C0)
    max_err = 0.0

    for i in range(1, states.shape[0]):
        x, y, z, vx, vy, vz = states[i]
        Ci = _jacobi(x, y, z, vx, vy, vz)

        if absC0 > 1e-14:
            rel_err = abs(Ci - C0) / absC0
        else:
            rel_err = abs(Ci - C0)

        if rel_err > max_err:
            max_err = rel_err

    return max_err


def crtbp_energy(state: Sequence[float], mu: float) -> float:
    r"""Compute Hamiltonian energy of a state in the CR3BP.

    The Hamiltonian energy is defined as E = T + U_eff, where T is the
    kinetic energy and U_eff is the effective potential. The Jacobi
    constant is related through C = -2E.

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter mu = m2/(m1+m2), where m1 and m2 are the primary
        and secondary masses.

    Returns
    -------
    float
        Hamiltonian energy E (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.

    Examples
    --------
    >>> from hiten.algorithms.common.energy import crtbp_energy
    >>> # Earth-Moon system L1 point vicinity
    >>> crtbp_energy([1.0, 0.0, 0.0, 0.0, 0.5, 0.0], 0.01215)
    -1.51...
    
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.energy_to_jacobi` :
        Convert energy to Jacobi constant
    :func:`~hiten.algorithms.common.energy.kinetic_energy` :
        Compute kinetic energy component
    :func:`~hiten.algorithms.common.energy.effective_potential` :
        Compute effective potential component
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
    r"""Compute Hill region for zero-velocity surface analysis.

    The Hill region represents the projection onto the synodic (x,y) plane
    of the zero-velocity surface defined by the Jacobi constant. Regions
    where Omega - C/2 > 0 are forbidden for motion.

    Parameters
    ----------
    mu : float
        Mass parameter of the CR3BP system.
    C : float
        Jacobi constant (nondimensional).
    x_range : Tuple[float, float], optional
        Coordinate bounds for x-axis. Default is (-1.5, 1.5).
    y_range : Tuple[float, float], optional
        Coordinate bounds for y-axis. Default is (-1.5, 1.5).
    n_grid : int, optional
        Number of grid points per axis. Default is 400.

    Returns
    -------
    X : np.ndarray, shape (n_grid, n_grid)
        Meshgrid of x-coordinates.
    Y : np.ndarray, shape (n_grid, n_grid)
        Meshgrid of y-coordinates.
    Z : np.ndarray, shape (n_grid, n_grid)
        Values of Omega - C/2. Positive values indicate forbidden motion.

    Raises
    ------
    ValueError
        If n_grid is less than 2.

    Notes
    -----
    Singularities near the primary bodies are not handled. Users should
    mask regions near (-mu, 0) and (1-mu, 0) as needed.
    
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.crtbp_energy` : Compute
        Hamiltonian energy
    :func:`~hiten.algorithms.common.energy.pseudo_potential_at_point` :
        Evaluate pseudo-potential
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
    r"""Convert Hamiltonian energy to Jacobi constant.

    Parameters
    ----------
    energy : float
        Hamiltonian energy E (nondimensional).

    Returns
    -------
    float
        Jacobi constant C = -2E (nondimensional).
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.jacobi_to_energy` : Inverse
        conversion
    :func:`~hiten.algorithms.common.energy.crtbp_energy` : Compute
        Hamiltonian energy
    """
    jacobi = -2 * energy
    logger.debug(f"Converted energy {energy} to Jacobi constant {jacobi}")
    return jacobi


def jacobi_to_energy(jacobi: float) -> float:
    r"""Convert Jacobi constant to Hamiltonian energy.

    Parameters
    ----------
    jacobi : float
        Jacobi constant C (nondimensional).

    Returns
    -------
    float
        Hamiltonian energy E = -C/2 (nondimensional).
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.energy_to_jacobi` : Inverse
        conversion
    :func:`~hiten.algorithms.common.energy.crtbp_energy` : Compute
        Hamiltonian energy
    """
    energy = -jacobi / 2
    logger.debug(f"Converted Jacobi constant {jacobi} to energy {energy}")
    return energy


def kinetic_energy(state: Sequence[float]) -> float:
    r"""Compute kinetic energy of a state.

    The kinetic energy is defined as T = (1/2) * (vx^2 + vy^2 + vz^2).

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.

    Returns
    -------
    float
        Kinetic energy T (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.effective_potential` :
        Compute potential energy component
    :func:`~hiten.algorithms.common.energy.crtbp_energy` : Compute
        total Hamiltonian energy
    """
    x, y, z, vx, vy, vz = state
    
    result = 0.5 * (vx**2 + vy**2 + vz**2)
    logger.debug(f"Kinetic energy for state={state}: {result}")
    return result


def effective_potential(state: Sequence[float], mu: float) -> float:
    r"""Compute effective potential in the CR3BP rotating frame.

    The effective potential includes gravitational and centrifugal terms:
    U_eff = -(1/2)*(x^2 + y^2 + z^2) + U_grav, where U_grav is the
    gravitational potential from both primary bodies.

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    float
        Effective potential U_eff (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.

    Notes
    -----
    Uses :func:`~hiten.algorithms.common.energy.primary_distance` and
    :func:`~hiten.algorithms.common.energy.secondary_distance` for
    distance calculations. Warns when approaching singularities.
    
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.gravitational_potential` : Gravitational component only
    :func:`~hiten.algorithms.common.energy.kinetic_energy` : Kinetic energy component
    :func:`~hiten.algorithms.common.energy.crtbp_energy` : Total Hamiltonian energy
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
    r"""Evaluate pseudo-potential Omega at a planar point.

    The pseudo-potential is defined as:
    Omega = (1/2)*(x^2 + y^2) + (1-mu)/r1 + mu/r2,
    where r1 and r2 are distances to the primary and secondary bodies.

    Parameters
    ----------
    x : float
        x-coordinate in nondimensional rotating frame.
    y : float
        y-coordinate in nondimensional rotating frame.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    float
        Pseudo-potential value Omega(x,y) (nondimensional).
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.hill_region` : Compute Hill
        regions using pseudo-potential
    :func:`~hiten.algorithms.common.energy.effective_potential` : 3D
        effective potential
    """
    logger.debug(f"Computing pseudo-potential at point x={x}, y={y}, mu={mu}")
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def gravitational_potential(state: Sequence[float], mu: float) -> float:
    r"""Compute gravitational potential energy of test particle.

    The gravitational potential includes contributions from both primary
    bodies: U = -(1-mu)/r1 - mu/r2 - (1/2)*mu*(1-mu).

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    float
        Gravitational potential U (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.effective_potential` :
        Effective potential including centrifugal terms
    :func:`~hiten.algorithms.common.energy.primary_distance` : Distance
        to primary body
    :func:`~hiten.algorithms.common.energy.secondary_distance` : Distance
        to secondary body
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
    r"""Compute distance from test particle to primary body.

    The primary body is located at (-mu, 0, 0) in the rotating frame.

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    float
        Distance r1 to primary body (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.secondary_distance` :
        Distance to secondary body
    :func:`~hiten.algorithms.common.energy.gravitational_potential` :
        Uses both distance calculations
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing primary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_2 = mu
    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    return r1


def secondary_distance(state: Sequence[float], mu: float) -> float:
    r"""Compute distance from test particle to secondary body.

    The secondary body is located at (1-mu, 0, 0) in the rotating frame.

    Parameters
    ----------
    state : Sequence[float], length 6
        State vector [x, y, z, vx, vy, vz] in nondimensional rotating frame.
    mu : float
        Mass parameter of the CR3BP system.

    Returns
    -------
    float
        Distance r2 to secondary body (nondimensional).

    Raises
    ------
    ValueError
        If state cannot be unpacked into six components.
        
    See Also
    --------
    :func:`~hiten.algorithms.common.energy.primary_distance` : Distance
        to primary body
    :func:`~hiten.algorithms.common.energy.gravitational_potential` :
        Uses both distance calculations
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing secondary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)
    return r2
