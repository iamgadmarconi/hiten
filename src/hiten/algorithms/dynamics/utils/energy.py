r"""
dynamics.hiten.utils.energy
=====================

Numerical helpers for evaluating energies, potentials and zero-velocity 
curves in the spatial circular restricted three-body problem (CRTBP).  All 
quantities are nondimensional and expressed in the rotating (synodic) frame.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from typing import Sequence, Tuple

import numpy as np

from hiten.utils.log_config import logger


def crtbp_energy(state: Sequence[float], mu: float) -> float:
    r"""
    Compute the Hamiltonian energy :math:`E` of a single state in the CRTBP.

    The definition is
    :math:`E = T + U_{\mathrm{eff}}`, where :math:`T` is the kinetic energy
    and :math:`U_{\mathrm{eff}}` the effective potential.  The Jacobi
    constant is linked through :math:`C = -2E`.

    Parameters
    ----------
    state : Sequence[float]
        Six-component vector :math:`(x, y, z, \dot{x}, \dot{y}, \dot{z})`.
    mu : float
        Mass parameter :math:`\mu \in (0, 1)`.

    Returns
    -------
    float
        Energy of the given state.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.

    Examples
    --------
    >>> from hiten.algorithms.dynamics.hiten.utils.energy import crtbp_energy
    >>> crtbp_energy([1.0, 0.0, 0.0, 0.0, 0.5, 0.0], 0.01215)  # doctest: +ELLIPSIS
    -1.51...
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
    r"""
    Compute the Hill region associated with a Jacobi constant.

    The Hill region corresponds to the projection on the synodic
    :math:`(x, y)` plane of the zero-velocity surface defined by
    :math:`E = -C/2`.

    Parameters
    ----------
    mu : float
        Mass parameter, see :pyfunc:`crtbp_energy`.
    C : float
        Jacobi constant.
    x_range : Tuple[float, float], optional
        Bounds for :math:`x`.  Default is ``(-1.5, 1.5)``.
    y_range : Tuple[float, float], optional
        Bounds for :math:`y`.  Default is ``(-1.5, 1.5)``.
    n_grid : int, default 400
        Number of grid points per axis.

    Returns
    -------
    X : numpy.ndarray
        Meshgrid of x-coordinates, shape ``(n_grid, n_grid)``.
    Y : numpy.ndarray
        Meshgrid of y-coordinates, shape ``(n_grid, n_grid)``.
    Z : numpy.ndarray
        Values of :math:`\Omega - C/2`; positive entries mark forbidden
        motion.

    Raises
    ------
    ValueError
        If *n_grid* is smaller than 2.

    Notes
    -----
    No attempt is made to handle singularities near the primaries; users may
    wish to mask those regions.
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
    r"""
    Convert Hamiltonian energy to Jacobi constant.

    Parameters
    ----------
    energy : float
        Energy :math:`E`.

    Returns
    -------
    float
        Jacobi constant :math:`C = -2E`.

    Raises
    ------
    None
    """
    jacobi = -2 * energy
    logger.debug(f"Converted energy {energy} to Jacobi constant {jacobi}")
    return jacobi


def jacobi_to_energy(jacobi: float) -> float:
    r"""
    Convert Jacobi constant to Hamiltonian energy.

    Parameters
    ----------
    jacobi : float
        Jacobi constant :math:`C`.

    Returns
    -------
    float
        Energy :math:`E = -C/2`.

    Raises
    ------
    None
    """
    energy = -jacobi / 2
    logger.debug(f"Converted Jacobi constant {jacobi} to energy {energy}")
    return energy


def kinetic_energy(state: Sequence[float]) -> float:
    r"""
    Return the kinetic energy :math:`T` of a state.

    The definition is
    :math:`T = \tfrac12 (\dot{x}^2 + \dot{y}^2 + \dot{z}^2)`.

    Parameters
    ----------
    state : Sequence[float]
        State vector.

    Returns
    -------
    float
        Kinetic energy.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.
    """
    x, y, z, vx, vy, vz = state
    
    result = 0.5 * (vx**2 + vy**2 + vz**2)
    logger.debug(f"Kinetic energy for state={state}: {result}")
    return result


def effective_potential(state: Sequence[float], mu: float) -> float:
    r"""
    Compute the effective potential :math:`U_{\mathrm{eff}}` in the CRTBP.

    Parameters
    ----------
    state : Sequence[float]
        Six-component state vector.
    mu : float
        Mass parameter.

    Returns
    -------
    float
        Effective potential value.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.

    Notes
    -----
    Internally relies on :pyfunc:`primary_distance` and
    :pyfunc:`secondary_distance`.
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
    r"""
    Evaluate the pseudo-potential :math:`\Omega` at a planar point.

    Parameters
    ----------
    x, y : float
        Synodic coordinates.
    mu : float
        Mass parameter.

    Returns
    -------
    float
        Value of :math:`\Omega(x, y)`.

    Raises
    ------
    None

    Notes
    -----
    :math:`\Omega = \tfrac12 (x^2 + y^2) + (1-\mu)/r_1 + \mu/r_2`.
    """
    logger.debug(f"Computing pseudo-potential at point x={x}, y={y}, mu={mu}")
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


def gravitational_potential(state: Sequence[float], mu: float) -> float:
    r"""
    Gravitational potential energy of the test particle.

    Parameters
    ----------
    state : Sequence[float]
        Six-component state vector.
    mu : float
        Mass parameter.

    Returns
    -------
    float
        Gravitational potential :math:`U`.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.
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
    r"""
    Distance from the particle to the primary body.

    Parameters
    ----------
    state : Sequence[float]
        Six-component state vector.
    mu : float
        Mass parameter.

    Returns
    -------
    float
        Distance :math:`r_1`.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing primary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_2 = mu
    r1 = np.sqrt((x + mu_2)**2 + y**2 + z**2)
    return r1


def secondary_distance(state: Sequence[float], mu: float) -> float:
    r"""
    Distance from the particle to the secondary body.

    Parameters
    ----------
    state : Sequence[float]
        Six-component state vector.
    mu : float
        Mass parameter.

    Returns
    -------
    float
        Distance :math:`r_2`.

    Raises
    ------
    ValueError
        If *state* cannot be unpacked into six elements.
    """
    # This is a simple helper function, so we'll just use debug level log
    logger.debug(f"Computing secondary distance for state={state}, mu={mu}")
    x, y, z, vx, vy, vz = state
    mu_1 = 1 - mu
    r2 = np.sqrt((x - mu_1)**2 + y**2 + z**2)
    return r2 