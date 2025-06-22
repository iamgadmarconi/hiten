r"""
hiten.system.base
===========

High-level abstractions for the Circular Restricted Three-Body Problem (CR3BP).

This module bundles the physical information of a binary system, computes the
mass parameter :math:`\mu`, instantiates the underlying vector field via
:pyfunc:`hiten.algorithms.dynamics.rtbp.rtbp_dynsys`, and pre-computes the five
classical Lagrange (libration) points.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from hiten.algorithms.dynamics.rtbp import rtbp_dynsys
from hiten.system.body import Body
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import L1Point, L2Point, L3Point
from hiten.system.libration.triangular import L4Point, L5Point
from hiten.utils.log_config import logger
from hiten.utils.precision import hp


@dataclass
class systemConfig:
    r"""
    Configuration container for a CR3BP hiten.system.

    Parameters
    ----------
    primary : Body
        Primary gravitating body.
    secondary : Body
        Secondary gravitating body.
    distance : float
        Characteristic separation between *primary* and *secondary* in
        consistent units.

    Attributes
    ----------
    Same as *Parameters*.

    Raises
    ------
    ValueError
        If :pyattr:`distance` is not strictly positive.

    Notes
    -----
    The class is a :pyclass:`dataclasses.dataclass` and therefore immutable
    once instantiated.
    """
    primary: Body
    secondary: Body
    distance: float

    def __post_init__(self):
        # Validate that distance is positive.
        if self.distance <= 0:
            raise ValueError("Distance must be a positive value.")


class System(object):
    r"""
    Lightweight wrapper around the CR3BP dynamical hiten.system.

    The class stores the physical properties of the primaries, computes the
    dimensionless mass parameter :math:`\mu = m_2 / (m_1 + m_2)`, instantiates
    the CR3BP vector field through :pyfunc:`hiten.algorithms.dynamics.rtbp.rtbp_dynsys`,
    and caches the five Lagrange points.

    Parameters
    ----------
    config : systemConfig
        Fully specified configuration of the hiten.system.

    Attributes
    ----------
    primary : Body
        Primary gravitating body.
    secondary : Body
        Secondary gravitating body.
    distance : float
        Characteristic separation between the bodies.
    mu : float
        Mass parameter :math:`\mu`.
    libration_points : dict[int, LibrationPoint]
        Mapping from integer identifiers {1,…,5} to the corresponding
        libration point objects.
    _dynsys : hiten.algorithms.dynamics.base.DynamicalSystemProtocol
        Underlying vector field instance compatible with the integrators
        defined in :pymod:`hiten.algorithms.integrators`.

    Notes
    -----
    The heavy computations reside in the dynamical system and individual
    libration point classes; this wrapper simply orchestrates them.
    """
    def __init__(self, config: systemConfig):
        """Initializes the CR3BP system based on the provided configuration."""

        logger.info(f"Initializing System with primary='{config.primary.name}', secondary='{config.secondary.name}', distance={config.distance:.4e}")
        
        self.primary = config.primary
        self.secondary = config.secondary
        self.distance = config.distance

        self.mu: float = self._get_mu()
        logger.info(f"Calculated mass parameter mu = {self.mu:.6e}")

        self.libration_points: Dict[int, LibrationPoint] = self._compute_libration_points()
        logger.info(f"Computed {len(self.libration_points)} Libration points.")

        self._dynsys = rtbp_dynsys(self.mu, name=self.primary.name + "_" + self.secondary.name)

    def __str__(self) -> str:
        return f"System(primary='{self.primary.name}', secondary='{self.secondary.name}', mu={self.mu:.4e})"

    def __repr__(self) -> str:
        return f"System(config=systemConfig(primary={self.primary!r}, secondary={self.secondary!r}, distance={self.distance}))"

    def _get_mu(self) -> float:
        r"""
        Compute the dimensionless mass parameter.

        Returns
        -------
        float
            Value of :math:`\mu`.

        Notes
        -----
        The calculation is performed in high precision using
        :pyfunc:`hiten.utils.precision.hp` to mitigate numerical cancellation when
        :math:`m_1 \approx m_2`.
        """
        logger.debug(f"Calculating mu: {self.secondary.mass} / ({self.primary.mass} + {self.secondary.mass})")

        # Use Number for critical mu calculation
        primary_mass_hp = hp(self.primary.mass)
        secondary_mass_hp = hp(self.secondary.mass)
        total_mass_hp = primary_mass_hp + secondary_mass_hp
        mu_hp = secondary_mass_hp / total_mass_hp

        mu = float(mu_hp) # Convert back to float for storage
        logger.debug(f"Calculated mu with high precision: {mu}")
        return mu

    def _compute_libration_points(self) -> Dict[int, LibrationPoint]:
        r"""
        Instantiate the five classical libration points.

        Returns
        -------
        dict[int, LibrationPoint]
            Mapping {1,…,5} to :pyclass:`hiten.system.libration.base.LibrationPoint`
            objects.
        """
        logger.debug(f"Computing Libration points for mu={self.mu}")
        points = {
            1: L1Point(self),
            2: L2Point(self),
            3: L3Point(self),
            4: L4Point(self),
            5: L5Point(self)
        }
        logger.debug(f"Finished computing Libration points.")
        return points

    def get_libration_point(self, index: int) -> LibrationPoint:
        r"""
        Access a pre-computed libration point.

        Parameters
        ----------
        index : int
            Identifier of the desired point in {1, 2, 3, 4, 5}.

        Returns
        -------
        LibrationPoint
            Requested libration point instance.

        Raises
        ------
        ValueError
            If *index* is not in the valid range.

        Examples
        --------
        >>> cfg = systemConfig(primary, secondary, distance)
        >>> sys = System(cfg)
        >>> L1 = sys.get_libration_point(1)
        """
        if index not in self.libration_points:
            logger.error(f"Invalid Libration point index requested: {index}. Must be 1-5.")
            raise ValueError(f"Invalid Libration point index: {index}. Must be 1, 2, 3, 4, or 5.")
        logger.debug(f"Retrieving Libration point L{index}")
        return self.libration_points[index]
