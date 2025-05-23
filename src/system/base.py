import warnings
from dataclasses import dataclass
from typing import Dict

from .body import Body
from .libration import LibrationPoint, L1Point, L2Point, L3Point, L4Point, L5Point
from utils.precision import high_precision_division, log_precision_info
from utils.log_config import logger


@dataclass
class systemConfig:
    primary: Body
    secondary: Body
    distance: float

    def __post_init__(self):
        # Validate that distance is positive.
        if self.distance <= 0:
            raise ValueError("Distance must be a positive value.")


class System(object):
    def __init__(self, config: systemConfig):
        """Initializes the CR3BP system based on the provided configuration."""

        logger.info(f"Initializing System with primary='{config.primary.name}', secondary='{config.secondary.name}', distance={config.distance:.4e}")
        
        # Log precision settings
        log_precision_info()

        self.primary = config.primary
        self.secondary = config.secondary
        self.distance = config.distance

        self.mu: float = self._get_mu()
        logger.info(f"Calculated mass parameter mu = {self.mu:.6e}")

        self.libration_points: Dict[int, LibrationPoint] = self._compute_libration_points(self.mu)
        logger.info(f"Computed {len(self.libration_points)} Libration points.")

    def __str__(self) -> str:
        return f"System(primary='{self.primary.name}', secondary='{self.secondary.name}', mu={self.mu:.4e})"

    def __repr__(self) -> str:
        return f"System(config=systemConfig(primary={self.primary!r}, secondary={self.secondary!r}, distance={self.distance}))"

    def _get_mu(self) -> float:
        """Calculates the mass parameter mu with high precision if enabled."""
        logger.debug(f"Calculating mu: {self.secondary.mass} / ({self.primary.mass} + {self.secondary.mass})")
        
        # Use high precision division for critical mu calculation
        total_mass = self.primary.mass + self.secondary.mass
        mu = high_precision_division(self.secondary.mass, total_mass)
        
        logger.debug(f"Calculated mu with high precision: {mu}")
        return mu

    def _compute_libration_points(self, mu: float) -> Dict[int, LibrationPoint]:
        """Computes all five Libration points for the given mass parameter."""
        logger.debug(f"Computing Libration points for mu={mu}")
        points = {
            1: L1Point(mu),
            2: L2Point(mu),
            3: L3Point(mu),
            4: L4Point(mu),
            5: L5Point(mu)
        }
        logger.debug(f"Finished computing Libration points.")
        return points

    def get_libration_point(self, index: int) -> LibrationPoint:
        """Returns the requested Libration point object."""
        if index not in self.libration_points:
            logger.error(f"Invalid Libration point index requested: {index}. Must be 1-5.")
            raise ValueError(f"Invalid Libration point index: {index}. Must be 1, 2, 3, 4, or 5.")
        logger.debug(f"Retrieving Libration point L{index}")
        return self.libration_points[index]
