"""Configuration for center manifold Poincare sections in the CR3BP.

This module provides configuration classes for computing Poincare sections
restricted to center manifolds of collinear libration points in the Circular
Restricted Three-Body Problem (CR3BP).
"""
from dataclasses import dataclass
from typing import Literal, Optional

from hiten.algorithms.poincare.core.config import (_ReturnMapConfig,
                                                   _SeedingConfig)
from hiten.algorithms.types.configs import IntegrationConfig
from hiten.algorithms.types.exceptions import EngineError
from hiten.utils.log_config import logger


@dataclass(frozen=True)
class CenterManifoldMapConfig(_ReturnMapConfig, _SeedingConfig):
    """Configuration for center manifold Poincare maps.

    This dataclass combines configuration from multiple base classes to provide
    comprehensive settings for center manifold map computation, including
    integration parameters, seeding strategies, and iteration controls.

    Parameters
    ----------
    seed_strategy : {'single', 'axis_aligned', 'level_sets', 'radial', 'random'}, default='axis_aligned'
        Strategy for generating initial conditions on the center manifold.
        - 'single': Single axis seeding along one coordinate direction
        - 'axis_aligned': Seeding aligned with coordinate axes
        - 'level_sets': Seeding based on level sets of the Hamiltonian
        - 'radial': Radial seeding pattern from the periodic orbit
        - 'random': Random seeding within specified bounds
    seed_axis : {'q2', 'p2', 'q3', 'p3'}, optional
        Coordinate axis for single-axis seeding strategy. Required when
        seed_strategy='single', ignored otherwise.
    section_coord : {'q2', 'p2', 'q3', 'p3'}, default='q3'
        Coordinate defining the Poincare section (set to zero).

    Notes
    -----
    The configuration inherits from multiple base classes:
    - :class:`~hiten.algorithms.poincare.core.config._ReturnMapConfig`: Basic return map settings
    - :class:`~hiten.algorithms.poincare.core.config._IntegrationConfig`: Integration method and parameters
    - :class:`~hiten.algorithms.poincare.core.config._IterationConfig`: Iteration control parameters
    - :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`: Seeding strategy parameters

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    seed_strategy: Literal[
        "single",
        "axis_aligned",
        "level_sets",
        "radial",
        "random",
    ] = "axis_aligned"
    seed_axis: Optional[Literal["q2", "p2", "q3", "p3"]] = None
    section_coord: Literal["q2", "p2", "q3", "p3"] = "q3"
    integration: IntegrationConfig = IntegrationConfig()

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.seed_strategy == "single" and self.seed_axis is None:
            raise EngineError("seed_axis must be specified when seed_strategy is 'single'")
        if self.seed_strategy != "single" and self.seed_axis is not None:
            logger.warning("seed_axis is ignored when seed_strategy is not 'single'")
        # Integration method constraints for CM backend
        method = getattr(self.integration, "method", "adaptive")
        if method == "adaptive":
            raise NotImplementedError("Adaptive integrator is not implemented for center-manifold backend; use 'fixed' (RK) or 'symplectic'.")
