"""Configuration classes for Poincare return map implementations (compile-time structure).

This module provides configuration dataclasses for Poincare return map computations.
These define the algorithm structure and problem definition (WHAT to compute), not
the runtime tuning parameters (HOW WELL to compute).

For runtime tuning parameters, see options.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.configs import _HitenBaseConfig


@dataclass(frozen=True)
class _ReturnMapConfig(_HitenBaseConfig):
    """Base configuration for Poincare return map implementations (compile-time structure).

    This abstract base class defines the minimal compile-time configuration
    parameters for return map implementations. Currently serves as a marker
    base class for return map configs.

    Notes
    -----
    This class serves as the base for all return map configurations.
    Concrete implementations should inherit from this class and
    add their specific compile-time structure parameters.

    All time units are in nondimensional units unless otherwise specified.
    """

    def _validate(self) -> None:
        """Validate the configuration."""
        pass


@dataclass(frozen=True)
class _SeedingConfig(_HitenBaseConfig):
    """Configuration for seeding strategy selection (compile-time structure).

    This dataclass defines WHAT seeding strategy to use, which is a
    structural choice that affects how initial conditions are generated.

    Notes
    -----
    Currently serves as a marker base class. Concrete implementations
    add strategy-specific parameters like `seed_strategy`, `seed_axis`, etc.

    Runtime seeding parameters (n_seeds) are in SeedingOptions, not here.
    """

    def _validate(self) -> None:
        """Validate the configuration."""
        pass
