"""Runtime options for Poincare return map computation.

This module re-exports common options from types/ and adds Poincare-specific
options. Import from here for consistency within the poincare module.

For compile-time configuration (problem structure), see config.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.options import _HitenBaseOptions


@dataclass(frozen=True)
class IterationOptions(_HitenBaseOptions):
    """Runtime options for iteration control (Poincare-specific).
    
    These parameters tune HOW MANY iterations to compute and can vary
    between method calls.
    
    Parameters
    ----------
    n_iter : int, default=40
        Number of return map iterations to compute. Runtime parameter that
        controls how much data to generate.

    Notes
    -----
    This is a runtime option because it controls HOW MUCH computation to
    perform, not WHAT algorithm to use.
    """
    _version: float = 1.0

    n_iter: int = 40

    def _validate(self) -> None:
        """Validate the options."""
        if self.n_iter <= 0:
            raise ValueError("n_iter must be positive.")


@dataclass(frozen=True)
class SeedingOptions(_HitenBaseOptions):
    """Runtime options for seeding strategies (Poincare-specific).
    
    These parameters tune HOW MANY seeds to generate and can vary
    between method calls.
    
    Parameters
    ----------
    n_seeds : int, default=20
        Number of initial seeds to generate. Runtime parameter that
        controls coverage/resolution.

    Notes
    -----
    This is a runtime option because it controls HOW MANY seeds to generate,
    not WHAT seeding strategy to use (which is defined in the config).
    """
    _version: float = 1.0

    n_seeds: int = 20

    def _validate(self) -> None:
        """Validate the options."""
        if self.n_seeds <= 0:
            raise ValueError("n_seeds must be positive.")
