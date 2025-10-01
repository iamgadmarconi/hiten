"""Provide configuration classes for continuation algorithms (compile-time structure).

This module provides configuration classes that define the algorithm structure
for continuation methods. These should be set once when creating a pipeline.

For runtime tuning parameters (target ranges, step sizes, etc.), see options.py.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from hiten.algorithms.types.configs import _HitenBaseConfig
from hiten.algorithms.types.states import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


@dataclass(frozen=True)
class ContinuationConfig(_HitenBaseConfig):
    """Base configuration for continuation algorithms (compile-time structure).

    This dataclass encapsulates compile-time configuration parameters that
    define the algorithm structure. These parameters define WHAT algorithm
    is used and HOW the problem is structured.

    Parameters
    ----------
    stepper : Literal["natural", "secant"], default="natural"
        Stepping strategy for continuation. This is a structural algorithm choice.
        
        - "natural": Simple natural parameter continuation
        - "secant": Secant predictor using tangent vectors

    Notes
    -----
    For runtime tuning parameters like `target`, `step`, `max_members`, etc.,
    use ContinuationOptions instead.

    Examples
    --------
    >>> # Compile-time: Choose stepping algorithm
    >>> config = ContinuationConfig(stepper="secant")
    >>> # Runtime: Set target range and step size
    >>> from hiten.algorithms.continuation.options import ContinuationOptions
    >>> options = ContinuationOptions(
    ...     target=(0.0, 1.0),
    ...     step=0.01
    ... )
    """
    stepper: Literal["natural", "secant"] = "natural"

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.stepper not in ["natural", "secant"]:
            raise ValueError(
                f"Invalid stepper: {self.stepper}. "
                "Must be 'natural' or 'secant'."
            )


@dataclass(frozen=True)
class OrbitContinuationConfig(ContinuationConfig):
    """Configuration for periodic orbit continuation (compile-time structure).

    Extends the base continuation configuration with orbit-specific structural
    parameters that define WHAT problem is being solved.

    Parameters
    ----------
    state : SynodicState or None, default=None
        State component(s) to vary during continuation. This defines the
        problem structure - which state components are the continuation parameters.
        
        - None: Vary all state components
        - SynodicState: Vary a single state component (e.g., SynodicState.Z)
        - Sequence[SynodicState]: Vary multiple components
        
    getter : callable or None, default=None
        Function to extract continuation parameter from periodic orbit.
        Defines how parameters are extracted from the solution.
        Should have signature: ``getter(orbit: PeriodicOrbit) -> float``
        If None, uses default parameter extraction based on `state`.

    Notes
    -----
    For runtime parameters like `target`, `step`, `max_members`, use
    OrbitContinuationOptions instead.

    The `state` parameter is compile-time because it fundamentally defines
    WHAT problem you're solving (which parameters to continue), not HOW WELL
    to solve it.

    Examples
    --------
    >>> # Compile-time: Define problem structure
    >>> from hiten.algorithms.types.states import SynodicState
    >>> config = OrbitContinuationConfig(
    ...     state=SynodicState.Z,  # Continue in z-direction
    ...     stepper="secant"
    ... )
    >>> # Runtime: Set ranges and step sizes
    >>> from hiten.algorithms.continuation.options import OrbitContinuationOptions
    >>> options = OrbitContinuationOptions(
    ...     target=(0.0, 0.5),
    ...     step=0.01,
    ...     max_members=100
    ... )
    """
    state: Optional[SynodicState] = None
    getter: Optional[Callable[[PeriodicOrbit], float]] = None

    def _validate(self) -> None:
        """Validate the configuration."""
        super()._validate()
        # State validation happens at runtime when indices are resolved
        # Getter validation happens when it's called
