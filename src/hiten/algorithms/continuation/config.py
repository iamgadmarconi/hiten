"""Provide configuration classes for continuation algorithms (compile-time structure).

This module provides configuration classes that define the algorithm structure
for continuation methods. These should be set once when creating a pipeline.

For runtime tuning parameters (target ranges, step sizes, etc.), see options.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence, Tuple

import numpy as np

from hiten.algorithms.types.configs import _HitenBaseConfig
from hiten.algorithms.types.states import SynodicState
from hiten.system.orbits.base import PeriodicOrbit


def _make_orbit_parameter_getter(
    indices: Optional[Tuple[int, ...]]
) -> Callable[[np.ndarray], np.ndarray]:
    if indices is None:
        return lambda vec: np.asarray(vec, dtype=float)
    idx_array = np.asarray(indices, dtype=int)
    return lambda vec: np.asarray(vec, dtype=float)[idx_array]


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
    state: Optional[Sequence[SynodicState] | SynodicState | int] = None
    getter: Optional[Callable[[PeriodicOrbit], float]] = None
    _state_indices: Optional[Tuple[int, ...]] = field(init=False, default=None, repr=False)

    def _validate(self) -> None:
        """Validate the configuration."""
        super()._validate()
        # Getter validation happens when it's called

    def __post_init__(self) -> None:
        super().__post_init__()
        indices = self._coerce_state_indices(self.state)
        object.__setattr__(self, "_state_indices", indices)

    @staticmethod
    def _coerce_state_indices(state: Optional[Sequence[SynodicState] | SynodicState | int]) -> Optional[Tuple[int, ...]]:
        if state is None:
            return None
        if isinstance(state, SynodicState):
            return (int(state.value),)
        if isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
            indices: list[int] = []
            for item in state:
                if isinstance(item, SynodicState):
                    indices.append(int(item.value))
                else:
                    indices.append(int(item))
            return tuple(indices)
        return (int(state),)

    @property
    def state_indices(self) -> Optional[Tuple[int, ...]]:
        """Return resolved continuation state indices (compile-time)."""
        return self._state_indices

    def make_parameter_getter(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create a vector-space parameter getter respecting state selection."""
        return _make_orbit_parameter_getter(self._state_indices)

    def make_representation_of(self) -> Callable[[object], np.ndarray]:
        """Create a representation helper from domain objects to vectors."""
        def _representation(obj: object) -> np.ndarray:
            if hasattr(obj, "initial_state"):
                return np.asarray(getattr(obj, "initial_state"), dtype=float)
            return np.asarray(obj, dtype=float)

        return _representation
