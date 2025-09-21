"""
Protocols for the continuation module.

These runtime-checkable protocols formalize the interfaces for
stepping strategies and engines in the continuation architecture.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from hiten.algorithms.continuation.types import ContinuationResult


@runtime_checkable
class ContinuationStepProtocol(Protocol):
    """Protocol for continuation step strategies with optional hooks.

    Implementations generate the next prediction from the last accepted
    solution and the current step size, and may adapt internal state via hooks.
    """

    def __call__(self, last_solution: object, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Generate the next prediction from the last accepted solution and the current step size.

        Parameters
        ----------
        last_solution : object
            The last accepted solution.
        step : np.ndarray
            The current step size.
        
        Returns
        -------
        prediction : np.ndarray
            The next prediction.
        step : np.ndarray
            The current step size.
        """
        ...


@runtime_checkable
class ContinuationEngineProtocol(Protocol):
    """Protocol for continuation engines.

    Engines drive the predict-instantiate-correct-accept loop and should
    return a standardized result object upon completion.
    """

    def solve(self) -> ContinuationResult:
        """Solve the continuation problem."""
        ...


