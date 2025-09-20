"""Abstract base class for continuation stepping strategies."""

from abc import ABC

import numpy as np


class _ContinuationStepBase(ABC):
    """Define the protocol for continuation stepping strategies.

    This protocol specifies the required interface for all stepping strategies
    used in continuation algorithms. Stepping strategies are responsible for
    predicting the next solution representation and potentially adapting the
    step size based on the current solution state.
    """

    def __call__(self, last_solution: object, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict next solution and adapt step size.

        This method performs the core stepping operation: generating a
        prediction for the next solution in the continuation sequence
        and potentially adapting the step size based on local conditions.

        Parameters
        ----------
        last_solution : object
            Current solution object (e.g., periodic orbit, equilibrium)
            from which to predict the next solution.
        step : ndarray
            Current step size(s) for continuation parameters.
            Shape should match the parameter dimension.

        Returns
        -------
        prediction : ndarray
            Numerical representation of the predicted next solution.
            This will be passed to the continuation engine's instantiation
            method to create a domain object for correction.
        adapted_step : ndarray
            Potentially modified step size for the next continuation step.
            Should have the same shape as the input step array.
        """
        ...
