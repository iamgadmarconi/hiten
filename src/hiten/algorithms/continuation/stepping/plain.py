"""Simple stepping strategy using a provided predictor function."""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.stepping.base import _ContinuationStepBase


class _ContinuationPlainStep(_ContinuationStepBase):
    """Implement a simple stepping strategy using a provided predictor function.

    The plain step strategy delegates prediction to a user-provided
    function and returns the step size unchanged, making it suitable
    for cases where step adaptation is handled elsewhere or not needed.

    Parameters
    ----------
    predictor : callable
        Function that generates solution predictions. Should take a
        solution object and step array, returning a numerical representation
        of the predicted next solution.
    """

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        self._predictor = predictor

    def __call__(self, last_solution, step):
        """Predict next solution using the provided predictor function.

        Parameters
        ----------
        last_solution : object
            Current solution object for prediction.
        step : ndarray
            Current step size array.

        Returns
        -------
        prediction : ndarray
            Result of the predictor function.
        step : ndarray
            Unchanged step size (no adaptation).

        Notes
        -----
        This implementation simply delegates to the predictor function
        and returns the step size unchanged. No step adaptation or
        error handling is performed.
        """
        return self._predictor(last_solution, step), step