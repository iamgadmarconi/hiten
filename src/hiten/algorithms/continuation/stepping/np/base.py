"""Abstract base class for natural parameter stepping strategy.

This module provides the base class for the natural parameter stepping strategy for the continuation package.
"""

from typing import Callable

import numpy as np

from hiten.algorithms.continuation.stepping.base import _ContinuationStepBase


class _NaturalParameterStep(_ContinuationStepBase):
    """Implement a natural parameter stepping strategy with user-supplied predictor.

    This class implements a simple stepping strategy for natural parameter
    continuation. It delegates prediction to a user-supplied function and
    keeps the step size unchanged, making it suitable for straightforward
    continuation scenarios without complex step adaptation requirements.

    All domain-specific logic (state component selection, amplitude
    manipulations, parameter scaling, etc.) is encapsulated in the
    predictor function, keeping the stepping strategy generic and reusable.

    Parameters
    ----------
    predictor : callable
        Function that generates solution predictions. Should have signature:
        ``predictor(solution: object, step: ndarray) -> ndarray``
        Returns numerical representation of the predicted next solution.

    Notes
    -----
    This implementation provides:
    
    - **Simple delegation**: All prediction logic in user function
    - **No step adaptation**: Step sizes remain unchanged
    - **Stateless operation**: No internal state maintenance
    - **No-op hooks**: All event hooks are empty implementations
    
    The strategy is most suitable for:
    - Natural parameter continuation with simple prediction
    - Cases where step adaptation is handled by the continuation engine
    - Domain-specific prediction logic that doesn't need strategy state
    - Rapid prototyping and simple continuation scenarios

    Examples
    --------
    >>> # Define prediction function
    >>> def predict_orbit_state(orbit, step):
    ...     new_state = orbit.initial_state.copy()
    ...     new_state[2] += step[0]  # Increment z-component
    ...     return new_state
    >>> 
    >>> # Create stepping strategy
    >>> stepper = _NaturalParameterStep(predict_orbit_state)
    >>> 
    >>> # Use in continuation algorithm
    >>> prediction, new_step = stepper(current_orbit, np.array([0.01]))

    See Also
    --------
    :class:`~hiten.algorithms.continuation.stepping.sc._SecantStep`
        More sophisticated stepping with tangent vector maintenance.
    :class:`~hiten.algorithms.continuation.stepping.base._ContinuationStepBase`
        Base class that this class implements.
    """

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        self._predictor = predictor

    def __call__(self, last_solution: object, step: np.ndarray):
        """Generate prediction using the supplied predictor function.

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
        This method simply delegates to the predictor function and
        returns the step size unchanged. No internal state is modified.
        """
        prediction = self._predictor(last_solution, step)
        return prediction, step
