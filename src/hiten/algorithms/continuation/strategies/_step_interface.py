"""Protocol and base classes for continuation stepping strategies.

This module defines the interface protocol for continuation stepping strategies
and provides basic implementations. Stepping strategies are responsible for
predicting the next solution in a continuation sequence based on the current
solution and step size.

The stepping strategy pattern allows different continuation algorithms to use
various prediction methods (natural parameter, secant, tangent, etc.) while
maintaining a consistent interface with the continuation engine.

All numerical computations use nondimensional units appropriate for the specific
dynamical system being studied.

See Also
--------
:mod:`hiten.algorithms.continuation.base`
    Base continuation engine that uses these stepping strategies.
:mod:`hiten.algorithms.continuation.strategies._algorithms`
    Algorithm classes that create and use stepping strategies.
:mod:`hiten.algorithms.continuation.strategies._stepping`
    Concrete stepping strategy implementations.
"""

from typing import Callable, Protocol

import numpy as np


class _ContinuationStep(Protocol):
    """Protocol defining the interface for continuation stepping strategies.

    This protocol specifies the required interface for all stepping strategies
    used in continuation algorithms. Stepping strategies are responsible for
    predicting the next solution representation and potentially adapting the
    step size based on the current solution state.

    The protocol follows the strategy pattern, allowing different continuation
    algorithms to use various prediction methods while maintaining interface
    consistency with the continuation engine.

    Methods
    -------
    __call__(last_solution, step)
        Predict next solution representation and return adapted step size.

    Notes
    -----
    Implementations of this protocol should handle:
    
    - **Prediction**: Generate numerical representation of next solution
    - **Step adaptation**: Modify step size based on local solution behavior
    - **Error handling**: Gracefully handle prediction failures
    - **State management**: Maintain any internal state needed for prediction
    
    Common stepping strategies include:
    - Natural parameter: linear extrapolation in parameter space
    - Secant method: linear extrapolation using solution history
    - Tangent method: prediction using computed tangent vectors
    - Predictor-corrector: sophisticated multi-stage prediction

    Examples
    --------
    >>> # Example implementation
    >>> class MySteppingStrategy:
    ...     def __call__(self, last_solution, step):
    ...         # Predict next solution representation
    ...         prediction = self.predict(last_solution, step)
    ...         # Adapt step size if needed
    ...         new_step = self.adapt_step(step, last_solution)
    ...         return prediction, new_step

    See Also
    --------
    :class:`_PlainStep`
        Simple implementation of this protocol.
    :class:`hiten.algorithms.continuation.strategies._stepping._NaturalParameterStep`
        Natural parameter stepping implementation.
    :class:`hiten.algorithms.continuation.strategies._stepping._SecantStep`
        Secant-based stepping implementation.
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

        Notes
        -----
        The prediction should be a reasonable approximation of the next
        solution that can be refined by the continuation engine's corrector.
        The quality of the prediction affects convergence speed and robustness.
        
        Step adaptation allows strategies to implement local step size
        control based on solution behavior, curvature, or other factors.
        """
        ...


class _PlainStep:
    """Simple stepping strategy using a provided predictor function.

    This class implements the :class:`_ContinuationStep` protocol using
    a simple predictor function without step size adaptation. It serves
    as a basic building block for continuation algorithms that don't
    require sophisticated prediction or adaptive stepping.

    The plain step strategy delegates prediction to a user-provided
    function and returns the step size unchanged, making it suitable
    for cases where step adaptation is handled elsewhere or not needed.

    Parameters
    ----------
    predictor : callable
        Function that generates solution predictions. Should take a
        solution object and step array, returning a numerical representation
        of the predicted next solution.

    Notes
    -----
    This implementation is stateless and thread-safe. It's commonly used
    as a wrapper around simple prediction functions in natural parameter
    continuation where step adaptation is handled by the continuation engine.

    The predictor function signature should be:
    ``predictor(solution: object, step: ndarray) -> ndarray``

    Examples
    --------
    >>> # Create simple predictor function
    >>> def predict_state(orbit, step):
    ...     new_state = orbit.initial_state.copy()
    ...     new_state[0] += step[0]  # Increment x-component
    ...     return new_state
    >>> 
    >>> # Wrap in plain step strategy
    >>> stepper = _PlainStep(predict_state)
    >>> prediction, new_step = stepper(current_orbit, np.array([0.01]))

    See Also
    --------
    :class:`_ContinuationStep`
        Protocol that this class implements.
    :class:`hiten.algorithms.continuation.strategies._stepping._NaturalParameterStep`
        More sophisticated natural parameter stepping.
    """

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        """Initialize plain stepping strategy with predictor function.

        Parameters
        ----------
        predictor : callable
            Prediction function with signature:
            ``predictor(solution: object, step: ndarray) -> ndarray``
        """
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