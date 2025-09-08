"""Provide concrete implementations of continuation stepping strategies.

This module provides concrete implementations of stepping strategies used in
continuation algorithms. These strategies handle the prediction phase of the
continuation process, generating numerical representations of the next solution
based on the current solution and step size.

The module includes both simple natural parameter stepping and
pseudo-arclength stepping with tangent vector maintenance for navigating
complex solution branches.

All numerical computations use nondimensional units appropriate for the specific
dynamical system being studied.

See Also
--------
:mod:`~hiten.algorithms.continuation.strategies._step_interface`
    Protocol definitions for stepping strategies.
:mod:`~hiten.algorithms.continuation.strategies._algorithms`
    Algorithm classes that use these stepping strategies.
:mod:`~hiten.algorithms.continuation.base`
    Base continuation engine that coordinates with stepping strategies.
"""

from typing import Callable, Protocol

import numpy as np


class _StepStrategy(Protocol):
    """Define an extended protocol for stepping strategies with event hooks.

    This protocol extends the basic stepping interface with event hooks
    that allow strategies to respond to various continuation events.
    This enables sophisticated stepping strategies that maintain internal
    state and adapt their behavior based on continuation progress.

    The protocol supports strategies that need to:
    - Track solution history for improved predictions
    - Adapt step sizes based on convergence behavior
    - Maintain auxiliary data structures (e.g., tangent vectors)
    - Respond to continuation events for internal bookkeeping

    Methods
    -------
    __call__(last_solution, step)
        Core stepping method for solution prediction.
    on_initialisation(*args, **kwargs)
        Hook called when continuation engine initializes.
    on_iteration(*args, **kwargs)
        Hook called at the start of each continuation iteration.
    on_success(*args, **kwargs)
        Hook called after successful solution acceptance.
    on_reject(*args, **kwargs)
        Hook called after solution rejection.
    on_failure(*args, **kwargs)
        Hook called after correction failure.

    Notes
    -----
    All hook methods are optional and can be implemented as no-ops.
    The hooks provide extension points for strategies that need to
    maintain state or adapt behavior based on continuation events.

    Common uses for hooks include:
    - Updating solution history for secant methods
    - Computing tangent vectors for pseudo-arclength continuation
    - Adapting step sizes based on convergence rates
    - Logging or debugging continuation progress

    See Also
    --------
    :class:`~hiten.algorithms.continuation.strategies._stepping._NaturalParameterStep`
        Simple implementation with no-op hooks.
    :class:`~hiten.algorithms.continuation.strategies._stepping._SecantStep`
        Sophisticated implementation using success hooks.
    """

    def __call__(
        self,
        last_solution: object,
        step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict next solution and return adapted step size.

        Parameters
        ----------
        last_solution : object
            Current solution object for prediction.
        step : ndarray
            Current step size array.

        Returns
        -------
        prediction : ndarray
            Numerical representation of predicted next solution.
        adapted_step : ndarray
            Potentially modified step size for next iteration.
        """
        ...

    def on_iteration(self, *args, **kwargs) -> None:
        """Hook called at the start of each continuation iteration.
        
        Parameters
        ----------
        *args, **kwargs
            Event-specific arguments (implementation dependent).
        """
        ...

    def on_reject(self, *args, **kwargs) -> None:
        """Hook called after solution rejection during continuation.
        
        Parameters
        ----------
        *args, **kwargs
            Event-specific arguments (implementation dependent).
        """
        ...

    def on_failure(self, *args, **kwargs) -> None:
        """Hook called after correction failure during continuation.
        
        Parameters
        ----------
        *args, **kwargs
            Event-specific arguments (implementation dependent).
        """
        ...

    def on_success(self, *args, **kwargs) -> None:
        """Hook called after successful solution acceptance.
        
        Parameters
        ----------
        *args, **kwargs
            Event-specific arguments (implementation dependent).
        """
        ...

    def on_initialisation(self, *args, **kwargs) -> None:
        """Hook called when continuation engine initializes.
        
        Parameters
        ----------
        *args, **kwargs
            Event-specific arguments (implementation dependent).
        """
        ...


class _NaturalParameterStep:
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
    :class:`~hiten.algorithms.continuation.strategies._stepping._SecantStep`
        More sophisticated stepping with tangent vector maintenance.
    :class:`~hiten.algorithms.continuation.strategies._stepping._StepStrategy`
        Protocol that this class implements.
    """

    def __init__(self, predictor: Callable[[object, np.ndarray], np.ndarray]):
        """Initialize natural parameter stepping with predictor function.

        Parameters
        ----------
        predictor : callable
            Prediction function with signature:
            ``predictor(solution: object, step: ndarray) -> ndarray``
        """
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
        return self._predictor(last_solution, step), step

    # Event hooks - all implemented as no-ops for simplicity
    def on_success(self, *_, **__):
        """No-op hook for successful solution acceptance.
        
        This implementation does nothing since natural parameter
        stepping doesn't require state updates on success.
        """
        pass

    def on_iteration(self, *_, **__):
        """No-op hook for iteration start.
        
        This implementation does nothing since natural parameter
        stepping doesn't require per-iteration processing.
        """
        pass

    def on_reject(self, *_, **__):
        """No-op hook for solution rejection.
        
        This implementation does nothing since natural parameter
        stepping doesn't require rejection handling.
        """
        pass

    def on_failure(self, *_, **__):
        """No-op hook for correction failure.
        
        This implementation does nothing since natural parameter
        stepping doesn't require failure handling.
        """
        pass

    def on_initialisation(self, *_, **__):
        """No-op hook for continuation initialization.
        
        This implementation does nothing since natural parameter
        stepping doesn't require initialization processing.
        """
        pass


class _SecantStep:
    """Implement a secant-based stepping strategy for pseudo-arclength continuation.

    This class implements a sophisticated stepping strategy that maintains
    solution history and computes tangent vectors for pseudo-arclength
    continuation. It can navigate around turning points and bifurcations
    that would cause simpler stepping strategies to fail.

    The strategy uses the secant method to estimate tangent vectors from
    solution history, enabling prediction along solution curves in the
    extended parameter-solution space characteristic of pseudo-arclength
    continuation.

    Parameters
    ----------
    representation_fn : callable
        Function that converts solution objects to numerical representations.
        Should have signature: ``representation_fn(solution: object) -> ndarray``
        Used for computing distances and tangent vectors.
    parameter_fn : callable
        Function that extracts continuation parameters from solution objects.
        Should have signature: ``parameter_fn(solution: object) -> ndarray``
        Used for parameter space components of tangent vectors.

    Attributes
    ----------
    _repr_hist : list of ndarray
        History of solution representations for tangent computation.
    _param_hist : list of ndarray
        History of parameter values for tangent computation.
    _tangent : ndarray or None
        Current normalized tangent vector in extended space, or None
        if insufficient history for tangent computation.

    Notes
    -----
    The stepping strategy implements:
    
    1. **History maintenance**: Tracks solution and parameter history
    2. **Tangent computation**: Uses secant method for tangent estimation
    3. **Fallback prediction**: Natural parameter step when tangent unavailable
    4. **Extended space**: Operates in combined solution-parameter space
    
    The tangent vector is computed in the extended space containing both
    solution representation and continuation parameters, enabling the
    pseudo-arclength method to navigate complex solution branches.

    Examples
    --------
    >>> # Define representation and parameter functions
    >>> def get_representation(orbit):
    ...     return orbit.initial_state
    >>> 
    >>> def get_parameter(orbit):
    ...     return np.array([orbit.energy])
    >>> 
    >>> # Create secant stepping strategy
    >>> stepper = _SecantStep(get_representation, get_parameter)
    >>> 
    >>> # Use in pseudo-arclength continuation
    >>> prediction, new_step = stepper(current_orbit, np.array([0.01]))

    See Also
    --------
    :class:`~hiten.algorithms.continuation.strategies._stepping._NaturalParameterStep`
        Simpler stepping strategy for natural parameter continuation.
    :class:`~hiten.algorithms.continuation.strategies._algorithms._SecantArcLength`
        Algorithm class that uses this stepping strategy.
    """

    def __init__(
        self,
        representation_fn: Callable[[object], np.ndarray],
        parameter_fn: Callable[[object], np.ndarray],
    ) -> None:
        """Initialize secant stepping with representation and parameter functions.

        Parameters
        ----------
        representation_fn : callable
            Function to convert solutions to numerical representations.
        parameter_fn : callable
            Function to extract parameters from solutions.
        """
        self._repr_fn = representation_fn
        self._param_fn = parameter_fn

        # History buffers (updated in on_success)
        self._repr_hist: list[np.ndarray] = []
        self._param_hist: list[np.ndarray] = []

        self._tangent: np.ndarray | None = None

    def __call__(self, last_solution: object, step) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction using secant method with tangent vector.

        This method implements the core prediction logic for pseudo-arclength
        continuation, using either tangent-based prediction when available
        or falling back to simple natural parameter stepping.

        Parameters
        ----------
        last_solution : object
            Current solution object for prediction.
        step : ndarray or float
            Current step size (scalar or array).

        Returns
        -------
        prediction : ndarray
            Numerical representation of predicted next solution.
        step : ndarray or float
            Unchanged step size (same type as input).

        Notes
        -----
        The prediction algorithm:
        
        1. Ensures current solution is in history (for initialization)
        2. Uses tangent-based prediction if tangent vector is available
        3. Falls back to natural parameter step if no tangent available
        4. Handles both scalar and vector step size inputs consistently
        
        Tangent-based prediction moves along the normalized tangent
        direction in the extended solution-parameter space by the
        specified step size magnitude.
        """
        # Ensure history contains last solution
        if not self._repr_hist:
            self._repr_hist.append(self._repr_fn(last_solution))
            self._param_hist.append(self._param_fn(last_solution))

        # If we have a valid tangent use it, otherwise perform small natural step
        if self._tangent is None:
            # Fallback: small natural-parameter step of magnitude |step|.
            # Ensure both scalar and vector ``step`` inputs are handled consistently.
            ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))
            n = self._repr_hist[-1].copy()
            n[0] += ds_scalar  # naive perturb of the first component by |step|
            return n, step

        ds_scalar = float(step) if np.ndim(step) == 0 else float(np.linalg.norm(step))
        n_repr = self._repr_hist[-1].size
        dr = self._tangent[:n_repr] * ds_scalar
        new_repr = self._repr_hist[-1] + dr
        return new_repr, step

    def _update_history_and_tangent(self, accepted_solution: object):
        """Update solution history and recompute secant tangent vector.

        This method maintains the solution history buffers and computes
        the normalized tangent vector using the secant method. The tangent
        vector is computed in the extended space containing both solution
        representation and continuation parameters.

        Parameters
        ----------
        accepted_solution : object
            Newly accepted solution to add to history.

        Notes
        -----
        The tangent vector computation:
        
        1. Extracts representation and parameters from new solution
        2. Appends to history buffers
        3. Computes differences from previous solution (secant method)
        4. Concatenates representation and parameter differences
        5. Normalizes to unit length (or sets to None if zero)
        
        The tangent vector enables prediction along the solution curve
        in the extended parameter-solution space, which is essential
        for pseudo-arclength continuation.
        """
        r = self._repr_fn(accepted_solution)
        p = self._param_fn(accepted_solution)

        self._repr_hist.append(r)
        self._param_hist.append(p)

        if len(self._repr_hist) < 2:
            self._tangent = None
            return

        dr = self._repr_hist[-1] - self._repr_hist[-2]
        dp = self._param_hist[-1] - self._param_hist[-2]
        vec = np.concatenate((dr.ravel(), dp.ravel()))
        norm = np.linalg.norm(vec)
        self._tangent = None if norm == 0 else vec / norm

    def on_success(self, accepted_solution: object):
        """Handle successful solution acceptance by updating history and tangent.

        This hook is called by the continuation engine after a solution
        is successfully corrected and accepted into the family. It updates
        the internal history buffers and recomputes the tangent vector
        for future predictions.

        Parameters
        ----------
        accepted_solution : object
            The solution object that was just accepted.

        Notes
        -----
        This is the key hook for maintaining the internal state needed
        for secant-based predictions. It ensures that the tangent vector
        stays current with the latest solution information.
        """
        self._update_history_and_tangent(accepted_solution)

    def on_iteration(self, *_, **__):
        """No-op hook for iteration start.
        
        This implementation does nothing since secant stepping
        doesn't require per-iteration processing beyond success handling.
        """
        pass

    def on_reject(self, *_, **__):
        """No-op hook for solution rejection.
        
        This implementation does nothing since secant stepping
        doesn't require special rejection handling. The history
        remains unchanged when solutions are rejected.
        """
        pass

    def on_failure(self, *_, **__):
        """No-op hook for correction failure.
        
        This implementation does nothing since secant stepping
        doesn't require special failure handling. The history
        remains unchanged when corrections fail.
        """
        pass

    def on_initialisation(self, first_solution: object):
        """Initialize history buffers with the seed solution.

        This hook is called by the continuation engine during initialization
        to prime the history buffers with the initial solution. This ensures
        that the stepping strategy has the necessary starting data.

        Parameters
        ----------
        first_solution : object
            The initial solution object for the continuation.

        Notes
        -----
        After initialization, the history contains one entry and the
        tangent vector is None (requiring at least two points for
        tangent computation).
        """
        # Prime history with seed when engine notifies initialisation
        self._repr_hist.append(self._repr_fn(first_solution))
        self._param_hist.append(self._param_fn(first_solution))