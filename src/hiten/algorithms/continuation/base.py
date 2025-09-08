"""Provide base classes for numerical continuation algorithms in dynamical systems.

This module provides the foundational abstract base class for numerical continuation
algorithms used to trace families of solutions in dynamical systems. The continuation
engine implements a generic predict-correct framework that can be specialized for
different types of problems (periodic orbits, equilibria, manifolds, etc.).

See Also
--------
:mod:`~hiten.algorithms.continuation.strategies`
    Stepping strategies for different continuation methods.
:mod:`~hiten.algorithms.corrector`
    Corrector algorithms for solution refinement.
:mod:`~hiten.system`
    Dynamical system definitions that use continuation.
"""

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from hiten.algorithms.continuation.strategies._step_interface import \
    _ContinuationStep
from hiten.utils.log_config import logger


class _ContinuationEngine(ABC):
    """Provide an abstract base class for numerical continuation algorithms.

    This class provides a generic framework for tracing families of solutions
    in dynamical systems using numerical continuation methods. It implements
    the standard predict-correct algorithm with adaptive step size control
    and flexible stopping criteria.

    The engine is designed to be subclassed with problem-specific implementations
    of key methods like correction, parameter extraction, and stopping conditions.
    It uses a strategy pattern for prediction steps to support different
    continuation methods (natural parameter, pseudo-arclength, etc.).

    Parameters
    ----------
    initial_solution : object
        Starting solution for the continuation (e.g., periodic orbit, equilibrium).
        The type depends on the specific problem domain.
    parameter_getter : callable
        Function that extracts continuation parameter(s) from a solution object.
        Should return float or ndarray.
    target : sequence
        Target parameter range(s) for continuation. For 1D: (min, max).
        For multi-dimensional: (2, m) array where each column specifies
        (min, max) for one parameter.
    step : float or sequence of float, default 1e-4
        Initial step size(s) for continuation parameters. If scalar,
        uses same step for all parameters.
    corrector_kwargs : dict, optional
        Additional keyword arguments passed to the corrector method.
    max_iters : int, default 256
        Maximum number of continuation steps before termination.

    Attributes
    ----------
    family : sequence of object
        Read-only view of generated solution family.
    parameter_values : sequence of ndarray
        Parameter values corresponding to each family member.

    Notes
    -----
    The continuation algorithm implements these steps:
    
    1. **Prediction**: Use stepping strategy to predict next solution
    2. **Instantiation**: Convert predicted representation to domain object
    3. **Correction**: Refine solution using problem-specific corrector
    4. **Acceptance**: Add to family if correction succeeds
    5. **Adaptation**: Adjust step size based on success/failure
    6. **Termination**: Check stopping condition and iteration limit
    
    Subclasses must implement:
    - :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._make_stepper`: Create stepping strategy
    - :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._stop_condition`: Define termination criteria
    - :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._instantiate`: Convert predictions to domain objects
    - :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._correct`: Problem-specific correction
    - :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._parameter`: Extract parameters from solutions

    Examples
    --------
    >>> # Subclass implementation example
    >>> class OrbitContinuation(_ContinuationEngine):
    ...     def _make_stepper(self):
    ...         return NaturalParameterStep()
    ...     
    ...     def _stop_condition(self):
    ...         current = self._parameter(self._family[-1])
    ...         return np.any(current >= self._target_max)
    ...     
    ...     def _instantiate(self, repr):
    ...         return Orbit.from_state(repr)
    ...     
    ...     def _correct(self, orbit, **kwargs):
    ...         return orbit.correct(**kwargs)
    ...     
    ...     def _parameter(self, orbit):
    ...         return orbit.energy

    See Also
    --------
    :class:`~hiten.algorithms.continuation.strategies._step_interface._ContinuationStep`
        Base class for stepping strategies.
    :mod:`~hiten.algorithms.corrector`
        Corrector algorithms for solution refinement.
    """

    def __init__(self, *,  initial_solution: object, parameter_getter: Callable[[object], "np.ndarray | float"],
                target: Sequence[Sequence[float] | float], step: float | Sequence[float] = 1e-4,
                corrector_kwargs: dict | None = None, max_iters: int = 256) -> None:

        self._getter = parameter_getter
        target_arr = np.asarray(target, dtype=float)
        if target_arr.ndim == 1:
            if target_arr.size != 2:
                raise ValueError("target must be (min,max) for 1-D or (2,m) for multi-D continuation")
            target_arr = target_arr.reshape(2, 1)
        elif not (target_arr.ndim == 2 and target_arr.shape[0] == 2):
            raise ValueError("target must be iterable shaped (2,) or (2,m)")

        current_param = np.asarray(self._getter(initial_solution), dtype=float)
        if current_param.ndim == 0:
            current_param = current_param.reshape(1)

        step_arr = np.asarray(step, dtype=float)
        if step_arr.size == 1:
            step_arr = np.full_like(current_param, float(step_arr))
        elif step_arr.size != current_param.size:
            raise ValueError("step length does not match number of continuation parameters")

        if target_arr.shape[1] != current_param.size:
            if target_arr.shape[1] == 1:
                target_arr = np.repeat(target_arr, current_param.size, axis=1)
            else:
                raise ValueError("target dimensionality mismatch with continuation parameter")

        self._target_min = np.minimum(target_arr[0], target_arr[1])
        self._target_max = np.maximum(target_arr[0], target_arr[1])

        self._step = step_arr.astype(float)

        self._family: list[object] = [initial_solution]
        self._param_history: list[np.ndarray] = [current_param.copy()]

        self._corrector_kwargs = corrector_kwargs or {}
        self._max_iters = int(max_iters)

        # Build stepper strategy (must be provided by subclass or mix-in)
        self._stepper: _ContinuationStep = self._make_stepper()
        # Notify strategy initialisation hook if present
        if hasattr(self._stepper, "on_initialisation"):
            try:
                self._stepper.on_initialisation(initial_solution)
            except Exception as exc:
                logger.debug("stepper on_initialisation hook error: %s", exc)

        logger.info(
            "Continuation initialised: parameter=%s, target=[%s - %s], step=%s, max_iters=%d",
            current_param,
            self._target_min,
            self._target_max,
            self._step,
            self._max_iters,
        )

    @property
    def family(self) -> Sequence[object]:  
        """Return a read-only view of the generated solution family.

        Returns
        -------
        sequence of object
            Tuple containing all solutions in the continuation family.
            The initial solution is at index 0, with subsequent solutions
            ordered by continuation step.

        Notes
        -----
        This property provides immutable access to the solution family.
        Solutions are added during the :meth:`~hiten.algorithms.continuation.base._ContinuationEngine.run`
        method as continuation progresses and corrections succeed.
        """
        return tuple(self._family)

    @property
    def parameter_values(self) -> Sequence[np.ndarray]:
        """Return parameter values corresponding to each family member.

        Returns
        -------
        sequence of ndarray
            Tuple containing parameter values for each solution in the family.
            Each array has shape matching the continuation parameter dimension.

        Notes
        -----
        Parameter values are extracted using the parameter_getter function
        provided during initialization. The values correspond one-to-one
        with solutions in the :attr:`~hiten.algorithms.continuation.base._ContinuationEngine.family` 
        property.
        """
        return tuple(self._param_history)

    def run(self) -> list[object]:
        """Execute the continuation algorithm to generate solution family.

        This method implements the main continuation loop, repeatedly applying
        the predict-correct algorithm until a stopping condition is met or
        the maximum iteration limit is reached.

        Returns
        -------
        list of object
            Complete solution family including the initial solution.
            Solutions are ordered by continuation step.

        Notes
        -----
        The algorithm performs these steps in each iteration:
        
        1. Check stopping condition and iteration limit
        2. Use stepping strategy to predict next solution
        3. Instantiate domain object from prediction
        4. Apply corrector to refine the solution
        5. Accept solution if correction succeeds
        6. Adapt step size based on success/failure
        7. Call optional hooks for custom processing
        
        Failed corrections trigger step size reduction and retry up to
        10 attempts before aborting. Successful steps may increase step size
        for efficiency.

        Examples
        --------
        >>> engine = OrbitContinuation(
        ...     initial_solution=orbit0,
        ...     parameter_getter=lambda o: o.energy,
        ...     target=(3.0, 4.0),
        ...     step=0.01
        ... )
        >>> family = engine.run()
        >>> print(f"Generated {len(family)} orbits")
        """
        logger.info("Starting continuation loop ...")
        attempts_at_current_step = 0

        while not self._stop_condition():
            if len(self._family) >= self._max_iters:
                logger.warning("Reached max_iters=%d, terminating continuation.", self._max_iters)
                break

            last_sol = self._family[-1]

            predicted_repr, next_step = self._stepper(last_sol, self._step)
            self._step = next_step.copy()

            candidate = self._instantiate(predicted_repr)

            try:
                candidate = self._correct(candidate, **self._corrector_kwargs)
            except Exception as exc:
                logger.debug(
                    "Correction failed at step %s (attempt %d): %s",
                    self._step,
                    attempts_at_current_step + 1,
                    exc,
                    exc_info=exc,
                )
                # Notify strategy of failure via _update_step fallback for now
                self._step = self._update_step(self._step, success=False)
                attempts_at_current_step += 1
                if attempts_at_current_step > 10:
                    logger.error("Too many failed attempts at current step; aborting continuation.")
                    break
                continue  # retry with reduced step

            attempts_at_current_step = 0  # reset counter on success
            self._family.append(candidate)

            param_val = self._parameter(candidate)
            self._param_history.append(np.asarray(param_val, dtype=float).copy())

            logger.info("Accepted member #%d, parameter=%s", len(self._family) - 1, param_val)

            # Call optional hook for subclasses/callbacks after successful acceptance
            try:
                self._on_accept(candidate)
            except Exception as exc:
                logger.warning("_on_accept hook raised exception: %s", exc)

            self._step = self._update_step(self._step, success=True)

            if hasattr(self._stepper, "on_success"):
                try:
                    self._stepper.on_success(candidate)
                except Exception as exc:
                    logger.debug("stepper on_success hook error: %s", exc)

        logger.info("Continuation finished : generated %d members.", len(self._family))
        return self._family

    def _instantiate(self, representation: np.ndarray):
        """Instantiate a domain object from the predicted representation.

        This method converts the numerical representation produced by the
        stepping strategy into a domain-specific object that can be corrected.

        Parameters
        ----------
        representation : ndarray
            Numerical representation of the predicted solution, typically
            state vectors or other parametric data.

        Returns
        -------
        object
            Domain-specific object (e.g., Orbit, Equilibrium) ready for correction.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses
        or domain-specific mix-ins. The representation format depends on
        the stepping strategy being used.
        """
        raise NotImplementedError("_instantiate must be provided by a domain mix-in")

    def _correct(self, obj: object, **kwargs):  
        """Apply problem-specific corrector to refine the predicted solution.

        This method takes a predicted domain object and applies appropriate
        correction algorithms to satisfy the problem constraints (e.g.,
        periodicity, equilibrium conditions).

        Parameters
        ----------
        obj : object
            Domain object to be corrected (from 
            :meth:`~hiten.algorithms.continuation.base._ContinuationEngine._instantiate`).
        **kwargs
            Additional correction parameters passed from corrector_kwargs.

        Returns
        -------
        object
            Corrected domain object satisfying problem constraints.

        Raises
        ------
        Exception
            If correction fails to converge or constraints cannot be satisfied.
            The engine will catch these exceptions and reduce step size.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Common correctors include Newton-Raphson for periodic orbits and
        fixed-point iteration for equilibria.
        """
        raise NotImplementedError("_correct must be implemented by a domain mix-in")

    def _parameter(self, obj: object) -> np.ndarray:  
        """Extract continuation parameter value from a domain object.

        This method extracts the current value of the continuation parameter(s)
        from a corrected solution object.

        Parameters
        ----------
        obj : object
            Domain object (e.g., corrected orbit or equilibrium).

        Returns
        -------
        ndarray
            Current parameter value(s) with shape matching the continuation
            parameter dimension.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        The parameter extraction should be consistent with the parameter_getter
        function provided during initialization.
        """
        raise NotImplementedError("_parameter must be implemented by a domain mix-in")

    def _update_step(self, current_step: np.ndarray, *, success: bool) -> np.ndarray:  
        """Adapt step size based on correction success or failure.

        This method implements the default step size adaptation strategy,
        increasing step size after successful corrections and decreasing
        it after failures to maintain algorithm efficiency and robustness.

        Parameters
        ----------
        current_step : ndarray
            Current step size(s) for continuation parameters.
        success : bool
            True if the last correction succeeded, False if it failed.

        Returns
        -------
        ndarray
            Adapted step size(s) with same shape as current_step.

        Notes
        -----
        The default strategy uses multiplicative adaptation:
        - Success: multiply by 2.0 (increase efficiency)
        - Failure: multiply by 0.5 (improve robustness)
        
        Step magnitudes are clamped to [1e-10, 1.0] to prevent numerical
        issues while preserving step direction. Subclasses can override
        this method for more sophisticated adaptation strategies.
        """
        factor = 2.0 if success else 0.5
        new_step = current_step * factor
        clipped_mag = np.clip(np.abs(new_step), 1e-10, 1.0)
        return np.sign(new_step) * clipped_mag

    @abstractmethod
    def _stop_condition(self) -> bool:  
        """Check if continuation should terminate.

        This method evaluates problem-specific termination criteria such as
        reaching target parameter values, detecting bifurcations, or
        encountering solution boundaries.

        Returns
        -------
        bool
            True if continuation should stop, False to continue.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Common stopping criteria include:
        - Parameter reaching target range boundaries
        - Solution stability changes (bifurcations)
        - Physical constraints (e.g., energy limits)
        - Convergence to known solutions
        """
        raise NotImplementedError("_stop_condition must be provided by a sub-class")

    @staticmethod
    def _clamp_step(
        step_value: float,
        *,
        reference_value: float = 1.0,
        min_relative: float = 1e-6,
        min_absolute: float = 1e-8,
    ) -> float:
        """Clamp step size to enforce minimum magnitude while preserving sign.

        This utility method ensures step sizes don't become too small to
        make numerical progress while preserving the continuation direction.

        Parameters
        ----------
        step_value : float
            Step size to be clamped.
        reference_value : float, default 1.0
            Reference value for relative minimum calculation.
        min_relative : float, default 1e-6
            Minimum step as fraction of reference value.
        min_absolute : float, default 1e-8
            Absolute minimum step size.

        Returns
        -------
        float
            Clamped step value with same sign as input.

        Notes
        -----
        The minimum step is the larger of min_absolute and
        min_relative * |reference_value|. This ensures progress
        in both absolute and relative terms.
        """
        if step_value == 0:
            return min_absolute

        ref_mag = abs(reference_value)
        min_step = max(min_absolute, ref_mag * min_relative) if ref_mag > min_absolute else min_absolute
        return np.sign(step_value) * max(min_step, abs(step_value))

    @staticmethod
    def _clamp_scale(scale_value: float, *, min_scale: float = 1e-3, max_scale: float = 1e3) -> float:
        """Clamp multiplicative scaling factors to prevent extreme adaptations.

        This utility method bounds scaling factors used in step size adaptation
        to maintain numerical stability and reasonable continuation behavior.

        Parameters
        ----------
        scale_value : float
            Scaling factor to be clamped.
        min_scale : float, default 1e-3
            Minimum allowed scaling factor.
        max_scale : float, default 1e3
            Maximum allowed scaling factor.

        Returns
        -------
        float
            Clamped scaling factor in [min_scale, max_scale].

        Notes
        -----
        This prevents extreme step size changes that could destabilize
        the continuation algorithm or cause numerical overflow/underflow.
        """
        return float(np.clip(scale_value, min_scale, max_scale))

    def __repr__(self) -> str:
        """Return a string representation of the continuation engine.

        Returns
        -------
        str
            Compact representation showing current state and configuration.
        """  
        return (
            f"{self.__class__.__name__}(n_members={len(self._family)}, step={self._step}, "
            f"target=[[{self._target_min}], [{self._target_max}]])"
        )

    def _on_accept(self, candidate: object) -> None:
        """Execute a hook after a candidate is accepted into the family.

        This method provides an extension point for subclasses to perform
        custom processing after successful solution acceptance without
        reimplementing the main continuation loop.

        Parameters
        ----------
        candidate : object
            The corrected solution that was just accepted into the family.

        Notes
        -----
        Common uses for this hook include:
        - Updating tangent vectors for pseudo-arclength continuation
        - Computing stability information
        - Saving intermediate results
        - Triggering bifurcation detection
        
        The default implementation does nothing. Exceptions raised by
        this method are caught and logged but do not stop continuation.
        """
        pass

    @abstractmethod
    def _make_stepper(self) -> _ContinuationStep:
        """Create the stepping strategy for this continuation run.

        This method must return a stepping strategy object that implements
        the prediction phase of the continuation algorithm. The strategy
        determines how to predict the next solution based on the current
        solution and step size.

        Returns
        -------
        :class:`~hiten.algorithms.continuation.strategies._step_interface._ContinuationStep`
            Stepping strategy instance for solution prediction.

        Notes
        -----
        This is an abstract method that must be implemented by subclasses.
        Common stepping strategies include:
        
        - Natural parameter continuation: vary one parameter linearly
        - Pseudo-arclength continuation: follow solution curve in extended space
        - Tangent prediction: use solution derivatives for prediction
        
        The stepping strategy should be compatible with the domain objects
        and parameter structure used by the continuation problem.

        Examples
        --------
        >>> def _make_stepper(self):
        ...     return NaturalParameterStep(parameter_index=0)
        """
        raise NotImplementedError










