"""Abstract base classes for different continuation algorithm strategies.

This module provides abstract base classes that implement specific continuation
algorithm strategies by extending the generic continuation engine. Each strategy
defines a particular approach to parameter continuation with specialized stepping
logic, stopping criteria, and parameter handling.

See Also
--------
:mod:`hiten.algorithms.continuation.base`
    Base continuation engine that these strategies extend.
:mod:`hiten.algorithms.continuation.strategies._stepping`
    Stepping strategy implementations used by these algorithms.
:mod:`hiten.algorithms.continuation.predictors`
    Concrete implementations that use these algorithm strategies.
"""

from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine
from hiten.algorithms.continuation.strategies._stepping import _SecantStep


class _NaturalParameter(_ContinuationEngine, ABC):
    """Abstract base class for natural parameter continuation algorithms.

    This class implements the natural parameter continuation strategy, where
    one or more parameters are varied monotonically within specified target
    ranges. The algorithm ensures that continuation progresses toward the
    target interval and terminates when parameters leave the prescribed bounds.

    Natural parameter continuation is the simplest and most robust continuation
    method, suitable for tracing solution families where parameters can be
    varied independently without encountering turning points or bifurcations
    that would require more sophisticated methods.

    Parameters
    ----------
    *args, **kwargs
        Arguments passed to the base :class:`_ContinuationEngine`.
        See base class documentation for parameter details.

    Notes
    -----
    The natural parameter strategy implements:
    
    1. **Monotonic progression**: Parameters advance monotonically toward targets
    2. **Direction enforcement**: Initial steps are oriented toward target ranges
    3. **Interval-based stopping**: Continuation stops when parameters exit bounds
    4. **Component-wise handling**: Multi-parameter cases handled independently
    
    This strategy is most effective when:
    - Solution families are well-behaved (no turning points)
    - Parameters can be varied independently
    - Target ranges are reachable via monotonic progression
    - Bifurcations or fold points are not expected

    For more complex scenarios with turning points or bifurcations,
    consider using :class:`_SecantArcLength` continuation instead.

    Examples
    --------
    >>> # Natural parameter continuation is typically used via concrete classes
    >>> class OrbitContinuation(_PeriodicOrbitContinuationInterface, _NaturalParameter):
    ...     def _make_stepper(self):
    ...         from hiten.algorithms.continuation.strategies import _NaturalParameterStep
    ...         return _NaturalParameterStep(self._predict_fn)
    ...
    >>> engine = OrbitContinuation(
    ...     initial_orbit=orbit0,
    ...     parameter_getter=lambda o: o.energy,
    ...     target=(3.0, 4.0),
    ...     step=0.01
    ... )

    See Also
    --------
    :class:`_SecantArcLength`
        Alternative strategy for pseudo-arclength continuation.
    :class:`hiten.algorithms.continuation.base._ContinuationEngine`
        Base continuation engine that this class extends.
    :class:`hiten.algorithms.continuation.predictors._StateParameter`
        Concrete implementation using natural parameter continuation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize natural parameter continuation with direction enforcement.

        This method initializes the base continuation engine and enforces
        natural parameter policies including monotonic parameter advancement
        toward target intervals and proper step direction orientation.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to the base continuation engine.

        Notes
        -----
        The initialization performs automatic step direction correction:
        if the initial step would move parameters away from their target
        ranges, the step signs are flipped component-wise to ensure
        progression toward the targets.
        """
        super().__init__(*args, **kwargs)

        # Ensure the initial step points from the current parameter value toward
        # the target interval.  If it does not, flip its sign component-wise.
        current_param = self._param_history[-1]
        for i in range(current_param.size):
            if (current_param[i] < self._target_min[i] and self._step[i] < 0) or (
                current_param[i] > self._target_max[i] and self._step[i] > 0
            ):
                self._step[i] = -self._step[i]

    def _stop_condition(self) -> bool:
        """Check if continuation should terminate based on parameter bounds.

        This method implements the natural parameter stopping criterion:
        continuation terminates when any parameter component exits its
        prescribed target interval [min, max].

        Returns
        -------
        bool
            True if any parameter is outside its target range, False otherwise.

        Notes
        -----
        The stopping condition is evaluated component-wise for multi-parameter
        continuation. Continuation stops as soon as any parameter exits its
        bounds, ensuring that the generated family stays within the specified
        parameter ranges.
        """
        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    def _make_stepper(self):
        """Create stepping strategy for natural parameter continuation.

        This abstract method must be implemented by subclasses to provide
        the specific stepping strategy appropriate for the problem domain.

        Returns
        -------
        :class:`hiten.algorithms.continuation.strategies._step_interface._ContinuationStep`
            Stepping strategy instance for natural parameter continuation.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.

        Notes
        -----
        Subclasses must implement this method to specify how predictions
        are generated for the specific problem type. Common implementations
        include :class:`hiten.algorithms.continuation.strategies._stepping._NaturalParameterStep`
        for state-based continuation.

        Examples
        --------
        >>> def _make_stepper(self):
        ...     from hiten.algorithms.continuation.strategies import _NaturalParameterStep
        ...     return _NaturalParameterStep(self._predict_function)
        """
        raise NotImplementedError(
            "Natural-parameter continuations must define a StepStrategy by "
            "overriding _make_stepper() or assigning self._stepper before "
            "calling super().__init__."
        )


class _SecantArcLength(_ContinuationEngine, ABC):
    """Abstract base class for pseudo-arclength continuation algorithms.

    This class implements the pseudo-arclength continuation strategy, which
    follows solution curves in an extended parameter-solution space. This
    method can navigate around turning points and bifurcations that would
    cause natural parameter continuation to fail.

    Pseudo-arclength continuation parameterizes the solution curve by
    arclength rather than by a natural parameter, allowing it to trace
    folded branches and handle cases where the Jacobian becomes singular
    with respect to the continuation parameter.

    Parameters
    ----------
    *args, **kwargs
        Arguments passed to the base :class:`_ContinuationEngine`.
        See base class documentation for parameter details.

    Notes
    -----
    The pseudo-arclength strategy implements:
    
    1. **Extended space**: Augments parameter-solution space with arclength
    2. **Secant prediction**: Uses secant method for tangent estimation
    3. **Orthogonal correction**: Maintains prescribed arclength steps
    4. **Fold navigation**: Can pass through turning points smoothly
    
    This strategy is most effective when:
    - Solution families have turning points or folds
    - Natural parameter continuation fails due to singularities
    - Bifurcation points need to be traversed
    - Complete solution branches need to be traced

    The method requires a representation function that maps domain
    objects to numerical vectors for arclength computation.

    Examples
    --------
    >>> # Pseudo-arclength continuation via concrete implementation
    >>> class OrbitArcLength(_PeriodicOrbitContinuationInterface, _SecantArcLength):
    ...     def _representation(self, orbit):
    ...         return orbit.initial_state
    ...
    >>> engine = OrbitArcLength(
    ...     initial_orbit=orbit0,
    ...     parameter_getter=lambda o: o.energy,
    ...     target=(3.0, 4.0),
    ...     step=0.01
    ... )

    See Also
    --------
    :class:`_NaturalParameter`
        Simpler strategy for monotonic parameter continuation.
    :class:`hiten.algorithms.continuation.strategies._stepping._SecantStep`
        Stepping strategy used by this algorithm.
    :class:`hiten.algorithms.continuation.base._ContinuationEngine`
        Base continuation engine that this class extends.
    """

    def __init__(self, *args, **kwargs):
        """Initialize pseudo-arclength continuation with secant stepping.

        This method initializes the base continuation engine and sets up
        the secant-based stepping strategy for pseudo-arclength continuation.
        The stepping strategy is configured with representation and parameter
        extraction functions.

        Parameters
        ----------
        *args, **kwargs
            Arguments passed to the base continuation engine.

        Notes
        -----
        The initialization automatically creates and configures a
        :class:`_SecantStep` strategy using the abstract :meth:`_representation`
        method that must be implemented by subclasses.
        """
        super().__init__(*args, **kwargs)

        # Build and assign secant stepper strategy
        self._stepper = _SecantStep(
            representation_fn=self._representation,
            parameter_fn=lambda obj: np.asarray(self._parameter(obj), dtype=float),
        )

        # Notify strategy of initialisation
        if hasattr(self._stepper, "on_initialisation"):
            self._stepper.on_initialisation(self._family[0])

    @abstractmethod
    def _representation(self, obj: object) -> np.ndarray:
        """Convert domain object to numerical representation for arclength computation.

        This method must be implemented by subclasses to provide a numerical
        representation of domain objects that can be used for arclength
        calculations in the extended parameter-solution space.

        Parameters
        ----------
        obj : object
            Domain object (e.g., periodic orbit, equilibrium) to represent.

        Returns
        -------
        ndarray
            Numerical vector representation of the object, typically
            containing state variables or other characteristic quantities.

        Notes
        -----
        The representation should capture the essential degrees of freedom
        of the solution and be suitable for computing distances in the
        extended space. Common choices include:
        
        - Initial state vectors for periodic orbits
        - Fourier coefficients for quasi-periodic solutions
        - Equilibrium positions for fixed points
        
        The quality of the representation affects the behavior of the
        pseudo-arclength continuation algorithm.

        Examples
        --------
        >>> def _representation(self, orbit):
        ...     # Use initial state for orbit representation
        ...     return orbit.initial_state
        >>> 
        >>> def _representation(self, equilibrium):
        ...     # Use position for equilibrium representation
        ...     return equilibrium.position
        """
        pass

    def _stop_condition(self) -> bool:
        """Check if continuation should terminate based on parameter bounds.

        This method implements the same parameter-based stopping criterion
        as natural parameter continuation, terminating when any parameter
        component exits its prescribed target interval.

        Returns
        -------
        bool
            True if any parameter is outside its target range, False otherwise.

        Notes
        -----
        Even though pseudo-arclength continuation follows curves in extended
        space, the stopping criterion is still based on the original
        continuation parameters. This ensures that the generated family
        covers the desired parameter range.
        """
        current = self._param_history[-1]
        return np.any(current < self._target_min) or np.any(current > self._target_max)

    def _on_accept(self, member: object) -> None:
        """Hook for additional processing after solution acceptance.

        This method provides an extension point for subclasses to perform
        custom processing after a solution is accepted. In pseudo-arclength
        continuation, tangent vector updates are handled by the stepping
        strategy, so this hook is primarily for additional bookkeeping.

        Parameters
        ----------
        member : object
            The solution object that was just accepted into the family.

        Notes
        -----
        The default implementation does nothing since tangent vector
        maintenance is handled automatically by the :class:`_SecantStep`
        strategy. Subclasses can override this method for:
        
        - Computing additional solution properties
        - Saving intermediate results
        - Updating auxiliary data structures
        - Triggering analysis or visualization
        """
        pass

    def _make_stepper(self):
        """Create stepping strategy for pseudo-arclength continuation.

        This method should not normally be called since the stepping strategy
        is automatically created during initialization. It exists for
        compatibility with the base class interface.

        Returns
        -------
        :class:`hiten.algorithms.continuation.strategies._stepping._SecantStep`
            The secant-based stepping strategy.

        Raises
        ------
        NotImplementedError
            This method should not be called directly.

        Notes
        -----
        Unlike natural parameter continuation, pseudo-arclength continuation
        automatically configures its stepping strategy during initialization.
        The :class:`_SecantStep` strategy is created with the representation
        and parameter functions provided by the subclass.
        """
        raise NotImplementedError(
            "Secant-based continuations must define a StepStrategy by "
            "overriding _make_stepper() or assigning self._stepper before "
            "calling super().__init__."
        )