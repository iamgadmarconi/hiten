"""Define step-size control interfaces for Newton-type correction algorithms.

This module provides abstract interfaces and concrete implementations for
step-size control strategies used in Newton-type iterative methods. These
interfaces enable different stepping strategies (plain Newton steps, line
search methods, trust region approaches) to be used interchangeably within
correction algorithms.

The module implements a protocol-based design that separates the step
computation logic from the overall Newton iteration framework, allowing
for flexible combinations of different stepping strategies with various
correction algorithms.

See Also
--------
:mod:`hiten.algorithms.corrector.line`
    Line search implementations used by step interfaces.
:mod:`hiten.algorithms.corrector.newton`
    Newton correction algorithms that use these interfaces.
:mod:`hiten.algorithms.corrector.base`
    Base correction framework that coordinates with step interfaces.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Protocol

import numpy as np

from hiten.algorithms.corrector.line import (_ArmijoLineSearch,
                                             _LineSearchConfig)

#: Type alias for residual function signatures.
#:
#: Functions of this type compute residual vectors from state vectors,
#: typically representing the nonlinear equations to be solved. The
#: residual should be zero (or close to zero) at the solution.
#:
#: Parameters
#: ----------
#: x : ndarray
#:     State vector at which to evaluate the residual.
#:
#: Returns
#: -------
#: residual : ndarray
#:     Residual vector of the same shape as the input.
ResidualFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for norm function signatures.
#:
#: Functions of this type compute scalar norms from residual vectors,
#: providing a measure of how close the current iterate is to satisfying
#: the nonlinear equations. Common choices include L2 norm, infinity norm,
#: and weighted norms.
#:
#: Parameters
#: ----------
#: residual : ndarray
#:     Residual vector to compute the norm of.
#:
#: Returns
#: -------
#: norm : float
#:     Scalar norm value (non-negative).
NormFn = Callable[[np.ndarray], float]


class _Stepper(Protocol):
    """Define the protocol for step transformation functions in Newton-type methods.

    This protocol defines the interface for functions that transform a
    computed Newton step into an accepted update. Different implementations
    can provide various step-size control strategies, from simple full
    steps to sophisticated line search and trust region methods.

    The protocol enables separation of concerns between:
    - Newton step computation (direction finding)
    - Step size control (distance along direction)
    - Convergence monitoring (residual evaluation)

    Implementations typically handle:
    - Step size scaling for convergence control
    - Safeguards against excessive step sizes
    - Line search for sufficient decrease conditions
    - Trust region constraints for robustness

    Parameters
    ----------
    x : ndarray
        Current iterate in the Newton method.
    delta : ndarray
        Newton step direction (typically from solving J*delta = -F).
    current_norm : float
        Norm of the residual at the current iterate *x*.

    Returns
    -------
    x_new : ndarray
        Updated iterate after applying the step transformation.
    r_norm_new : float
        Norm of the residual at the new iterate *x_new*.
    alpha_used : float
        Step-size scaling factor actually employed. A value of 1.0
        indicates the full Newton step was taken, while smaller values
        indicate step size reduction for convergence control.

    Notes
    -----
    The protocol allows for flexible step-size control strategies:
    
    - **Full Newton steps**: alpha_used = 1.0, x_new = x + delta
    - **Scaled steps**: alpha_used < 1.0, x_new = x + alpha * delta
    - **Line search**: alpha chosen to satisfy decrease conditions
    - **Trust region**: delta modified to stay within trust region
    
    Implementations should ensure that r_norm_new is computed consistently
    with the norm function used in the overall Newton algorithm.

    Examples
    --------
    >>> # Simple full-step implementation
    >>> def full_step(x, delta, current_norm):
    ...     x_new = x + delta
    ...     r_norm_new = norm_fn(residual_fn(x_new))
    ...     return x_new, r_norm_new, 1.0
    >>>
    >>> # Scaled step implementation
    >>> def scaled_step(x, delta, current_norm):
    ...     alpha = 0.5  # Half step
    ...     x_new = x + alpha * delta
    ...     r_norm_new = norm_fn(residual_fn(x_new))
    ...     return x_new, r_norm_new, alpha

    See Also
    --------
    :class:`_PlainStepInterface`
        Simple implementation with optional step size capping.
    :class:`_ArmijoStepInterface`
        Line search implementation using Armijo conditions.
    """

    def __call__(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        current_norm: float,
    ) -> tuple[np.ndarray, float, float]:
        """Transform Newton step into accepted update.

        Parameters
        ----------
        x : ndarray
            Current iterate.
        delta : ndarray
            Newton step direction.
        current_norm : float
            Norm of residual at current iterate.

        Returns
        -------
        x_new : ndarray
            Updated iterate.
        r_norm_new : float
            Norm of residual at new iterate.
        alpha_used : float
            Step scaling factor employed.
        """
        ...


class _StepInterface(ABC):
    """Provide an abstract base class for step-size control strategy interfaces.

    This class provides the foundation for implementing different step-size
    control strategies in Newton-type correction algorithms. It defines the
    interface that correction algorithms use to obtain step transformation
    functions tailored to specific problems.

    The interface follows the strategy pattern, allowing correction algorithms
    to be parameterized with different stepping behaviors without changing
    their core logic. This enables flexible combinations of:

    - Different Newton variants (standard, damped, quasi-Newton)
    - Different step control strategies (full steps, line search, trust region)
    - Different problem-specific constraints and safeguards

    Subclasses must implement the step transformation logic while this base
    class handles common initialization patterns and ensures compatibility
    with multiple inheritance chains commonly used in the correction framework.

    Notes
    -----
    The interface is designed to work seamlessly with multiple inheritance,
    allowing correction algorithms to mix step interfaces with other
    capabilities (convergence monitoring, Jacobian computation, etc.).

    The abstract method :meth:`_build_line_searcher` is responsible for
    creating :class:`_Stepper` objects that encapsulate the step
    transformation logic for specific problems.

    Examples
    --------
    >>> class CustomStepInterface(_StepInterface):
    ...     def _build_line_searcher(self, residual_fn, norm_fn, max_delta):
    ...         def custom_step(x, delta, current_norm):
    ...             # Custom step logic here
    ...             alpha = compute_step_size(x, delta, current_norm)
    ...             x_new = x + alpha * delta
    ...             r_norm_new = norm_fn(residual_fn(x_new))
    ...             return x_new, r_norm_new, alpha
    ...         return custom_step

    See Also
    --------
    :class:`_PlainStepInterface`
        Concrete implementation for simple Newton steps.
    :class:`_ArmijoStepInterface`
        Concrete implementation with Armijo line search.
    :class:`_Stepper`
        Protocol for step transformation functions.
    """

    def __init__(self, **kwargs):
        """Initialize step interface with multiple inheritance support.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to parent classes.
            This enables clean cooperation in multiple-inheritance chains.
        """
        # Allow clean cooperation in multiple-inheritance chains
        super().__init__(**kwargs)

    @abstractmethod
    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Build a step transformation function for the current problem.

        This method creates a :class:`_Stepper` object that encapsulates
        the step-size control logic for a specific nonlinear system.
        The stepper uses the provided residual and norm functions to
        evaluate candidate steps and determine appropriate step sizes.

        Parameters
        ----------
        residual_fn : ResidualFn
            Function that computes residual vectors from state vectors.
        norm_fn : NormFn
            Function that computes scalar norms from residual vectors.
        max_delta : float or None
            Maximum allowed step size (infinity norm), or None for
            no limit. Used as a safeguard against excessively large steps.

        Returns
        -------
        stepper : :class:`_Stepper`
            Step transformation function configured for this problem.

        Notes
        -----
        The returned stepper should be thread-safe and reusable for
        multiple Newton iterations on the same problem. It typically
        captures the residual and norm functions in a closure.

        The max_delta parameter provides a safety mechanism to prevent
        numerical overflow or instability from very large Newton steps,
        which can occur with poorly conditioned problems or bad initial
        guesses.
        """


class _PlainStepInterface(_StepInterface):
    """Provide a step interface for plain Newton updates with safeguards.

    This class implements the simplest step-size control strategy: taking
    full Newton steps with optional step size capping for numerical stability.
    It provides a robust baseline stepping strategy suitable for well-behaved
    problems where the Newton method converges reliably.

    The interface includes an infinity-norm safeguard that prevents
    excessively large steps, which can cause numerical overflow or
    instability. This makes it suitable for a wide range of problems
    while maintaining the simplicity of the basic Newton method.

    Features:
    - Full Newton steps (alpha = 1.0) when possible
    - Infinity-norm capping for numerical stability
    - Minimal computational overhead
    - Predictable behavior for debugging and analysis

    This stepping strategy is recommended for:
    - Well-conditioned problems with good initial guesses
    - Problems where Newton's method converges reliably
    - Situations where computational efficiency is prioritized
    - Debugging and development of correction algorithms

    Notes
    -----
    While simple, this interface provides a solid foundation for more
    sophisticated stepping strategies. The optional step size capping
    prevents the most common failure modes while preserving the fast
    convergence properties of Newton's method.

    The implementation uses the infinity norm for step size measurement,
    which provides component-wise control and is computationally efficient.

    See Also
    --------
    :class:`_ArmijoStepInterface`
        More sophisticated interface with line search capabilities.
    :class:`_StepInterface`
        Abstract base class that this class extends.
    """

    def _make_plain_stepper(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Create a plain Newton stepper with optional step size capping.

        This method builds a step transformation function that implements
        the plain Newton update with an optional infinity-norm safeguard.
        The resulting stepper takes full Newton steps unless the step
        size exceeds the specified maximum.

        Parameters
        ----------
        residual_fn : ResidualFn
            Function to compute residual vectors.
        norm_fn : NormFn
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed infinity norm of the Newton step.
            If None or infinite, no capping is applied.

        Returns
        -------
        stepper : _Stepper
            Step transformation function implementing plain Newton updates.

        Notes
        -----
        The step size capping algorithm:
        
        1. Compute the infinity norm of the Newton step
        2. If the norm exceeds max_delta, scale the step proportionally
        3. Apply the (possibly scaled) step to get the new iterate
        4. Evaluate the residual norm at the new iterate
        5. Return the new iterate, residual norm, and effective step size
        
        The effective step size is always 1.0 for this implementation,
        even when step capping is applied, since the capping modifies
        the step direction rather than scaling it.
        """
        def _plain_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            # Optional safeguard against excessively large steps
            if (max_delta is not None) and (not np.isinf(max_delta)):
                delta_norm = float(np.linalg.norm(delta, ord=np.inf))
                if delta_norm > max_delta:
                    delta = delta * (max_delta / delta_norm)

            # Apply the (possibly capped) Newton step
            x_new = x + delta
            r_norm_new = norm_fn(residual_fn(x_new))
            return x_new, r_norm_new, 1.0

        return _plain_step

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Build a plain Newton stepper for the current problem.

        This method implements the abstract interface by delegating to
        the plain stepper implementation. Despite the name "line_searcher",
        this implementation does not perform line search but provides
        a consistent interface for step transformation.

        Parameters
        ----------
        residual_fn : ResidualFn
            Function to compute residual vectors.
        norm_fn : NormFn
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed step size for safeguarding.

        Returns
        -------
        stepper : _Stepper
            Plain Newton step transformation function.
        """
        return self._make_plain_stepper(residual_fn, norm_fn, max_delta)


class _ArmijoStepInterface(_PlainStepInterface):
    """Provide a step interface with Armijo line search for robust convergence.

    This class extends the plain step interface with optional Armijo line
    search capabilities. It provides a more robust stepping strategy that
    can handle poorly conditioned problems, bad initial guesses, and
    nonlinear systems where full Newton steps might diverge.

    The interface supports both plain Newton steps (for efficiency) and
    Armijo line search (for robustness), with the choice determined by
    configuration. This flexibility allows algorithms to adapt their
    stepping strategy based on problem characteristics or user preferences.

    Features:
    - Optional Armijo line search with backtracking
    - Configurable line search parameters
    - Fallback to plain Newton steps when line search is disabled
    - Automatic sufficient decrease condition checking
    - Robust convergence for challenging problems

    The Armijo line search ensures sufficient decrease in the residual norm,
    providing theoretical convergence guarantees under appropriate conditions.
    This makes it suitable for:

    - Poorly conditioned nonlinear systems
    - Problems with bad initial guesses
    - Situations requiring guaranteed convergence properties
    - Production code where robustness is critical

    Attributes
    ----------
    _line_search_config : _LineSearchConfig or None
        Configuration object for line search parameters.
    _use_line_search : bool
        Flag indicating whether line search should be used.

    Parameters
    ----------
    line_search_config : _LineSearchConfig, bool, or None, optional
        Line search configuration. Can be:
        - None: Disable line search (use plain Newton steps)
        - True: Enable line search with default parameters
        - False: Explicitly disable line search
        - _LineSearchConfig: Enable line search with custom parameters
    **kwargs
        Additional arguments passed to parent classes.

    Notes
    -----
    The interface inherits plain Newton step capabilities from its parent
    class, ensuring that it can fall back to simple stepping when line
    search is not needed or fails to improve convergence.

    The Armijo condition requires that the residual norm decrease by a
    sufficient amount proportional to the step size, providing a balance
    between convergence speed and robustness.

    Examples
    --------
    >>> # Enable line search with default parameters
    >>> interface = _ArmijoStepInterface(line_search_config=True)
    >>>
    >>> # Disable line search (use plain Newton)
    >>> interface = _ArmijoStepInterface(line_search_config=False)
    >>>
    >>> # Custom line search configuration
    >>> config = _LineSearchConfig(c1=1e-4, rho=0.5, max_iter=20)
    >>> interface = _ArmijoStepInterface(line_search_config=config)

    See Also
    --------
    :class:`_PlainStepInterface`
        Parent class providing plain Newton step capabilities.
    :class:`hiten.algorithms.corrector.line._ArmijoLineSearch`
        Line search implementation used by this interface.
    :class:`hiten.algorithms.corrector.line._LineSearchConfig`
        Configuration class for line search parameters.
    """

    _line_search_config: Optional[_LineSearchConfig]
    _use_line_search: bool

    def __init__(
        self,
        *,
        line_search_config: _LineSearchConfig | bool | None = None,
        **kwargs,
    ) -> None:
        """Initialize Armijo step interface with line search configuration.

        Parameters
        ----------
        line_search_config : :class:`_LineSearchConfig`, bool, or None, optional
            Line search configuration:
            - None: Disable line search, use plain Newton steps
            - True: Enable line search with default parameters
            - False: Explicitly disable line search
            - _LineSearchConfig: Enable line search with custom parameters
        **kwargs
            Additional arguments passed to parent classes.
        """
        super().__init__(**kwargs)

        # Process line-search configuration with flexible input handling
        if line_search_config is None:
            self._line_search_config = None
            self._use_line_search = False
        elif isinstance(line_search_config, bool):
            if line_search_config:
                self._line_search_config = _LineSearchConfig()
                self._use_line_search = True
            else:
                self._line_search_config = None
                self._use_line_search = False
        else:
            self._line_search_config = line_search_config
            self._use_line_search = True

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> _Stepper:
        """Build a step transformation function with optional line search.

        This method creates either a plain Newton stepper or an Armijo
        line search stepper based on the configuration. The choice is
        made at stepper creation time and remains fixed for the lifetime
        of the stepper.

        Parameters
        ----------
        residual_fn : ResidualFn
            Function to compute residual vectors.
        norm_fn : NormFn
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed step size (used by plain stepper fallback).

        Returns
        -------
        stepper : _Stepper
            Step transformation function, either plain Newton or Armijo
            line search based on configuration.

        Notes
        -----
        When line search is disabled, this method falls back to the
        plain Newton stepper from the parent class, ensuring consistent
        behavior and maintaining the step size capping safeguard.

        When line search is enabled, the method creates an Armijo line
        search object and wraps it in a stepper function that matches
        the expected interface.
        """
        if not getattr(self, "_use_line_search", False):
            return self._make_plain_stepper(residual_fn, norm_fn, max_delta)

        cfg = self._line_search_config
        searcher = _ArmijoLineSearch(
            config=cfg._replace(residual_fn=residual_fn, norm_fn=norm_fn)
        )

        def _armijo_step(x: np.ndarray, delta: np.ndarray, current_norm: float):
            """Armijo line search step transformation.
            
            This closure wraps the Armijo line search object to provide
            the standard stepper interface expected by Newton algorithms.
            """
            return searcher(x0=x, delta=delta, current_norm=current_norm)

        return _armijo_step