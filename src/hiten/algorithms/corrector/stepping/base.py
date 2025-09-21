"""Define the base class for step-size control strategies.

This module provides the base class for step-size control strategies.
"""

from abc import ABC, abstractmethod

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.types import NormFn, ResidualFn


class _CorrectorStepBase(ABC):
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

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed to parent classes.
        This enables clean cooperation in multiple-inheritance chains.

    Notes
    -----
    The interface is designed to work seamlessly with multiple inheritance,
    allowing correction algorithms to mix step interfaces with other
    capabilities (convergence monitoring, Jacobian computation, etc.).

    The abstract method :meth:`~hiten.algorithms.corrector.stepping.base._CorrectorStepBase._build_line_searcher` is responsible for
    creating :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol` objects that encapsulate the step
    transformation logic for specific problems.

    Examples
    --------
    >>> class CustomStepInterface(_CorrectorStepBase):
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
    :class:`~hiten.algorithms.corrector.stepping.plain._CorrectorPlainStep`
        Concrete implementation for simple Newton steps.
    :class:`~hiten.algorithms.corrector.stepping.armijo._ArmijoStep`
        Concrete implementation with Armijo line search.
    :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
        Protocol for step transformation functions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> CorrectorStepProtocol:
        """Build a step transformation function for the current problem.

        This method creates a :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol` object that encapsulates
        the step-size control logic for a specific nonlinear system.
        The stepper uses the provided residual and norm functions to
        evaluate candidate steps and determine appropriate step sizes.

        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function that computes residual vectors from state vectors.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function that computes scalar norms from residual vectors.
        max_delta : float or None
            Maximum allowed step size (infinity norm), or None for
            no limit. Used as a safeguard against excessively large steps.

        Returns
        -------
        stepper : :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
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
        ...
