"""Define the plain step interface for step-size control strategies.

This module provides the plain step interface for step-size control strategies.
"""

import numpy as np

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.stepping.base import _CorrectorStepBase
from hiten.algorithms.corrector.types import NormFn, ResidualFn


class _CorrectorPlainStep(_CorrectorStepBase):
    """Provide a step interface for plain Newton updates with safeguards.

    This class implements the simplest step-size control strategy: taking
    full Newton steps with optional step size capping for numerical stability.
    It provides a robust baseline stepping strategy suitable for well-behaved
    problems where the Newton method converges reliably.

    The interface includes an infinity-norm safeguard that prevents
    excessively large steps, which can cause numerical overflow or
    instability. This makes it suitable for a wide range of problems
    while maintaining the simplicity of the basic Newton method.

    See Also
    --------
    :class:`~hiten.algorithms.corrector.stepping.armijo._ArmijoStep`
        More sophisticated interface with line search capabilities.
    :class:`~hiten.algorithms.corrector.stepping.base._CorrectorStepBase`
        Abstract base class that this class extends.
    """

    def _make_plain_stepper(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> CorrectorStepProtocol:
        """Create a plain Newton stepper with optional step size capping.

        This method builds a step transformation function that implements
        the plain Newton update with an optional infinity-norm safeguard.
        The resulting stepper takes full Newton steps unless the step
        size exceeds the specified maximum.

        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual vectors.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed infinity norm of the Newton step.
            If None or infinite, no capping is applied.

        Returns
        -------
        stepper : :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
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
                    scale = max_delta / delta_norm
                    delta = delta * scale
                    x_new = x + delta
                    r_norm_new = norm_fn(residual_fn(x_new))
                    return x_new, r_norm_new, float(scale)

            # Apply the (uncapped) Newton step
            x_new = x + delta
            r_norm_new = norm_fn(residual_fn(x_new))
            return x_new, r_norm_new, 1.0

        return _plain_step

    def _build_line_searcher(
        self,
        residual_fn: ResidualFn,
        norm_fn: NormFn,
        max_delta: float | None,
    ) -> CorrectorStepProtocol:
        """Build a plain Newton stepper for the current problem.

        This method implements the abstract interface by delegating to
        the plain stepper implementation. Despite the name "line_searcher",
        this implementation does not perform line search but provides
        a consistent interface for step transformation.

        Parameters
        ----------
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual vectors.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute residual norms.
        max_delta : float or None
            Maximum allowed step size for safeguarding.

        Returns
        -------
        stepper : :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
            Plain Newton step transformation function.
        """
        return self._make_plain_stepper(residual_fn, norm_fn, max_delta)