"""Provide base classes and configuration for iterative correction algorithms.

This module provides the foundational components for implementing iterative
correction algorithms used throughout the hiten framework. These algorithms
solve nonlinear systems of equations that arise in dynamical systems analysis,
such as finding periodic orbits, invariant manifolds, and fixed points.

The correction framework is designed to work with abstract vector representations,
allowing domain-specific objects (orbits, manifolds, etc.) to be corrected
using the same underlying algorithms. This promotes code reuse and enables
consistent numerical behavior across different problem domains.

See Also
--------
:mod:`~hiten.algorithms.corrector.backends.newton`
    Newton-Raphson correction implementations.
:mod:`~hiten.algorithms.corrector.interfaces`
    Interface classes for different correction strategies.
:mod:`~hiten.algorithms.corrector.stepping`
    Step-size control interfaces for robust convergence.
"""

from abc import abstractmethod
from typing import Any, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn
from hiten.algorithms.types.core import _HitenBaseBackend


class _CorrectorBackend(_HitenBaseBackend):
    """Define an abstract base class for iterative correction algorithms.

    This class defines the interface for iterative correction algorithms
    used throughout the hiten framework to solve nonlinear systems of
    equations. It provides a generic, domain-independent interface that
    can be specialized for different types of problems (periodic orbits,
    invariant manifolds, fixed points, etc.).

    Notes
    -----
    Subclasses must implement the 
    :meth:`~hiten.algorithms.corrector.backends.base._CorrectorBackend.correct` 
    method and are expected
    to document any additional keyword arguments specific to their
    correction strategy (maximum iterations, tolerances, step control
    parameters, etc.).

    Examples
    --------
    >>> # Typical usage pattern (conceptual)
    >>> class NewtonCorrector(_CorrectorBackend):
    ...     def correct(self, x0, residual_fn, **kwargs):
    ...         # Newton-Raphson implementation
    ...         pass
    >>>
    >>> corrector = NewtonCorrector()
    >>> x_corrected, info = corrector.correct(
    ...     x0=initial_guess,
    ...     residual_fn=lambda x: compute_constraints(x),
    ...     jacobian_fn=lambda x: compute_jacobian(x)
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.corrector.config._BaseCorrectionConfig`
        Configuration class for correction parameters.
    :mod:`~hiten.algorithms.corrector.backends.newton`
        Concrete Newton-Raphson implementations.
    :mod:`~hiten.algorithms.corrector.stepping`
        Step-size control interfaces for robust convergence.
    """

    # NOTE: Subclasses are expected to document additional keyword arguments
    # (max_iter, tolerance, step control parameters, etc.) relevant to their
    # specific correction strategy. This documentation should include:
    # - Parameter descriptions with types and defaults
    # - Algorithm-specific behavior and limitations
    # - Performance characteristics and trade-offs
    # - Recommended parameter ranges for different problem types

    @abstractmethod
    def run(
        self,
        x0: np.ndarray,
        residual_fn: ResidualFn,
        *,
        jacobian_fn: JacobianFn | None = None,
        norm_fn: NormFn | None = None,
        stepper_factory: Callable[[ResidualFn, NormFn, float | None], CorrectorStepProtocol] | None = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Any]:
        """Solve nonlinear system to find x such that ||R(x)|| < tolerance.

        This method implements the core correction algorithm, iteratively
        refining an initial guess until the residual norm falls below the
        specified tolerance or the maximum number of iterations is reached.

        The method is designed to handle a wide range of nonlinear systems
        arising in dynamical systems analysis, with particular emphasis on
        robustness and numerical stability for problems in astrodynamics.

        Parameters
        ----------
        x0 : ndarray
            Initial guess for the parameter vector. Should be reasonably
            close to the expected solution for best convergence properties.
            The quality of the initial guess significantly affects both
            convergence rate and success probability.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function computing the residual vector R(x) for parameter
            vector x. The residual should be zero (or close to zero) at
            the desired solution. Must be well-defined and preferably
            continuous in a neighborhood of the solution.
        jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn`, optional
            Function returning the Jacobian matrix J(x) = dR/dx. If not
            provided, implementations may use finite-difference approximation
            or other Jacobian-free methods. Analytic Jacobians generally
            provide better convergence properties.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`, optional
            Custom norm function for assessing convergence. If not provided,
            implementations typically default to the L2 (Euclidean) norm.
            The choice of norm can affect convergence behavior and should
            be appropriate for the problem scaling.
        stepper_factory : callable, optional
            Factory producing a :class:`~hiten.algorithms.corrector.protocols.CorrectorStepProtocol`
            instance for the current problem. Allows callers to override the
            backend's default step strategy on a per-problem basis.
        **kwargs
            Additional algorithm-specific parameters. Common parameters
            include maximum iterations, convergence tolerance, step control
            settings, and line search configuration. See subclass
            documentation for specific options.

        Returns
        -------
        x_corrected : ndarray
            Corrected parameter vector satisfying ||R(x_corrected)|| < tol.
            Has the same shape as the input x0.
        info : Any
            Algorithm-specific auxiliary information about the correction
            process. Common contents include:
            - Number of iterations performed
            - Final residual norm achieved
            - Convergence status and diagnostics
            - Computational cost metrics
            The exact structure and content is implementation-defined.

        Raises
        ------
        ConvergenceError
            If the algorithm fails to converge within the specified
            maximum number of iterations or encounters numerical difficulties.
        ValueError
            If input parameters are invalid or incompatible.

        Examples
        --------
        >>> # Basic usage with analytic Jacobian
        >>> x_corr, info = corrector.correct(
        ...     x0=np.array([1.0, 0.0, 0.5]),
        ...     residual_fn=lambda x: compute_orbit_constraints(x),
        ...     jacobian_fn=lambda x: compute_constraint_jacobian(x)
        ... )
        >>>
        >>> # Usage with custom norm and finite differences
        >>> x_corr, info = corrector.correct(
        ...     x0=initial_state,
        ...     residual_fn=manifold_constraints,
        ...     norm_fn=lambda r: np.linalg.norm(r, ord=np.inf),
        ...     max_attempts=100,
        ...     tol=1e-12
        ... )
        """
        # Subclasses must provide concrete implementation
        raise NotImplementedError("Subclasses must implement the correct method")
