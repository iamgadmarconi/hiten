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
from typing import Any, Callable

import numpy as np

from hiten.algorithms.corrector.protocols import CorrectorStepProtocol
from hiten.algorithms.corrector.stepping import make_plain_stepper
from hiten.algorithms.corrector.types import (CorrectorInput, CorrectorOutput,
                                              JacobianCallable, NormCallable,
                                              ResidualCallable, StepperFactory)
from hiten.algorithms.types.core import _HitenBaseBackend
from hiten.utils.log_config import logger


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
    :class:`~hiten.algorithms.corrector.config.CorrectionConfig`
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

    def __init__(
        self,
        *,
        stepper_factory: StepperFactory | None = None,
    ) -> None:
        """Initialize corrector backend with step control factory.

        Parameters
        ----------
        stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` or None
            The stepper factory to use for step control. If None, uses plain
            Newton steps with optional capping.
        """
        super().__init__()
        self._stepper_factory: StepperFactory = (
            make_plain_stepper() if stepper_factory is None else stepper_factory
        )

    def _compute_residual(self, x: np.ndarray, residual_fn: ResidualCallable) -> np.ndarray:
        """Compute residual vector R(x).

        Separated for easy overriding or acceleration (e.g., with numba).

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual.

        Returns
        -------
        np.ndarray
            Residual vector R(x).
        """
        return residual_fn(x)

    def _compute_norm(self, residual: np.ndarray, norm_fn: NormCallable) -> float:
        """Compute residual norm for convergence checking.

        Parameters
        ----------
        residual : np.ndarray
            Residual vector.
        norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn`
            Function to compute norm.

        Returns
        -------
        float
            Scalar norm value.
        """
        return norm_fn(residual)

    def _compute_jacobian(
        self,
        x: np.ndarray,
        residual_fn: ResidualCallable,
        jacobian_fn: JacobianCallable | None,
        fd_step: float,
    ) -> np.ndarray:
        """Compute Jacobian matrix J(x) = dR/dx.

        Uses analytical Jacobian if provided, otherwise computes central
        finite-difference approximation with O(h^2) accuracy.

        Parameters
        ----------
        x : np.ndarray
            Current parameter vector.
        residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
            Function to compute residual.
        jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` or None
            Analytical Jacobian function, if available.
        fd_step : float
            Step size for finite-difference approximation.

        Returns
        -------
        np.ndarray
            Jacobian matrix with shape (m, n) where m is residual size
            and n is parameter size.
        """
        if jacobian_fn is not None:
            return jacobian_fn(x)

        # Finite-difference approximation (central diff, O(h**2))
        n = x.size
        r0 = residual_fn(x)
        J = np.zeros((r0.size, n))
        for i in range(n):
            x_p = x.copy()
            x_m = x.copy()
            h_i = fd_step * max(1.0, abs(x[i]))
            x_p[i] += h_i
            x_m[i] -= h_i
            J[:, i] = (residual_fn(x_p) - residual_fn(x_m)) / (2.0 * h_i)
        return J

    def _solve_delta_dense(
        self, J: np.ndarray, r: np.ndarray, cond_threshold: float = 1e8
    ) -> np.ndarray:
        """Solve Newton linear system J * delta = -r using dense linear algebra.

        Handles ill-conditioned and rectangular systems automatically:

        - Applies Tikhonov regularization for ill-conditioned square systems
        - Uses least-squares for rectangular systems
        - Falls back to SVD for singular systems

        Parameters
        ----------
        J : np.ndarray
            Jacobian matrix.
        r : np.ndarray
            Residual vector.
        cond_threshold : float, default=1e8
            Condition number threshold for regularization.

        Returns
        -------
        np.ndarray
            Newton step vector delta.
        """
        try:
            cond_J = np.linalg.cond(J)
        except np.linalg.LinAlgError:
            cond_J = np.inf

        lambda_reg = 0.0

        if J.shape[0] == J.shape[1]:
            # Square system
            if np.isnan(cond_J) or cond_J > cond_threshold:
                lambda_reg = 1e-12
                J_reg = J + np.eye(J.shape[0]) * lambda_reg
            else:
                J_reg = J

            try:
                delta = np.linalg.solve(J_reg, -r)
            except np.linalg.LinAlgError:
                logger.warning("Jacobian singular; switching to SVD least-squares update")
                delta = np.linalg.lstsq(J_reg, -r, rcond=None)[0]
        else:
            # Rectangular system (over/under-determined)
            lambda_reg = (
                1e-12 if (np.isnan(cond_J) or cond_J > cond_threshold) else 0.0
            )
            JTJ = J.T @ J + lambda_reg * np.eye(J.shape[1])
            JTr = J.T @ r
            try:
                delta = np.linalg.solve(JTJ, -JTr)
            except np.linalg.LinAlgError:
                logger.warning("Normal equations singular; falling back to SVD lstsq")
                delta = np.linalg.lstsq(J, -r, rcond=None)[0]

        return delta

    @abstractmethod
    def run(
        self,
        *,
        request: CorrectorInput,
        stepper_factory: StepperFactory | None = None,
    ) -> CorrectorOutput:
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
