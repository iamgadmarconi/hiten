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
:mod:`~hiten.algorithms.corrector.newton`
    Newton-Raphson correction implementations.
:mod:`~hiten.algorithms.corrector.interfaces`
    Interface classes for different correction strategies.
:mod:`~hiten.algorithms.corrector._step_interface`
    Step-size control interfaces for robust convergence.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from hiten.algorithms.corrector.types import JacobianFn, NormFn, ResidualFn


class _Corrector(ABC):
    """Define an abstract base class for iterative correction algorithms.

    This class defines the interface for iterative correction algorithms
    used throughout the hiten framework to solve nonlinear systems of
    equations. It provides a generic, domain-independent interface that
    can be specialized for different types of problems (periodic orbits,
    invariant manifolds, fixed points, etc.).

    The design follows the strategy pattern, separating the algorithmic
    aspects of correction (Newton-Raphson, quasi-Newton, etc.) from the
    domain-specific problem formulation. This enables:

    - **Code reuse**: Same algorithms work for different problem types
    - **Modularity**: Easy to swap different correction strategies
    - **Testing**: Algorithms can be tested independently of domain logic
    - **Flexibility**: Custom correction strategies can be implemented

    The corrector operates on abstract parameter vectors and residual
    functions, requiring domain-specific objects to provide thin wrapper
    interfaces that translate between their natural representation and
    the vector-based interface expected by the correction algorithms.

    Key Design Principles
    ---------------------
    1. **Domain Independence**: Works with any problem that can be
       expressed as finding zeros of a vector-valued function.
    2. **Algorithm Flexibility**: Supports different correction strategies
       through subclassing and configuration.
    3. **Robustness**: Includes safeguards and error handling for
       challenging numerical situations.
    4. **Performance**: Designed for efficient implementation with
       minimal overhead.

    Typical Usage Pattern
    ---------------------
    1. Domain object (e.g., periodic orbit) creates parameter vector
    2. Domain object provides residual function for constraint violations
    3. Corrector iteratively refines parameter vector to minimize residual
    4. Domain object reconstructs corrected state from final parameter vector

    Notes
    -----
    Subclasses must implement the :meth:`~hiten.algorithms.corrector.base._Corrector.correct` method and are expected
    to document any additional keyword arguments specific to their
    correction strategy (maximum iterations, tolerances, step control
    parameters, etc.).

    The abstract interface allows for different correction algorithms:
    - Newton-Raphson with various step control strategies
    - Quasi-Newton methods (BFGS, Broyden, etc.)
    - Trust region methods
    - Hybrid approaches combining multiple strategies

    Examples
    --------
    >>> # Typical usage pattern (conceptual)
    >>> class NewtonCorrector(_Corrector):
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
    :mod:`~hiten.algorithms.corrector.newton`
        Concrete Newton-Raphson implementations.
    :mod:`~hiten.algorithms.corrector._step_interface`
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
    def correct(
        self,
        x0: np.ndarray,
        residual_fn: ResidualFn,
        *,
        jacobian_fn: JacobianFn | None = None,
        norm_fn: NormFn | None = None,
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
        
        Notes
        -----
        Convergence Criteria:
        The algorithm terminates successfully when ||R(x)|| < tolerance,
        where the norm is computed using the provided norm_fn or a default
        choice. The tolerance should be chosen considering:
        - Required solution accuracy
        - Numerical conditioning of the problem
        - Computational cost constraints

        Robustness Considerations:
        Implementations should include safeguards for:
        - Step size control to prevent divergence
        - Detection and handling of singular Jacobians
        - Graceful degradation for poorly conditioned problems
        - Meaningful error reporting for debugging

        Performance Optimization:
        For computationally intensive problems, consider:
        - Reusing Jacobian evaluations when possible
        - Exploiting problem structure (sparsity, symmetry)
        - Adaptive tolerance and iteration limits
        - Warm starting from previous solutions

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