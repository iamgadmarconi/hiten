"""Base classes and configuration for iterative correction algorithms.

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
:mod:`hiten.algorithms.corrector.newton`
    Newton-Raphson correction implementations.
:mod:`hiten.algorithms.corrector.interfaces`
    Interface classes for different correction strategies.
:mod:`hiten.algorithms.corrector._step_interface`
    Step-size control interfaces for robust convergence.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np

from hiten.algorithms.corrector.line import _LineSearchConfig

#: Type alias for residual function signatures.
#:
#: Functions of this type compute residual vectors from parameter vectors,
#: representing the nonlinear equations to be solved. The residual should
#: approach zero as the parameter vector approaches the solution.
#:
#: In dynamical systems contexts, the residual typically represents:
#: - Constraint violations for periodic orbits
#: - Boundary condition errors for invariant manifolds
#: - Fixed point equations for equilibrium solutions
#:
#: Parameters
#: ----------
#: x : ndarray
#:     Parameter vector at which to evaluate the residual.
#:
#: Returns
#: -------
#: residual : ndarray
#:     Residual vector of the same shape as the input.
#:
#: Notes
#: -----
#: The residual function should be well-defined and continuous in
#: the neighborhood of the expected solution. For best convergence
#: properties, it should also be differentiable with a non-singular
#: Jacobian at the solution.
ResidualFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for Jacobian function signatures.
#:
#: Functions of this type compute Jacobian matrices (first derivatives)
#: of residual functions with respect to parameter vectors. The Jacobian
#: is essential for Newton-type methods and provides information about
#: the local linearization of the nonlinear system.
#:
#: Parameters
#: ----------
#: x : ndarray
#:     Parameter vector at which to evaluate the Jacobian.
#:
#: Returns
#: -------
#: jacobian : ndarray
#:     Jacobian matrix with shape (n, n) where n is the length of x.
#:     Element (i, j) contains the partial derivative of residual[i]
#:     with respect to x[j].
#:
#: Notes
#: -----
#: For Newton methods to converge quadratically, the Jacobian should
#: be continuous and non-singular in a neighborhood of the solution.
#: When analytic Jacobians are not available, finite-difference
#: approximations can be used at the cost of reduced convergence rate.
JacobianFn = Callable[[np.ndarray], np.ndarray]

#: Type alias for norm function signatures.
#:
#: Functions of this type compute scalar norms from vectors, providing
#: a measure of vector magnitude used for convergence assessment and
#: step-size control. The choice of norm can affect convergence behavior
#: and numerical stability.
#:
#: Parameters
#: ----------
#: vector : ndarray
#:     Vector to compute the norm of.
#:
#: Returns
#: -------
#: norm : float
#:     Scalar norm value (non-negative).
#:
#: Notes
#: -----
#: Common choices include:
#: - L2 norm (Euclidean): Good general-purpose choice
#: - Infinity norm: Emphasizes largest component
#: - Weighted norms: Account for different scales in components
#:
#: The norm should be consistent across all uses within a single
#: correction process to ensure proper convergence assessment.
NormFn = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class _BaseCorrectionConfig:
    """Base configuration class for correction algorithm parameters.

    This dataclass encapsulates the common configuration parameters used
    by correction algorithms throughout the hiten framework. It provides
    sensible defaults while allowing customization for specific problem
    requirements and numerical considerations.

    The configuration is designed to be immutable (frozen) for thread safety
    and to prevent accidental modification during algorithm execution. The
    slots optimization reduces memory overhead when many configuration
    objects are created.

    Parameters
    ----------
    max_attempts : int, default=50
        Maximum number of Newton iterations to attempt before declaring
        convergence failure. This prevents infinite loops in cases where
        the algorithm fails to converge.
    tol : float, default=1e-10
        Convergence tolerance for the residual norm. The algorithm terminates
        successfully when the norm of the residual falls below this value.
        Should be chosen based on the required precision and numerical
        conditioning of the problem.
    max_delta : float, default=1e-2
        Maximum allowed infinity norm of Newton steps. This serves as a
        safeguard against excessively large steps that could cause numerical
        overflow or move far from the solution. Particularly important for
        poorly conditioned problems or bad initial guesses.
    line_search_config : :class:`hiten.algorithms.corrector.line._LineSearchConfig`, bool, or None, default=True
        Configuration for line search behavior:
        - True: Enable line search with default parameters
        - False or None: Disable line search (use full Newton steps)
        - :class:`hiten.algorithms.corrector.line._LineSearchConfig`: Enable line search with custom parameters
        Line search improves robustness for challenging problems at the
        cost of additional function evaluations.
    finite_difference : bool, default=False
        Force finite-difference approximation of Jacobians even when
        analytic Jacobians are available. Useful for debugging, testing,
        or when analytic Jacobians are suspected to be incorrect.
        Generally results in slower convergence but can be more robust.

    Notes
    -----
    The default parameters are chosen to work well for typical problems
    in astrodynamics and dynamical systems, particularly in the context
    of the Circular Restricted Three-Body Problem (CR3BP).

    Parameter Selection Guidelines:
    
    - **Tolerance**: Should be 2-3 orders of magnitude larger than
      machine epsilon to account for numerical errors in function
      evaluation and Jacobian computation.
    - **Max attempts**: Should be large enough to allow convergence
      from reasonable initial guesses but not so large as to waste
      computation on hopeless cases.
    - **Max delta**: Should be scaled appropriately for the problem's
      characteristic length scales to prevent numerical instability.
    - **Line search**: Generally recommended for production use,
      especially when initial guesses may be poor.

    Examples
    --------
    >>> # Default configuration
    >>> config = _BaseCorrectionConfig()
    >>>
    >>> # High-precision configuration
    >>> config = _BaseCorrectionConfig(tol=1e-12, max_attempts=100)
    >>>
    >>> # Robust configuration with custom line search
    >>> from hiten.algorithms.corrector.line import _LineSearchConfig
    >>> ls_config = _LineSearchConfig(armijo_c=1e-4, alpha_reduction=0.5)
    >>> config = _BaseCorrectionConfig(
    ...     line_search_config=ls_config,
    ...     max_delta=1e-3
    ... )
    >>>
    >>> # Fast configuration without line search
    >>> config = _BaseCorrectionConfig(
    ...     line_search_config=False,
    ...     tol=1e-8,
    ...     max_attempts=20
    ... )

    See Also
    --------
    :class:`hiten.algorithms.corrector.line._LineSearchConfig`
        Configuration class for line search parameters.
    :class:`_Corrector`
        Abstract base class that uses this configuration.
    """
    max_attempts: int = 50
    tol: float = 1e-10

    max_delta: float = 1e-2

    line_search_config: _LineSearchConfig | bool | None = True
    """Line search configuration.
    
    Controls the line search behavior for step-size control:
    - True: Use default line search parameters for robust convergence
    - False or None: Disable line search, use full Newton steps
    - :class:`hiten.algorithms.corrector.line._LineSearchConfig`: Use custom line search parameters
    
    Line search is generally recommended for production use as it
    significantly improves convergence robustness, especially for
    problems with poor initial guesses or ill-conditioning.
    """

    finite_difference: bool = False
    """Force finite-difference Jacobian approximation.
    
    When True, forces the use of finite-difference approximation for
    Jacobians even when analytic Jacobians are available. This can be
    useful for:
    - Debugging analytic Jacobian implementations
    - Testing convergence behavior with different Jacobian sources
    - Working around bugs in analytic Jacobian code
    
    Generally results in slower convergence but may be more robust
    in some cases. The finite-difference step size is chosen automatically
    based on the problem scaling and machine precision.
    """


class _Corrector(ABC):
    """Abstract base class for iterative correction algorithms.

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
    Subclasses must implement the :meth:`correct` method and are expected
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
    :class:`_BaseCorrectionConfig`
        Configuration class for correction parameters.
    :mod:`hiten.algorithms.corrector.newton`
        Concrete Newton-Raphson implementations.
    :mod:`hiten.algorithms.corrector._step_interface`
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
        residual_fn : ResidualFn
            Function computing the residual vector R(x) for parameter
            vector x. The residual should be zero (or close to zero) at
            the desired solution. Must be well-defined and preferably
            continuous in a neighborhood of the solution.
        jacobian_fn : JacobianFn, optional
            Function returning the Jacobian matrix J(x) = dR/dx. If not
            provided, implementations may use finite-difference approximation
            or other Jacobian-free methods. Analytic Jacobians generally
            provide better convergence properties.
        norm_fn : NormFn, optional
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