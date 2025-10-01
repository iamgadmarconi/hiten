"""
Types for the corrector module.

This module provides the types for the corrector module.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.corrector.protocols import CorrectorStepProtocol

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

StepperFactory = Callable[[ResidualFn, NormFn, float | None], "CorrectorStepProtocol"]


@dataclass
class CorrectionResult:
    """Standardized result for a backend correction run.
    
    Attributes
    ----------
    converged : bool
        Whether the correction converged.
    x_corrected : ndarray
        Corrected parameter vector.
    residual_norm : float
        Final residual norm.
    iterations : int
        Number of iterations performed.
    """
    converged: bool
    x_corrected: np.ndarray
    residual_norm: float
    iterations: int


@dataclass
class OrbitCorrectionResult(CorrectionResult):
    """Result for an orbit correction run.
    
    Attributes
    ----------
    half_period : float
        Half-period associated with the corrected orbit.
    """
    half_period: float


@dataclass
class _CorrectionProblem:
    """Defines the inputs for a backend correction run.

    Attributes
    ----------
    initial_guess : ndarray
        Initial parameter vector.
    residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
        Residual function R(x).
    jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` | None
        Optional analytical Jacobian.
    norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` | None
        Optional norm function for convergence checks.
    max_attempts : int
        Maximum number of Newton iterations to attempt.
    tol : float
        Convergence tolerance for the residual norm.
    max_delta : float
        Maximum allowed infinity norm of Newton steps.
    finite_difference : bool
        Force finite-difference approximation of Jacobians.
    fd_step : float
        Finite-difference step size.
    method : str
        Integration method for trajectory computation.
    order : int
        Integration order for numerical methods.
    steps : int
        Number of integration steps.
    forward : int
        Integration direction (1 for forward, -1 for backward).
    stepper_factory : callable or None
        Optional factory producing a stepper compatible with the backend.
    """
    initial_guess: np.ndarray
    residual_fn: ResidualFn
    jacobian_fn: Optional[JacobianFn]
    norm_fn: Optional[NormFn]
    max_attempts: int
    tol: float
    max_delta: float
    finite_difference: bool
    fd_step: float
    method: str
    order: int
    steps: int
    forward: int
    stepper_factory: Optional[StepperFactory]


@dataclass
class _OrbitCorrectionProblem(_CorrectionProblem):
    """Defines the inputs for a backend orbit correction run.
    
    Attributes
    ----------
    domain_obj: Any
        Orbit to be corrected.
    residual_indices : tuple of int
        State components used to build the residual vector.
    control_indices : tuple of int
        State components allowed to change during correction.
    extra_jacobian : callable or None
        Additional Jacobian contribution function.
    target : tuple of float
        Target values for the residual components.
    event_func : callable
        Function to detect Poincare section crossings.
    """
    domain_obj: Any
    residual_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    extra_jacobian: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    target: tuple[float, ...]
    event_func: Callable[..., tuple[float, np.ndarray]]


@dataclass
class MultipleShootingResult(CorrectionResult):
    """Result for a multiple shooting correction run.

    Extends the base :class:`CorrectionResult` with multiple shooting-specific
    information including patch states and continuity diagnostics.

    Attributes
    ----------
    converged : bool
        Whether the correction converged successfully.
    x_corrected : np.ndarray
        Final corrected state vector (full state at initial patch).
    residual_norm : float
        Final residual norm achieved (includes continuity + boundary errors).
    iterations : int
        Number of Newton iterations performed.
    n_patches : int
        Number of shooting segments used in the correction.
    patch_states : list[np.ndarray]
        Converged state vectors at each patch point.
        Length: n_patches, each element is full state vector.
    patch_times : np.ndarray
        Time values at patch boundaries [t₀, t₁, ..., tₙ].
        Length: n_patches + 1
    continuity_errors : np.ndarray
        Continuity errors at each patch junction after convergence.
        Length: (n_patches - 1) * n_continuity
        Should be near zero for successful correction.
    half_period : float or None
        Half-period of the corrected orbit (for periodic orbits).
        None for non-periodic problems.

    Notes
    -----
    The `patch_states` contain the initial state at each shooting segment.
    These should be nearly continuous when `continuity_errors` is small.

    For a periodic orbit with n_patches=3:
    - patch_states[0]: Initial state
    - patch_states[1]: State at first patch junction
    - patch_states[2]: State at second patch junction
    - Integration from patch_states[2] reaches final boundary

    Examples
    --------
    >>> result = MultipleShootingResult(
    ...     converged=True,
    ...     x_corrected=final_state,
    ...     residual_norm=1.2e-11,
    ...     iterations=8,
    ...     n_patches=5,
    ...     patch_states=[state0, state1, state2, state3, state4],
    ...     patch_times=np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]),
    ...     continuity_errors=np.array([...]),
    ...     half_period=2.5
    ... )
    >>> print(f"Converged in {result.iterations} iters with {result.n_patches} patches")
    >>> print(f"Max continuity error: {np.max(np.abs(result.continuity_errors)):.2e}")

    See Also
    --------
    :class:`CorrectionResult`
        Base class for all correction results.
    :class:`OrbitCorrectionResult`
        Single shooting orbit correction result.
    """
    n_patches: int
    patch_states: list[np.ndarray]
    patch_times: np.ndarray
    continuity_errors: np.ndarray
    half_period: float | None = None


@dataclass
class _MultipleShootingProblem(_CorrectionProblem):
    """Defines the inputs for a multiple shooting correction run.

    Extends :class:`_CorrectionProblem` with multiple shooting-specific
    parameters including patch configuration, continuity indices, and
    boundary condition specifications.

    Attributes
    ----------
    initial_guess : np.ndarray
        Initial parameter vector containing all patch states stacked.
        Shape: (n_patches * n_control,)
    residual_fn : :class:`~hiten.algorithms.corrector.types.ResidualFn`
        Function computing continuity and boundary residuals.
    jacobian_fn : :class:`~hiten.algorithms.corrector.types.JacobianFn` | None
        Optional analytical block-structured Jacobian.
    norm_fn : :class:`~hiten.algorithms.corrector.types.NormFn` | None
        Optional norm function for convergence checks.
    max_attempts : int
        Maximum number of Newton iterations.
    tol : float
        Convergence tolerance for residual norm.
    max_delta : float
        Maximum allowed infinity norm of Newton steps.
    finite_difference : bool
        Whether to use finite-difference Jacobian approximation.
    fd_step : float
        Finite-difference step size.
    method : str
        Integration method for patch propagation.
    order : int
        Integration order for numerical methods.
    steps : int
        Number of integration steps per patch.
    forward : int
        Integration direction (1 for forward, -1 for backward).
    stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` | None
        Optional factory for step control strategies.
    domain_obj : Any
        The domain object being corrected (e.g., PeriodicOrbit).
    n_patches : int
        Number of shooting segments.
    patch_times : np.ndarray
        Time values at patch boundaries [t₀, t₁, ..., tₙ].
        Length: n_patches + 1
    patch_strategy : str
        Strategy used for patch placement ("uniform", "adaptive", "manual").
    residual_indices : tuple[int, ...]
        State components with boundary conditions.
    control_indices : tuple[int, ...]
        State components allowed to vary during correction.
    continuity_indices : tuple[int, ...]
        State components enforced continuous at patch junctions.
        If empty, uses control_indices.
    boundary_indices : tuple[int, ...]
        State components with boundary conditions (typically same as residual_indices).
    target : tuple[float, ...]
        Target values for boundary conditions.
    event_func : callable or None
        Optional event function for final boundary detection.

    Notes
    -----
    The parameter vector in `initial_guess` has the structure:

    .. math::

        \\mathbf{x} = [\\mathbf{x}_0^T, \\mathbf{x}_1^T, \\ldots, \\mathbf{x}_{N-1}^T]^T

    where each :math:`\\mathbf{x}_i` contains the control indices of the state
    at patch :math:`i`.

    The residual function should return:

    .. math::

        \\mathbf{R}(\\mathbf{x}) = \\begin{bmatrix}
            \\mathbf{x}_1^- - \\mathbf{x}_1 \\\\
            \\mathbf{x}_2^- - \\mathbf{x}_2 \\\\
            \\vdots \\\\
            \\mathbf{x}_N^- - \\mathbf{x}_{\\text{target}}
        \\end{bmatrix}

    where :math:`\\mathbf{x}_i^-` is the state propagated from patch :math:`i-1`
    to time :math:`t_i`.

    Examples
    --------
    >>> problem = _MultipleShootingProblem(
    ...     initial_guess=patch_params,
    ...     residual_fn=continuity_residual,
    ...     jacobian_fn=block_jacobian,
    ...     norm_fn=lambda r: np.linalg.norm(r, np.inf),
    ...     max_attempts=50,
    ...     tol=1e-10,
    ...     max_delta=1e-2,
    ...     n_patches=5,
    ...     patch_times=np.linspace(0, 2.5, 6),
    ...     patch_strategy="uniform",
    ...     control_indices=(0, 2, 4),
    ...     continuity_indices=(0, 2, 4),
    ...     boundary_indices=(2, 3, 4),
    ...     target=(0.0, 0.0, 0.0)
    ... )

    See Also
    --------
    :class:`_CorrectionProblem`
        Base class for all correction problems.
    :class:`_OrbitCorrectionProblem`
        Single shooting orbit correction problem.
    :class:`MultipleShootingResult`
        Result type for multiple shooting corrections.
    """
    domain_obj: Any
    n_patches: int
    patch_times: np.ndarray
    patch_strategy: str
    residual_indices: tuple[int, ...]
    control_indices: tuple[int, ...]
    continuity_indices: tuple[int, ...]
    boundary_indices: tuple[int, ...]
    target: tuple[float, ...]
    event_func: Callable[..., tuple[float, np.ndarray]] | None = None
