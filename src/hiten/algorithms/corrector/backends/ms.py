"""Multiple shooting backend with block-structured Newton-Raphson.

This module provides a dedicated backend for multiple shooting differential
correction. Unlike single shooting, this backend operates on an augmented
parameter space containing all patch initial states and enforces continuity
constraints through the residual function.

The multiple shooting method divides a trajectory into N segments (patches)
and treats each patch's initial state as an independent variable. Continuity
constraints ensure the trajectory is continuous across patch boundaries.

See Also
--------
:class:`~hiten.algorithms.corrector.backends.newton._NewtonBackend`
    Single shooting Newton backend.
:class:`~hiten.algorithms.corrector.backends.base._CorrectorBackend`
    Base class for all correction backends.
"""

from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import spsolve

from hiten.algorithms.corrector.backends.base import _CorrectorBackend
from hiten.algorithms.corrector.types import (
    CorrectorInput,
    CorrectorOutput,
    JacobianCallable,
    JacobianFn,
    NormCallable,
    NormFn,
    ResidualCallable,
    ResidualFn,
    StepperFactory,
)
from hiten.algorithms.types.exceptions import ConvergenceError
from hiten.utils.log_config import logger


class _MultipleShootingBackend(_CorrectorBackend):
    """Implement multiple shooting with Newton-Raphson iterations.

    This backend handles the multiple shooting algorithm with block-structured
    Jacobian assembly, robust linear algebra, and optional performance
    optimizations for large numbers of patches.

    The parameter vector has shape `(n_patches * n_control,)` containing
    the initial state of each shooting segment stacked sequentially:
    
    .. math::
        
        \\mathbf{x} = [\\mathbf{x}_0^T, \\mathbf{x}_1^T, \\ldots, \\mathbf{x}_{N-1}^T]^T

    The residual vector contains continuity errors at patch junctions and
    boundary conditions:
    
    .. math::
        
        \\mathbf{R}(\\mathbf{x}) = \\begin{bmatrix}
            \\mathbf{x}_1^- - \\mathbf{x}_1 \\\\
            \\mathbf{x}_2^- - \\mathbf{x}_2 \\\\
            \\vdots \\\\
            \\mathbf{x}_N^- - \\mathbf{x}_{\\text{target}}
        \\end{bmatrix}

    where :math:`\\mathbf{x}_i^-` is the state propagated from patch :math:`i-1`
    to time :math:`t_i`.

    Parameters
    ----------
    stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory`, optional
        Factory for creating step control strategies. If None, uses plain
        Newton steps with optional capping.
    use_sparse : bool, default=False
        Whether to use sparse linear algebra for Jacobian solve. Recommended
        for problems with `n_patches > 10` where the block tridiagonal
        structure provides significant efficiency gains.
    parallel_patches : bool, default=False
        Whether to propagate patches in parallel. Currently a placeholder
        for future feature; does not affect computation.

    Attributes
    ----------
    _stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory`
        Step control factory for creating steppers per problem.
    _use_sparse : bool
        Flag indicating sparse solver usage.
    _parallel_patches : bool
        Flag indicating parallel propagation (future feature).

    Notes
    -----
    The Jacobian has a block tridiagonal structure:

    .. math::

        \\mathbf{J} = \\begin{bmatrix}
            \\Phi_0 & -I & 0 & 0 \\\\
            0 & \\Phi_1 & -I & 0 \\\\
            0 & 0 & \\Phi_2 & -I \\\\
            \\partial BC & \\partial BC & \\partial BC & \\Phi_3
        \\end{bmatrix}

    where :math:`\\Phi_i` is the state transition matrix (STM) from patch
    :math:`i` to patch :math:`i+1`, and :math:`\\partial BC` represents
    the derivatives of boundary conditions with respect to each patch.

    This structure can be exploited for O(n) solving time instead of O(n³)
    when sparse solvers are enabled.

    Examples
    --------
    Basic usage with default settings:

    >>> backend = _MultipleShootingBackend()
    >>> x_patches = np.concatenate([patch_0, patch_1, patch_2])
    >>> x_corr, iterations, residual_norm = backend.run(
    ...     x0=x_patches,
    ...     residual_fn=continuity_residual_fn,
    ...     jacobian_fn=block_jacobian_fn,
    ...     tol=1e-10
    ... )

    Using sparse linear algebra for large problems:

    >>> backend = _MultipleShootingBackend(use_sparse=True)
    >>> x_corr, iters, rnorm = backend.run(
    ...     x0=many_patches,
    ...     residual_fn=residual_fn,
    ...     jacobian_fn=jacobian_fn
    ... )

    With custom step control (e.g., Armijo line search):

    >>> from hiten.algorithms.corrector.stepping import make_armijo_stepper
    >>> from hiten.algorithms.corrector.config import _LineSearchConfig
    >>> 
    >>> stepper = make_armijo_stepper(_LineSearchConfig())
    >>> backend = _MultipleShootingBackend(stepper_factory=stepper)
    >>> x_corr, iters, rnorm = backend.run(x0=x0, residual_fn=rfn, jacobian_fn=jfn)

    See Also
    --------
    :class:`~hiten.algorithms.corrector.interfaces_ms._MultipleShootingOrbitCorrectionInterface`
        Interface that constructs residual and Jacobian functions.
    :mod:`~hiten.algorithms.corrector.stepping`
        Step control strategies for robust convergence.
    """

    def __init__(
        self,
        *,
        stepper_factory: StepperFactory | None = None,
        use_sparse: bool = False,
        parallel_patches: bool = False,
    ) -> None:
        super().__init__(stepper_factory=stepper_factory)
        self._use_sparse = use_sparse
        self._parallel_patches = parallel_patches

    def _solve_delta(
        self, J: np.ndarray, r: np.ndarray, cond_threshold: float = 1e8
    ) -> np.ndarray:
        """Solve Newton linear system J * delta = -r.

        Handles block-structured Jacobians with optional sparse solving
        and Tikhonov regularization for ill-conditioned systems. Chooses
        between dense and sparse solvers based on backend configuration.

        Parameters
        ----------
        J : np.ndarray
            Block-structured Jacobian matrix.
        r : np.ndarray
            Residual vector.
        cond_threshold : float, default=1e8
            Condition number threshold for applying Tikhonov regularization.

        Returns
        -------
        np.ndarray
            Newton step vector delta satisfying J * delta ≈ -r.

        Notes
        -----
        For large numbers of patches, the block tridiagonal structure
        can be exploited for O(n) solving time instead of O(n³). This
        is enabled when `use_sparse=True` in the constructor.

        See Also
        --------
        _solve_delta_dense : Dense solver with regularization (inherited from base).
        _solve_delta_sparse : Sparse solver exploiting structure.
        """
        if self._use_sparse:
            return self._solve_delta_sparse(J, r, cond_threshold)
        else:
            return self._solve_delta_dense(J, r, cond_threshold)

    def _solve_delta_sparse(
        self, J: np.ndarray, r: np.ndarray, cond_threshold: float
    ) -> np.ndarray:
        """Solve using sparse linear algebra.

        Exploits the block tridiagonal structure of the multiple shooting
        Jacobian for more efficient solving. Especially beneficial for
        large numbers of patches (n_patches > 10) where the Jacobian
        becomes increasingly sparse.

        Parameters
        ----------
        J : np.ndarray
            Block tridiagonal Jacobian (provided as dense array).
        r : np.ndarray
            Residual vector.
        cond_threshold : float
            Condition number threshold (used for fallback decisions).

        Returns
        -------
        np.ndarray
            Newton step vector.

        Warnings
        --------
        Requires scipy to be installed. Falls back to dense solver if
        scipy is not available or if sparse solve fails.

        Notes
        -----
        This implementation converts the dense Jacobian to sparse format
        and uses scipy.sparse.linalg solvers. For truly large problems,
        the Jacobian should ideally be assembled directly in sparse format
        to avoid the conversion overhead.

        Future implementations may:
        - Accept Jacobian in sparse format directly
        - Use block Thomas algorithm for tridiagonal systems
        - Exploit structure-specific solvers
        """
        # Convert to sparse format if not already sparse (CSR for efficient ops)
        if issparse(J):
            # Already sparse, ensure CSR format
            J_sparse = J if J.format == 'csr' else J.tocsr()
        else:
            # Dense matrix, convert to sparse
            J_sparse = csr_matrix(J)

        try:
            if J.shape[0] == J.shape[1]:
                # Square system: direct sparse solve
                try:
                    delta = spsolve(J_sparse, -r)
                except Exception:
                    # Add regularization and retry
                    J_reg = J + 1e-12 * np.eye(J.shape[0])
                    J_reg_sparse = csr_matrix(J_reg)
                    delta = spsolve(J_reg_sparse, -r)
            else:
                # Rectangular system: solve normal equations
                JT_sparse = J_sparse.T
                JTJ = JT_sparse @ J_sparse
                JTr = JT_sparse @ (-r)
                delta = spsolve(JTJ, JTr)

            # Ensure proper array format
            if hasattr(delta, "toarray"):
                delta = delta.toarray().flatten()
            else:
                delta = np.asarray(delta).flatten()

        except Exception as e:
            logger.warning(f"Sparse solve failed ({e}); falling back to dense")
            return self._solve_delta_dense(J, r, cond_threshold)

        return delta

    def run(
        self,
        *,
        request: CorrectorInput,
        stepper_factory: StepperFactory | None = None,
    ) -> CorrectorOutput:
        """Solve multiple shooting system using Newton-Raphson method.

        Iteratively refines the patch initial states until all continuity
        constraints and boundary conditions are satisfied to within the
        specified tolerance.

        Parameters
        ----------
        request : :class:`~hiten.algorithms.corrector.types.CorrectorInput`
            Structured request containing initial guess, residual/Jacobian functions,
            and solver parameters.
        stepper_factory : :class:`~hiten.algorithms.corrector.types.StepperFactory` or None, optional
            Step control factory. Overrides backend default if provided.
            Use for custom line search or trust region strategies.

        Returns
        -------
        :class:`~hiten.algorithms.corrector.types.CorrectorOutput`
            Structured response containing converged solution, iteration count,
            final residual norm, and metadata.

        Raises
        ------
        ConvergenceError
            If Newton method fails to converge within max_attempts or
            encounters numerical difficulties.

        Notes
        -----
        The algorithm follows these steps in each iteration:

        1. Evaluate residual R(x) and its norm
        2. Check convergence: ||R|| < tol
        3. Compute Jacobian J = dR/dx (analytical or finite-diff)
        4. Solve linear system: J * delta = -R
        5. Apply step control: x_new = x + alpha * delta
        6. Repeat until convergence or max_attempts

        The parameter vector x contains all patch initial states:

        .. math::

            \\mathbf{x} = [\\mathbf{x}_0, \\mathbf{x}_1, \\ldots, \\mathbf{x}_{N-1}]

        where each :math:`\\mathbf{x}_i` contains only the control indices
        (components allowed to vary during correction).

        Examples
        --------
        Basic usage with typed request:

        >>> from hiten.algorithms.corrector.types import CorrectorInput
        >>> backend = _MultipleShootingBackend()
        >>> request = CorrectorInput(
        ...     initial_guess=x_patches,
        ...     residual_fn=my_residual_fn,
        ...     jacobian_fn=my_jacobian_fn,
        ...     norm_fn=None,
        ...     tol=1e-10,
        ...     max_attempts=25,
        ...     max_delta=1e-2,
        ...     fd_step=1e-8
        ... )
        >>> result = backend.run(request=request)
        >>> print(f"Converged in {result.iterations} iterations")

        See Also
        --------
        :class:`~hiten.algorithms.corrector.backends.newton._NewtonBackend`
            Single shooting Newton backend with similar interface.
        :class:`~hiten.algorithms.corrector.interfaces_ms._MultipleShootingOrbitCorrectionInterface`
            Interface that constructs residual and Jacobian functions.
        """
        # Extract parameters from request
        tol = request.tol
        max_attempts = request.max_attempts
        fd_step = request.fd_step
        max_delta = request.max_delta
        residual_fn = request.residual_fn
        jacobian_fn = request.jacobian_fn
        norm_fn = request.norm_fn

        if norm_fn is None:
            norm_callable: NormCallable = lambda r: float(np.linalg.norm(r))
        else:
            norm_callable = norm_fn

        x = request.initial_guess.copy()

        # Get stepper from factory
        factory = (
            self._stepper_factory if stepper_factory is None else stepper_factory
        )
        stepper = factory(residual_fn, norm_callable, max_delta)

        logger.info(
            f"Starting multiple shooting Newton iteration "
            f"(n_params={x.size}, tol={tol:.2e}, max_iter={max_attempts})"
        )

        for k in range(max_attempts):
            # Compute residual and norm
            r = self._compute_residual(x, residual_fn)
            r_norm = self._compute_norm(r, norm_callable)

            # Callback for iteration monitoring (optional hook)
            try:
                self.on_iteration(k, x, r_norm)
            except Exception:
                pass

            # Check convergence
            if r_norm < tol:
                logger.info(
                    f"Multiple shooting converged after {k} iterations "
                    f"(|R|={r_norm:.2e})"
                )
                try:
                    self.on_accept(x, iterations=k, residual_norm=r_norm)
                except Exception:
                    pass
                return CorrectorOutput(
                    x_corrected=x,
                    iterations=k,
                    residual_norm=r_norm,
                    metadata={},
                )

            # Compute Jacobian
            J = self._compute_jacobian(x, residual_fn, jacobian_fn, fd_step)

            # Log Jacobian properties on first iteration
            if k == 0:
                if issparse(J):
                    # Sparse matrix: use .nnz for nonzero count
                    total_elements = J.shape[0] * J.shape[1]
                    sparsity = 1.0 - J.nnz / total_elements
                else:
                    # Dense matrix: use original approach
                    sparsity = 1.0 - np.count_nonzero(J) / J.size

            # Solve for Newton step
            delta = self._solve_delta(J, r)

            # Apply step control (line search, trust region, or plain step)
            try:
                x_new, r_norm_new, alpha_used = stepper(x, delta, r_norm)
            except Exception as exc:
                raise ConvergenceError(
                    f"Step strategy failed at iteration {k}: {exc}"
                ) from exc

            logger.debug(
                f"Iteration {k}: |R|={r_norm:.2e} → {r_norm_new:.2e} "
                f"(alpha={alpha_used:.3f}, |δ|={np.linalg.norm(delta, np.inf):.2e})"
            )

            x = x_new

        # Maximum iterations reached - check final state
        r_final = self._compute_residual(x, residual_fn)
        r_final_norm = self._compute_norm(r_final, norm_callable)

        if r_final_norm < tol:
            # Converged on final check
            logger.info(
                f"Multiple shooting converged on final check after {max_attempts} "
                f"iterations (|R|={r_final_norm:.2e})"
            )
            self.on_accept(x, iterations=max_attempts, residual_norm=r_final_norm)
            return CorrectorOutput(
                x_corrected=x,
                iterations=max_attempts,
                residual_norm=r_final_norm,
                metadata={},
            )

        # Failed to converge
        self.on_failure(x, iterations=max_attempts, residual_norm=r_final_norm)

        raise ConvergenceError(
            f"Multiple shooting did not converge after {max_attempts} iterations "
            f"(|R|={r_final_norm:.2e}). Try: increasing max_attempts, improving "
            f"initial guess, or increasing number of patches."
        ) from None
