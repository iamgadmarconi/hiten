"""Runtime options for linear algebra algorithms.

These classes define runtime tuning parameters that control HOW WELL the
linear algebra algorithms perform. They can vary between method calls without
changing the algorithm structure.

For compile-time configuration (problem structure), see config.py.
"""

from dataclasses import dataclass

from hiten.algorithms.types.options import _HitenBaseOptions


@dataclass(frozen=True)
class EigenDecompositionOptions(_HitenBaseOptions):
    """Runtime options for eigenvalue decomposition.
    
    These parameters tune HOW WELL the eigenvalue decomposition and
    classification performs. They can vary between method calls without
    changing the problem structure.
    
    Parameters
    ----------
    delta : float, default=1e-6
        Half-width of neutral band around stability threshold for eigenvalue
        classification. This is a runtime parameter because it tunes the
        sensitivity of classification without changing WHAT is being classified.
        
        - For continuous systems: |Re(lambda)| < delta → center
        - For discrete systems: ||lambda| - 1| < delta → center
        
        Larger values make classification more conservative (larger center
        subspace), smaller values make it stricter. Typical range: 1e-4 to 1e-8.
    tol : float, default=1e-8
        Tolerance for stability index calculation and reciprocal eigenvalue
        pairing in symplectic systems. This is a runtime parameter because it
        tunes numerical accuracy without changing the algorithm structure.
        
        Used for:
        - Detecting reciprocal eigenvalue pairs (lambda, 1/lambda)
        - Identifying unit-magnitude eigenvalues
        - Numerical cleanup of eigenvector components
        
        Smaller values are stricter but may fail on poorly conditioned systems.
        Typical range: 1e-6 to 1e-12.

    Notes
    -----
    These are runtime options because they control HOW WELL the algorithm
    performs (classification sensitivity, numerical accuracy) rather than
    WHAT problem is being solved (continuous vs discrete, which outputs to
    compute).

    The tolerances affect:
    
    - Classification robustness vs precision tradeoff
    - Handling of near-marginal eigenvalues
    - Numerical stability of eigenvector computations
    - Reciprocal pairing in symplectic systems

    Typical values for different use cases:
    
    - Standard analysis: delta=1e-6, tol=1e-8
    - High precision: delta=1e-8, tol=1e-12
    - Robust/forgiving: delta=1e-4, tol=1e-6

    Examples
    --------
    >>> # Default runtime options
    >>> options = EigenDecompositionOptions()
    >>> 
    >>> # High precision analysis
    >>> precise_options = EigenDecompositionOptions(
    ...     delta=1e-8,
    ...     tol=1e-12
    ... )
    >>> 
    >>> # Robust analysis for poorly conditioned systems
    >>> robust_options = EigenDecompositionOptions(
    ...     delta=1e-4,
    ...     tol=1e-6
    ... )

    See Also
    --------
    :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
        Compile-time configuration for problem structure.
    :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
        Main class that uses these options.
    """
    delta: float = 1e-6
    tol: float = 1e-8

    def _validate(self) -> None:
        """Validate the options."""
        if self.delta <= 0:
            raise ValueError("delta must be positive.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.delta > 1.0:
            raise ValueError(
                "delta should be small (typically < 1.0). "
                f"Got delta={self.delta}."
            )

