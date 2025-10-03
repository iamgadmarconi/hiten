"""Provide linear algebra utilities for dynamical systems analysis.

This module provides specialized linear algebra routines for analyzing
dynamical systems, particularly in the context of the Circular Restricted
Three-Body Problem (CR3BP). The functions focus on eigenvalue analysis,
stability classification, and matrix decompositions relevant to periodic
orbits and manifold computations.

References
----------
.. [Koon2011] Koon, W. S., Lo, M. W., Marsden, J. E., Ross, S. D. (2011).
   *Dynamical Systems, the Three-Body Problem and Space Mission Design*.
   Springer.
"""
from typing import Any, Dict, List, Tuple

import numpy as np

from hiten.algorithms.linalg.types import (
    EigenDecompositionResults,
    LinalgBackendRequest,
    LinalgBackendResponse,
    _ProblemType,
    _StabilityType,
    _SystemType,
)
from hiten.algorithms.types.core import _HitenBaseBackend
from hiten.algorithms.types.exceptions import BackendError


class _LinalgBackend(_HitenBaseBackend):
    """Minimal backend for linear algebra operations.

    Stateless wrapper exposing the core linear-algebra routines used across
    the library. Provides a small object interface while preserving the
    existing module-level API for backward compatibility.

    Parameters
    ----------
    system_type : :class:`~hiten.algorithms.linalg.types._SystemType`, optional
        The type of system to analyze. Default is 
        :class:`~hiten.algorithms.linalg.types._SystemType.CONTINUOUS`.
    """

    def __init__(self, system_type: _SystemType = _SystemType.CONTINUOUS):
        self.system_type = system_type

    def run(self, request: LinalgBackendRequest) -> LinalgBackendResponse:
        """Execute eigenvalue analysis according to the request."""
        self.system_type = request.system_type

        n = request.matrix.shape[0]
        empty_vals = np.array([], dtype=np.complex128)
        empty_vecs = np.zeros((n, 0), dtype=np.complex128)
        empty_complex = np.array([], dtype=np.complex128)

        results = EigenDecompositionResults(
            stable=empty_vals,
            unstable=empty_vals,
            center=empty_vals,
            Ws=empty_vecs,
            Wu=empty_vecs,
            Wc=empty_vecs,
            nu=empty_complex,
            eigvals=empty_complex,
            eigvecs=np.zeros((0, 0), dtype=np.complex128),
        )

        metadata: Dict[str, Any] = dict(request.metadata)

        if request.problem_type in (_ProblemType.EIGENVALUE_DECOMPOSITION, _ProblemType.ALL):
            sn, un, cn, Ws, Wu, Wc = self.eigenvalue_decomposition(
                request.matrix, request.delta
            )
            results = EigenDecompositionResults(
                stable=sn,
                unstable=un,
                center=cn,
                Ws=Ws,
                Wu=Wu,
                Wc=Wc,
                nu=results.nu,
                eigvals=results.eigvals,
                eigvecs=results.eigvecs,
            )

        if request.problem_type in (_ProblemType.STABILITY_INDICES, _ProblemType.ALL):
            nu, eigvals, eigvecs = self.stability_indices(request.matrix, request.tol)
            results = EigenDecompositionResults(
                stable=results.stable,
                unstable=results.unstable,
                center=results.center,
                Ws=results.Ws,
                Wu=results.Wu,
                Wc=results.Wc,
                nu=nu,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            metadata.update({"nu": nu, "eigvals": eigvals})

        return LinalgBackendResponse(results=results, metadata=metadata)

    def eigenvalue_decomposition(self, A: np.ndarray, delta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Classify eigenvalue-eigenvector pairs into stable, unstable, and center subspaces.

        Performs eigenvalue decomposition and classifies the spectrum based on
        stability criteria for either continuous-time or discrete-time dynamical
        systems. Each eigenvector is pivot-normalized for consistent representation.

        Parameters
        ----------
        A : numpy.ndarray, shape (n, n)
            Real or complex square matrix to analyze.
        delta : float, optional
            Half-width of neutral band around stability threshold. Default is 1e-4.
            For continuous systems: absolute value of real part of lambda < delta -> center
            For discrete systems: absolute value of (absolute value of lambda - 1) < delta -> center

        Returns
        -------
        sn : numpy.ndarray
            Stable eigenvalues. For continuous: real part of lambda < -delta.
            For discrete: absolute value of lambda < 1-delta.
        un : numpy.ndarray
            Unstable eigenvalues. For continuous: real part of lambda > +delta.
            For discrete: absolute value of lambda > 1+delta.
        cn : numpy.ndarray
            Center eigenvalues (neutral spectrum within delta band).
        Ws : numpy.ndarray, shape (n, n_s)
            Stable eigenvectors stacked column-wise.
        Wu : numpy.ndarray, shape (n, n_u)
            Unstable eigenvectors stacked column-wise.
        Wc : numpy.ndarray, shape (n, n_c)
            Center eigenvectors stacked column-wise.

        Raises
        ------
        numpy.linalg.LinAlgError
            If eigenvalue decomposition fails. Returns empty arrays in this case.

        Notes
        -----
        - Eigenvectors are pivot-normalized: first non-zero entry equals 1
        - Small imaginary parts (< 1e-14) are set to zero for numerical stability
        - Empty subspaces return zero-column matrices with correct dimensions
        
        Examples
        --------
        >>> import numpy as np
        >>> from hiten.algorithms.common.linalg import eigenvalue_decomposition
        >>> # Continuous-time system with stable, center, unstable eigenvalues
        >>> A = np.diag([-2.0, 0.0, 0.5])
        >>> sn, un, cn, Ws, Wu, Wc = eigenvalue_decomposition(A)
        >>> sn
        array([-2.])
        >>> un
        array([0.5])
        >>> cn
        array([0.])
        
        See Also
        --------
        :func:`~hiten.algorithms.common.linalg.eigenvalue_decomposition` :
            General eigenvalue classification
        """

        eigvals, eigvecs = self._compute_eigendecomposition(A)

        # Normalize and canonicalize via helper, preserving order
        val, vec = self._sort_eigenvalues(eigvals, eigvecs)

        results = [self._classify_eigenvalue(v, w, delta) for v, w in zip(val, vec.T)]

        # Flatten eigenvalue lists
        sn_vals = [ev for (_, s, _, _, _, _, _) in results for ev in s]
        un_vals = [ev for (_, _, u, _, _, _, _) in results for ev in u]
        cn_vals = [ev for (_, _, _, c, _, _, _) in results for ev in c]

        # Flatten eigenvector lists
        Ws_vecs = [w for (_, _, _, _, svecs, _, _) in results for w in svecs]
        Wu_vecs = [w for (_, _, _, _, _, uvecs, _) in results for w in uvecs]
        Wc_vecs = [w for (_, _, _, _, _, _, cvecs) in results for w in cvecs]

        sn = np.array(sn_vals, dtype=np.complex128)
        un = np.array(un_vals, dtype=np.complex128)
        cn = np.array(cn_vals, dtype=np.complex128)

        n = A.shape[0]
        Ws = np.column_stack(Ws_vecs) if len(Ws_vecs) > 0 else np.zeros((n, 0), dtype=np.complex128)
        Wu = np.column_stack(Wu_vecs) if len(Wu_vecs) > 0 else np.zeros((n, 0), dtype=np.complex128)
        Wc = np.column_stack(Wc_vecs) if len(Wc_vecs) > 0 else np.zeros((n, 0), dtype=np.complex128)

        return sn, un, cn, Ws, Wu, Wc

    def stability_indices(self, M: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Floquet stability indices for periodic orbit analysis.

        Calculates the three stability indices nu_i = (lambda_i + 1/lambda_i)/2
        from the monodromy matrix of a periodic orbit. For symplectic systems,
        eigenvalues occur in reciprocal pairs (lambda, 1/lambda), and this
        function explicitly searches for such pairs.

        Parameters
        ----------
        M : numpy.ndarray, shape (6, 6)
            Monodromy matrix from one-period state transition matrix integration.
            Expected to be symplectic for CR3BP applications.
        tol : float, optional
            Tolerance for reciprocal eigenvalue pairing and unit-magnitude
            detection. Default is 1e-8.

        Returns
        -------
        nu : numpy.ndarray, shape (3,)
            Stability indices nu_i = (lambda_i + 1/lambda_i)/2.
            Contains np.nan for unpaired eigenvalues.
        eigvals : numpy.ndarray, shape (6,)
            Eigenvalues sorted by decreasing magnitude.
        eigvecs : numpy.ndarray, shape (6, 6)
            Corresponding eigenvectors.

        Raises
        ------
        :class:`~hiten.algorithms.types.exceptions.BackendError`
            If M is not shape (6, 6).
        numpy.linalg.LinAlgError
            If eigenvalue computation fails. Returns NaN arrays in this case.

        Notes
        -----
        - Assumes symplectic structure with reciprocal eigenvalue pairs
        - Robust to small numerical symmetry-breaking errors
        - Identifies trivial pairs (magnitude near 1) first
        - Warns if expected number of pairs (3) cannot be found
        
        For stable periodic orbits, all |nu_i| should be <= 1.
        
        Examples
        --------
        >>> import numpy as np
        >>> from hiten.algorithms.linalg import _stability_indices
        >>> # Identity matrix (trivial case)
        >>> M = np.eye(6)
        >>> nu, eigvals, eigvecs = _stability_indices(M)
        >>> np.allclose(nu, 1.0)
        True
        
        See Also
        --------
        :func:`~hiten.algorithms.linalg.eigenvalue_decomposition` :
            General eigenvalue classification
        """
        self._validate_monodromy_shape(M)

        eigvals_sorted, eigvecs_sorted = self._compute_sorted_spectrum(M)

        nu = self._compute_nu_from_eigvals(eigvals_sorted, tol)

        return nu, eigvals_sorted, eigvecs_sorted

    def _compute_eigendecomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the eigenvalue decomposition of a matrix.
        
        Parameters
        ----------
        A : numpy.ndarray
            Matrix to decompose.

        Returns
        -------
        numpy.ndarray
            Eigenvalues.
        numpy.ndarray
            Eigenvectors.
        """
        try:
            eigvals, eigvecs = np.linalg.eig(A)
            return eigvals, eigvecs
        except np.linalg.LinAlgError:
            n = A.shape[0]
            empty_complex = np.array([], dtype=np.complex128)
            empty_matrix = np.zeros((n, 0), dtype=np.complex128)
            return empty_complex, empty_matrix

    def _classify_eigenvalue(self, val: complex, vec: np.ndarray, delta: float) -> _StabilityType:
        """Classify an eigenvalue into stable, unstable, or center.
        
        Parameters
        ----------
        val : complex
            Eigenvalue to classify.
        vec : numpy.ndarray
            Eigenvector corresponding to the eigenvalue.
        delta : float
            Half-width of neutral band around stability threshold.
        sn : list[complex]
            List of stable eigenvalues.
        un : List[complex]
            List of unstable eigenvalues.
        cn : List[complex]
            List of center eigenvalues.
        Ws_list : List[np.ndarray]
            List of stable eigenvectors.
        Wu_list : List[np.ndarray]
            List of unstable eigenvectors.
        Wc_list : List[np.ndarray]
            List of center eigenvectors.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.backend._StabilityType`
            Classification of the eigenvalue.
        """
        sn, un, cn = [], [], [] # stable, unstable, center eigenvalues
        Ws_list, Wu_list, Wc_list = [], [], [] # stable, unstable, center eigenvectors

        classification = _StabilityType.CENTER
        if self.system_type == _SystemType.DISCRETE:
            mag = abs(val)
            if mag < 1 - delta:
                classification = _StabilityType.STABLE
                sn.append(val)
                Ws_list.append(vec)
            elif mag > 1 + delta:
                classification = _StabilityType.UNSTABLE
                un.append(val)
                Wu_list.append(vec)
            else:
                classification = _StabilityType.CENTER
                cn.append(val)
                Wc_list.append(vec)

        elif self.system_type == _SystemType.CONTINUOUS:
            if val.real < -delta:
                classification = _StabilityType.STABLE
                sn.append(val)
                Ws_list.append(vec)
            elif val.real > +delta:
                classification = _StabilityType.UNSTABLE
                un.append(val)
                Wu_list.append(vec)
            else:
                classification = _StabilityType.CENTER
                cn.append(val)
                Wc_list.append(vec)

        else:
            raise BackendError("Invalid system type encountered in _classify_eigenvalue.")

        return classification, sn, un, cn, Ws_list, Wu_list, Wc_list

    def _sort_eigenvalues(self, eigvals: np.ndarray, eigvecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and canonicalize eigenpairs, preserving original order.

        Parameters
        ----------
        eigvals : np.ndarray
            Eigenvalues (input order preserved).
        eigvecs : np.ndarray
            Corresponding eigenvectors as columns (order preserved).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Cleaned eigenvalues and pivot-normalized eigenvectors with
            deterministic sign (pivot component made positive for real vectors).
        """

        cleaned_vals = np.array([self._zero_small_imag_part(ev, tol=1e-14) for ev in eigvals])
        cleaned_vecs = eigvecs.astype(np.complex128)

        for k in range(cleaned_vecs.shape[1]):
            vec = cleaned_vecs[:, k]
            pivot_index = 0
            while pivot_index < vec.shape[0] and abs(vec[pivot_index]) < 1e-14:
                pivot_index += 1
            if pivot_index < vec.shape[0]:
                pivot = vec[pivot_index]
                if abs(pivot) > 1e-14:
                    vec = vec / pivot
                    cleaned_vecs[:, k] = vec
            cleaned_vecs[:, k] = self._remove_infinitesimals_array(cleaned_vecs[:, k], tol=1e-14)

            # Deterministic sign for effectively real eigenvectors
            if np.max(np.abs(cleaned_vecs[:, k].imag)) < 1e-14 and pivot_index < cleaned_vecs.shape[0]:
                if cleaned_vecs[pivot_index, k].real < 0:
                    cleaned_vecs[:, k] = -cleaned_vecs[:, k]

        return cleaned_vals, cleaned_vecs

    def _validate_monodromy_shape(self, M: np.ndarray) -> None:
        """Ensure monodromy matrix has expected shape (6, 6).
        
        Parameters
        ----------
        M : numpy.ndarray
            Monodromy matrix to validate.

        Raises
        ------
        :class:`~hiten.algorithms.types.exceptions.BackendError`
            If M has incorrect shape.
        """
        if M.shape != (6, 6):
            raise BackendError("Input matrix M has incorrect shape {M.shape}, expected (6, 6).")

    def _compute_sorted_spectrum(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute, clean and sort the eigenspectrum of matrix A.
        
        Parameters
        ----------
        A : numpy.ndarray
            Matrix to compute the eigenspectrum of.

        Returns
        -------
        numpy.ndarray
            Eigenvalues.
        numpy.ndarray
            Eigenvectors.   
        """
        eigvals_raw, eigvecs_raw = self._compute_eigendecomposition(A)
        eigvals_clean = np.array([self._zero_small_imag_part(ev, tol=1e-12) for ev in eigvals_raw])
        eigvals_sorted, eigvecs_sorted = self._sort_eigenvalues(eigvals_clean, eigvecs_raw)
        return eigvals_sorted, eigvecs_sorted

    def _calc_stability_index(self, eigval: complex) -> complex:
        """Compute stability index nu = (lambda + 1/lambda) / 2 for eigenvalue.
        
        Parameters
        ----------
        eigval : complex
            Eigenvalue to compute the stability index of.
        """
        return (eigval + 1.0 / eigval) / 2.0

    def _compute_nu_from_eigvals(self, eigvals_sorted: np.ndarray, tol: float) -> np.ndarray:
        """Compute up to three stability indices from sorted eigenvalues.

        Parameters
        ----------
        eigvals_sorted : numpy.ndarray
            Sorted eigenvalues.
        tol : float
            Tolerance for eigenvalue comparison.

        Strategy:
        - Prefer a trivial pair near unit modulus first (if present)
        - Then find reciprocal pairs among remaining eigenvalues
        - Pad with NaN if fewer than three indices can be determined
        """
        used = np.zeros(eigvals_sorted.shape[0], dtype=bool)
        nu_list: List[complex] = []

        # 1) Trivial near-unit-magnitude pair
        unit_indices = [i for i, ev in enumerate(eigvals_sorted) if not used[i] and np.isclose(abs(ev), 1.0, rtol=tol, atol=tol)]
        if len(unit_indices) >= 2:
            i0, i1 = unit_indices[0], unit_indices[1]
            nu_list.append(self._calc_stability_index(eigvals_sorted[i0]))
            used[i0] = True
            used[i1] = True

        elif len(unit_indices) == 1:
            i0 = unit_indices[0]
            nu_list.append(self._calc_stability_index(eigvals_sorted[i0]))
            used[i0] = True

        n = eigvals_sorted.shape[0]
        for i in range(n):
            if used[i]:
                continue
            ev = eigvals_sorted[i]
            if abs(ev) < tol * tol:
                used[i] = True
                continue
            target = 1.0 / ev
            match_j = -1
            for j in range(i + 1, n):
                if used[j]:
                    continue
                if np.isclose(eigvals_sorted[j], target, rtol=tol, atol=tol):
                    match_j = j
                    break
            if match_j >= 0:
                nu_list.append(self._calc_stability_index(ev))
                used[i] = True
                used[match_j] = True
            else:
                used[i] = True

            if len(nu_list) >= 3:
                break

        if len(nu_list) < 3:
            while len(nu_list) < 3:
                nu_list.append(np.nan + 0j)

        return np.array(nu_list, dtype=np.complex128)

    def _remove_infinitesimals_in_place(self, vec: np.ndarray, tol: float = 1e-14) -> None:
        """Remove numerical noise from complex vector components in-place.

        Sets real and imaginary parts smaller than tolerance to exactly zero,
        helping prevent numerical artifacts from affecting downstream calculations.

        Parameters
        ----------
        vec : numpy.ndarray
            Complex vector to clean in-place.
        tol : float, optional
            Tolerance below which components are zeroed. Default is 1e-14.

        Notes
        -----
        Modifies the input vector directly. Particularly useful for cleaning
        eigenvectors that may contain tiny numerical artifacts.
        
        See Also
        --------
        :func:`~hiten.algorithms.common.linalg._remove_infinitesimals_array` : Non-destructive version
        :func:`~hiten.algorithms.common.linalg._zero_small_imag_part` : Cleanup for scalar complex values
        """
        for i in range(len(vec)):
            re = vec[i].real
            im = vec[i].imag
            if abs(re) < tol:
                re = 0.0
            if abs(im) < tol:
                im = 0.0
            vec[i] = re + 1j*im

    def _remove_infinitesimals_array(self, vec: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """Create cleaned copy of vector with numerical noise removed.

        Returns a copy of the input vector with real and imaginary components
        smaller than tolerance set to exactly zero. Preserves the original vector.

        Parameters
        ----------
        vec : numpy.ndarray
            Complex vector to clean.
        tol : float, optional
            Tolerance below which components are zeroed. Default is 1e-12.

        Returns
        -------
        numpy.ndarray
            Copy of input vector with small values replaced by exact zeros.
        
        See Also
        --------
        :func:`~hiten.algorithms.common.linalg._remove_infinitesimals_in_place` : In-place version
        :func:`~hiten.algorithms.common.linalg._zero_small_imag_part` : Cleanup for scalar complex values
        """
        vcopy = vec.copy()
        self._remove_infinitesimals_in_place(vcopy, tol)
        return vcopy

    def _zero_small_imag_part(self, eig_val: complex, tol: float = 1e-12) -> complex:
        """Remove small imaginary part from complex number.

        Sets imaginary part to zero if smaller than tolerance. Useful for
        cleaning eigenvalues that should be real but have numerical artifacts.

        Parameters
        ----------
        eig_val : complex
            Complex value to clean.
        tol : float, optional
            Tolerance below which imaginary part is zeroed. Default is 1e-12.

        Returns
        -------
        complex
            Cleaned complex value with small imaginary part removed.
            
        See Also
        --------
        :func:`~hiten.algorithms.common.linalg._remove_infinitesimals_array` : Vector version for arrays
        :func:`~hiten.algorithms.common.linalg._remove_infinitesimals_in_place` : In-place vector cleanup
        """
        if abs(eig_val.imag) < tol:
            return complex(eig_val.real, 0.0)
        return eig_val
