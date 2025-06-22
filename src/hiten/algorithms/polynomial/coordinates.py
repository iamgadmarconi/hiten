r"""
polynomial.coordinates
======================

Lightweight helpers for manipulating 6-D complex phase-space vectors
:math:`(q_{1}, q_{2}, q_{3}, p_{1}, p_{2}, p_{3})` that appear in the
polynomial normal-form machinery.

The routines are deliberately minimal and allocation-friendly because they
are frequently invoked from performance-critical, Numba-accelerated kernels
in :pymod:`hiten.algorithms.polynomial`.

* :pyfunc:`_clean_coordinates` removes numerical noise below a configurable
  tolerance from both the real and imaginary parts of a complex vector.
* :pyfunc:`_substitute_coordinates` applies a linear change of variables
  represented by a :math:`6\times 6` substitution matrix.
"""

import numpy as np

from hiten.utils.log_config import logger


def _clean_coordinates(coords: np.ndarray, tol: float = 1e-30) -> np.ndarray:
    r"""
    Remove tiny numerical artefacts from a complex coordinate vector.

    Parameters
    ----------
    coords : numpy.ndarray
        Input array of shape ``(6,)`` interpreted as complex coordinates.
        The function accepts any real- or complex-typed array that can be cast
        to ``np.complex128``.
    tol : float, default 1e-30
        Absolute threshold below which real or imaginary components are set to
        zero. The default is conservative and tailored to IEEE-754 double
        precision.

    Returns
    -------
    numpy.ndarray
        Cleaned copy of *coords* with the same dtype as
        :pydata:`numpy.complex128`.

    Notes
    -----
    The routine operates element-wise. If any changes are made, a warning is
    emitted via :pyfunc:`hiten.utils.log_config.logger`.

    Examples
    --------
    >>> import numpy as np
    >>> from hiten.algorithms.polynomial.coordinates import _clean_coordinates
    >>> c = np.array([1+1e-40j, 0+0j, 1e-32+2j, 0j, -3e-31, 5], dtype=complex)
    >>> _clean_coordinates(c, tol=1e-30)
    array([1.+0.j, 0.+0.j, 0.+2.j, 0.+0.j, 0.+0.j, 5.+0.j])
    """
    before = np.asarray(coords, dtype=np.complex128)
    
    real_part = np.real(before)
    imag_part = np.imag(before)

    cleaned_real = np.where(np.abs(real_part) < tol, 0.0, real_part)
    cleaned_imag = np.where(np.abs(imag_part) < tol, 0.0, imag_part)

    after = cleaned_real + 1j * cleaned_imag

    if np.any(before != after):
        logger.warning(
            "Cleaned %d coordinates.\nBefore: %s\nAfter:  %s",
            np.sum(before != after), before, after,
        )

    return after


def _substitute_coordinates(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    r"""
    Apply a linear substitution matrix to a coordinate vector.

    The operation performed is

    :math:`y_{i} = \sum_{j=1}^{6} M_{ij}\,x_{j},`

    where :math:`M` is *matrix*, :math:`x` is *coords* and the result
    :math:`y` is returned.

    Parameters
    ----------
    coords : numpy.ndarray
        Source coordinates with shape ``(6,)``.
    matrix : numpy.ndarray
        Transformation matrix with shape ``(6, 6)``. Only non-zero entries are
        accessed so sparse structures are inexpensive although *matrix* itself
        must support NumPy indexing.

    Returns
    -------
    numpy.ndarray
        Transformed coordinates with dtype :pydata:`numpy.complex128`.

    Raises
    ------
    ValueError
        If *matrix* is not of shape ``(6, 6)``.

    Notes
    -----
    The implementation avoids temporary allocations by accumulating directly
    into the output array. Real inputs are seamlessly promoted to complex.

    Examples
    --------
    >>> import numpy as np
    >>> from hiten.algorithms.polynomial.coordinates import _substitute_coordinates
    >>> M = np.eye(6)
    >>> M[0, 1] = 2  # simple shear: q1 -> q1 + 2*q2
    >>> _substitute_coordinates(np.arange(6), M)
    array([ 2.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  5.+0.j])
    """
    if matrix.shape != (6, 6):
        raise ValueError(
            "Expected substitution matrix of shape (6, 6), "
            f"got {matrix.shape} instead."
        )

    transformed_coords = np.zeros(6, dtype=np.complex128)
    
    for i in range(6):
        for j in range(6):
            coeff = matrix[i, j]
            if coeff != 0:
                transformed_coords[i] += coeff * coords[j]
    
    return transformed_coords