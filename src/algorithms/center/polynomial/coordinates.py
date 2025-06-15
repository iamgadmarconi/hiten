import numpy as np

from utils.log_config import logger


def _clean_coordinates(coords: np.ndarray, tol: float = 1e-30) -> np.ndarray:
    """Clean tiny numerical artifacts from complex coordinates."""
    before = np.asarray(coords, dtype=np.complex128)
    
    real_part = np.real(before)
    imag_part = np.imag(before)

    cleaned_real = np.where(np.abs(real_part) < tol, 0.0, real_part)
    cleaned_imag = np.where(np.abs(imag_part) < tol, 0.0, imag_part)

    after = cleaned_real + 1j * cleaned_imag

    if np.any(before != after):
        logger.warning(f"Cleaned {np.sum(before != after)} coordinates. \nBefore: {before}\nAfter: {after}")

    return after

def _substitute_coordinates(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a coordinate transformation using a matrix as coordinate substitution.
    
    Parameters
    ----------
    coords : np.ndarray
        Input coordinates to transform
    matrix : np.ndarray
        Transformation matrix (6x6)
        
    Returns
    -------
    np.ndarray
        Transformed coordinates where result[i] = Σ_j matrix[i,j] * coords[j]
    """
    transformed_coords = np.zeros(6, dtype=np.complex128)
    
    for i in range(6):
        for j in range(6):
            if matrix[i, j] != 0:
                transformed_coords[i] += matrix[i, j] * coords[j]
    
    return transformed_coords