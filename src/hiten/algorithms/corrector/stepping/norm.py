import numpy as np


def _default_norm(r: np.ndarray) -> float:
    """Compute L2 norm of residual vector.

    Parameters
    ----------
    r : np.ndarray
        Residual vector.
        
    Returns
    -------
    float
        L2 norm of the residual.
        
    Notes
    -----
    Uses L2 norm as default because most invariance residuals
    are already normalized by the number of components.
    """
    return float(np.linalg.norm(r))

def _infinity_norm(r: np.ndarray) -> float:
    """Compute infinity norm of residual vector.

    Parameters
    ----------
    r : np.ndarray
        Residual vector.
        
    Returns
    -------
    float
        Maximum absolute component of the residual.
    """
    return float(np.linalg.norm(r, ord=np.inf))
