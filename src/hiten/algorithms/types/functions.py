from typing import Callable

import numpy as np

from hiten.algorithms.types.core import _HitenBaseEvent


class _NormFn(_HitenBaseEvent):
    """Function to compute the norm of a vector."""
    _version: float = 1.0

    norm_fn: Callable[[np.ndarray], float]

class _ResidualFn(_HitenBaseEvent):
    """Function to compute the residual of a vector."""
    _version: float = 1.0

    residual_fn: Callable[[np.ndarray], np.ndarray]

class _JacobianFn(_HitenBaseEvent):
    """Function to compute the Jacobian of a vector."""
    _version: float = 1.0

    jacobian_fn: Callable[[np.ndarray], np.ndarray]