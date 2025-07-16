from abc import ABC, abstractmethod

import numpy as np

from hiten.algorithms.continuation.base import _ContinuationEngine


class _PseudoArcLength(_ContinuationEngine, ABC):
    """Abstract base class for pseudo arclength continuation algorithms"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)