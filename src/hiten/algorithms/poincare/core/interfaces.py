from abc import ABC
from typing import Tuple

from hiten.algorithms.types.core import (ConfigT, OutputsT, ProblemT,
                                         ResultT, _HitenBaseInterface)


class _SectionInterface(ABC):
    pass

class _PoincareBaseInterface(
    _HitenBaseInterface[
        ConfigT, 
        ProblemT, 
        ResultT, 
        OutputsT
    ]
):
    """Shared functionality for poincare map interfaces."""
    pass