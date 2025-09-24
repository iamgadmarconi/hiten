from abc import ABC
from typing import Tuple

from hiten.algorithms.utils.core import (ConfigT, DomainT, OutputsT, ProblemT,
                                         ResultT, _HitenBaseInterface)


class _SectionInterface(ABC):
    section_coord: str
    plane_coords: Tuple[str, str]


class _PoincareBaseInterface(_HitenBaseInterface[DomainT, ConfigT, ProblemT, ResultT, OutputsT]):
    """Shared functionality for poincare map interfaces."""

    section_interface: _SectionInterface | None = None