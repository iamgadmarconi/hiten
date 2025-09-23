from abc import ABC
from typing import Tuple


class _SectionInterface(ABC):
    """Abstract base class for Poincare section interfaces.

    This abstract base class defines the interface for section
    adapters that specify the section coordinate and plane
    coordinates for Poincare map computation.

    Parameters
    ----------
    section_coord : str
        The coordinate that defines the section (e.g., "q2", "p2").
        This is the coordinate that is held constant on the section.
    plane_coords : tuple[str, str]
        Tuple of two coordinate labels that define the section plane
        (e.g., ("q2", "p2")). These are the coordinates that vary
        in the section plane.

    Notes
    -----
    This class serves as the base for all section interface
    implementations. Concrete implementations should inherit from
    this class and add their specific section parameters/behaviour.

    The section coordinate determines which coordinate is held
    constant, while the plane coordinates determine which two
    coordinates are used to represent points in the section.
    """
    section_coord: str
    plane_coords: Tuple[str, str]