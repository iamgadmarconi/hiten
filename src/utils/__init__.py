"""utils
========
Convenient re-exports for the most frequently accessed helper utilities.

Typical usage examples::

    from utils import Constants, format_cm_table

    mass_earth = Constants.get_mass("earth")
"""

from .constants import Constants, G  # noqa: F401 – re-export for public API
from .printing import format_cm_table

__all__ = [
    # Constants
    "Constants",
    "G",
    # Printing helpers
    "format_cm_table",
]
