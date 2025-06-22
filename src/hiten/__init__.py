"""
hiten
=====

Top-level public API for the *hiten* package.

This module re-exports the most frequently used symbols from the
sub-packages so that user code can simply write, for example::

    >>> from hiten import Constants, CenterManifold, PoincareMap

instead of navigating the full internal hierarchy (``hiten.utils``,
``hiten.system`` …).

The full sub-packages remain accessible under their canonical names
(``hiten.utils``, ``hiten.system``, ``hiten.algorithms``).
"""

from __future__ import annotations

from importlib import metadata as _metadata

try:
    __version__: str = _metadata.version("hiten")
except _metadata.PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from . import algorithms, system, utils
from .system import *
from .system import __all__ as _SYSTEM_ALL
from .utils import *
from .utils import __all__ as _UTILS_ALL

__all__: list[str] = list(_UTILS_ALL) + list(_SYSTEM_ALL) + [
    "utils",
    "system",
    "algorithms",
    "__version__",
]

del _metadata, _UTILS_ALL, _SYSTEM_ALL
