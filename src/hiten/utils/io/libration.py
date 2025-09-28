"""Input/output utilities for LibrationPoint objects.

This module provides helpers to serialize and deserialize libration points
using pickle, which preserves object relationships automatically.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.base import System


def save_libration_point(lp: "LibrationPoint", path: str | Path, **kwargs) -> None:
    """Serialize a LibrationPoint to a pickle file.

    Parameters
    ----------
    lp : :class:`~hiten.system.libration.base.LibrationPoint`
        Libration point instance to serialize.
    path : str or pathlib.Path
        File path where to save the libration point.
    **kwargs
        Additional arguments (ignored for pickle serialization).
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(lp, f)


def load_libration_point(path: str | Path, **kwargs) -> "LibrationPoint":
    """Load a LibrationPoint from a pickle file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the libration point data.
    **kwargs
        Additional arguments (ignored for pickle deserialization).

    Returns
    -------
    :class:`~hiten.system.libration.base.LibrationPoint`
        The reconstructed libration point instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_libration_point_inplace(obj: "LibrationPoint", path: str | Path) -> None:
    """Populate an existing LibrationPoint object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point instance to populate.
    path : str or pathlib.Path
        File path containing the libration point data.
    """
    tmp = load_libration_point(path)
    obj.__dict__.update(tmp.__dict__)


