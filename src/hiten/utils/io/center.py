"""Input/output utilities for center manifold data.

This module provides functions for serializing and deserializing center manifold
objects using pickle, which preserves object relationships automatically.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.center import CenterManifold

__all__ = ["save_center_manifold", "load_center_manifold", "load_center_manifold_inplace"]



def save_center_manifold(cm: "CenterManifold", path: str | Path, **kwargs) -> None:
    """Serialize center manifold to a pickle file.

    Parameters
    ----------
    cm : :class:`~hiten.system.center.CenterManifold`
        The center manifold object to serialize.
    path : str or pathlib.Path
        File path where to save the center manifold data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
        
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.center import CenterManifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifold(L2, degree=10)
    >>> save_center_manifold(cm, "my_center_manifold.pkl")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(cm, f)


def load_center_manifold(path: str | Path) -> "CenterManifold":
    """Load a center manifold from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the center manifold data.
        
    Returns
    -------
    :class:`~hiten.system.center.CenterManifold`
        The reconstructed center manifold object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Examples
    --------
    >>> cm = load_center_manifold("my_center_manifold.pkl")
    >>> print(f"Loaded center manifold with degree {cm.degree}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


def load_center_manifold_inplace(obj: "CenterManifold", path: str | Path) -> None:
    """Populate an existing center manifold object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.center.CenterManifold`
        The center manifold object to populate.
    path : str or pathlib.Path
        File path containing the center manifold data.
    """
    tmp = load_center_manifold(path)
    obj.__dict__.update(tmp.__dict__)
