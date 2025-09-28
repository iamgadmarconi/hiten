"""Input/output utilities for manifold data.

This module provides functions for serializing and deserializing manifold
objects using pickle, which preserves object relationships automatically.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold

def save_manifold(manifold: "Manifold", path: str | Path, **kwargs) -> None:
    """Serialize manifold to a pickle file.

    Parameters
    ----------
    manifold : :class:`~hiten.system.manifold.Manifold`
        The manifold object to serialize.
    path : str or pathlib.Path
        File path where to save the manifold data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(manifold, f)


def load_manifold(path: str | Path) -> "Manifold":
    """Load a manifold from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the manifold data.
        
    Returns
    -------
    :class:`~hiten.system.manifold.Manifold`
        The reconstructed manifold object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)
