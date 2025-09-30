"""Input/output utilities for Poincare map data.

This module provides functions for serializing and deserializing Poincare map
objects using pickle, which preserves object relationships automatically.
"""

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.maps.center import CenterManifoldMap


def save_poincare_map(pmap: "CenterManifoldMap", path: str | Path, **kwargs) -> None:
    """Serialize Poincare map to a pickle file.

    Parameters
    ----------
    pmap : :class:`~hiten.system.maps.center.CenterManifoldMap`
        The Poincare map object to serialize.
    path : str or pathlib.Path
        File path where to save the Poincare map data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
        
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.maps.center import CenterManifoldMap
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifoldMap(L2, degree=10)
    >>> pmap = cm.poincare_map(energy=0.1)
    >>> save_poincare_map(pmap, "my_poincare_map.pkl")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(pmap, f)


def load_poincare_map_inplace(obj: "CenterManifoldMap", path: str | Path) -> None:
    """Populate an existing Poincare map object from a pickle file.
    
    Parameters
    ----------
    obj : :class:`~hiten.system.maps.center.CenterManifoldMap`
        The Poincare map object to populate with data.
    path : str or pathlib.Path
        File path containing the Poincare map data.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Examples
    --------
    >>> from hiten.system.maps.center import CenterManifoldMap
    >>> pmap = CenterManifoldMap(cm, energy=0.1)
    >>> load_poincare_map_inplace(pmap, "my_poincare_map.pkl")
    """
    tmp = load_poincare_map(path)
    obj.__dict__.update(tmp.__dict__)


def load_poincare_map(path: str | Path) -> "CenterManifoldMap":
    """Load a Poincare map from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the Poincare map data.
    cm : :class:`~hiten.system.maps.center.CenterManifoldMap`
        The center manifold object to associate with the map.
        
    Returns
    -------
    :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
        The reconstructed Poincare map object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.center import CenterManifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifold(L2, degree=10)
    >>> pmap = load_poincare_map("my_poincare_map.pkl", cm)
    >>> print(f"Loaded map with energy {pmap.energy}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)
