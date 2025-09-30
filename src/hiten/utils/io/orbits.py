"""Input/output utilities for periodic orbit data.

This module provides functions for serializing and deserializing periodic orbit
objects using pickle, which preserves object relationships automatically.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit




def save_periodic_orbit(orbit: "PeriodicOrbit", path: str | Path, **kwargs) -> None:
    """Serialize periodic orbit to a pickle file.

    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The periodic orbit object to serialize.
    path : str or pathlib.Path
        File path where to save the orbit data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(orbit, f)


def load_periodic_orbit_inplace(obj: "PeriodicOrbit", path: str | Path) -> None:
    """Load periodic orbit data into existing object.
    
    Parameters
    ----------
    obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The periodic orbit object to populate with data.
    path : str or pathlib.Path
        File path containing the orbit data.
    """
    tmp = load_periodic_orbit(path)
    obj.__dict__.update(tmp.__dict__)


def load_periodic_orbit(path: str | Path) -> "PeriodicOrbit":
    """Load a periodic orbit from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the orbit data.
        
    Returns
    -------
    :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The reconstructed periodic orbit object.
        
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
