
"""Input/output utilities for orbit family data.

This module provides functions for serializing and deserializing orbit family
objects using pickle, which preserves object relationships automatically.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.family import OrbitFamily
    from hiten.system.orbits.base import PeriodicOrbit


def save_family(family: "OrbitFamily", filepath: str | Path, **kwargs) -> None:
    """Serialize an orbit family to a pickle file."""
    path = Path(filepath)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(family, f)

def load_family(filepath: str | Path, **kwargs) -> "OrbitFamily":
    """Load an orbit family from a pickle file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def load_family_inplace(family: "OrbitFamily", filepath: str | Path, **kwargs) -> None:
    """Load an orbit family into an existing object from a pickle file."""
    tmp = load_family(filepath)
    family.__dict__.update(tmp.__dict__)