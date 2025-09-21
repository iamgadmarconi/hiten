"""Input/output utilities for Poincare map data.

This module provides functions for serializing and deserializing Poincare map
objects and their associated data to/from HDF5 files. It includes utilities
for saving and loading center manifold maps, their sections, and configuration
data.

Notes
-----
All data is stored in HDF5 format with version tracking. The module supports
compression and handles both in-place loading and new object creation.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
    from hiten.system.center import CenterManifold


HDF5_VERSION = "2.0"
"""HDF5 format version for Poincare map data."""


def save_poincare_map(
    pmap: "CenterManifoldMap",
    path: str | Path,
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialize Poincare map to HDF5 file.

    Parameters
    ----------
    pmap : :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
        The Poincare map object to serialize.
    path : str or pathlib.Path
        File path where to save the Poincare map data.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The function saves the Poincare map's energy, configuration, and all
    cached sections to an HDF5 file. If no sections are cached, the map
    is computed first.
    
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.center import CenterManifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifold(L2, degree=10)
    >>> pmap = cm.poincare_map(energy=0.1)
    >>> save_poincare_map(pmap, "my_poincare_map.h5")
    """

    if not pmap._sections:
        try:
            # Prefer computing the default section via the facade's solver
            pmap._solve_and_cache(None)
        except Exception:
            # Fallback: try default section explicitly
            pmap._solve_and_cache(pmap.config.section_coord)

    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = pmap.__class__.__name__
        f.attrs["energy"] = float(pmap.energy)
        f.attrs["config_json"] = json.dumps(asdict(pmap.config))

        sec_root = f.create_group("sections")

        for coord, sec in pmap._sections.items():
            g = sec_root.create_group(str(coord))
            _write_dataset(g, "points", np.asarray(sec.points), compression=compression, level=level)
            if sec.states is not None:
                _write_dataset(g, "states", np.asarray(sec.states), compression=compression, level=level)
            g.attrs["labels_json"] = json.dumps(list(sec.labels))


def load_poincare_map_inplace(
    obj: "CenterManifoldMap",
    path: str | Path,
) -> None:
    """Populate Poincare map object with data from HDF5 file.
    
    Parameters
    ----------
    obj : :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
        The Poincare map object to populate with data.
    path : str or pathlib.Path
        File path containing the Poincare map data.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Notes
    -----
    This function loads data into an existing Poincare map object,
    clearing any existing sections and replacing them with the loaded data.
    
    Examples
    --------
    >>> from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
    >>> pmap = CenterManifoldMap(cm, energy=0.1)
    >>> load_poincare_map_inplace(pmap, "my_poincare_map.h5")
    """

    from hiten.algorithms.poincare.centermanifold.config import _CenterManifoldMapConfig
    from hiten.algorithms.poincare.centermanifold.types import CenterManifoldMapResults

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        obj._energy = float(f.attrs["energy"])

        cfg_json = f.attrs.get("config_json", "{}")
        obj.config = _CenterManifoldMapConfig(**json.loads(cfg_json))

        obj._sections.clear()

        sec_root = f["sections"]
        for coord in sec_root.keys():
            g = sec_root[coord]
            pts = g["points"][()]
            sts = g["states"][()] if "states" in g else np.full((pts.shape[0], 4), np.nan)
            labels_json = g.attrs.get("labels_json")
            labels = tuple(json.loads(labels_json)) if labels_json else ("q2", "p2")
            obj._sections[str(coord)] = CenterManifoldMapResults(pts, sts, labels)

        obj._section = obj._sections[obj.config.section_coord]


def load_poincare_map(path: str | Path, cm: "CenterManifold") -> "CenterManifoldMap":
    """Load a Poincare map from an HDF5 file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the Poincare map data.
    cm : :class:`~hiten.system.center.CenterManifold`
        The center manifold object to associate with the map.
        
    Returns
    -------
    :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
        The reconstructed Poincare map object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Notes
    -----
    This function creates a new Poincare map object and loads data into it.
    The energy is read from the file attributes, and the map is populated
    with the loaded section data.
    
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.center import CenterManifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifold(L2, degree=10)
    >>> pmap = load_poincare_map("my_poincare_map.h5", cm)
    >>> print(f"Loaded map with energy {pmap.energy}")
    """
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        energy = float(f.attrs["energy"])

    pmap = CenterManifoldMap(cm, energy)
    load_poincare_map_inplace(pmap, path)
    return pmap
