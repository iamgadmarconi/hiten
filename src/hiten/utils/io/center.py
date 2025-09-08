"""Input/output utilities for center manifold data.

This module provides functions for serializing and deserializing center manifold
objects and their associated data to/from HDF5 files. It includes utilities
for saving and loading center manifolds, their cached Hamiltonians, and
associated Poincare maps.

Notes
-----
All data is stored in HDF5 format with version tracking. The module supports
compression and optional inclusion of cached Hamiltonian data for faster
subsequent loads.
"""

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir
from hiten.utils.io.map import save_poincare_map, load_poincare_map

if TYPE_CHECKING:
    from hiten.system.center import CenterManifold
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap
    from hiten.system.hamiltonians.pipeline import HamiltonianPipeline

__all__ = ["save_center_manifold", "load_center_manifold"]

HDF5_VERSION = "2.0"
"""HDF5 format version for center manifold data."""

def _serialize_hamiltonians(pipeline: "HamiltonianPipeline") -> bytes:
    """Return a pickled representation of the pipeline's Hamiltonian cache.
    
    Parameters
    ----------
    pipeline : :class:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline`
        The Hamiltonian pipeline containing the cache to serialize.
        
    Returns
    -------
    bytes
        Pickled representation of the Hamiltonian cache data.
        
    Notes
    -----
    The serialized data includes polynomial coefficients, degrees, number of
    degrees of freedom, and names for each cached Hamiltonian.
    """
    data: Dict[str, Dict[str, Any]] = {}
    for form, ham in pipeline._hamiltonian_cache.items():
        data[form] = {
            "poly_H": ham.poly_H,
            "degree": ham.degree,
            "ndof": ham.ndof,
            "name": ham.name,
        }
    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_hamiltonians(blob: bytes, pipeline: "HamiltonianPipeline") -> None:
    """Deserialize Hamiltonian cache data into the pipeline.
    
    Parameters
    ----------
    blob : bytes
        Pickled representation of the Hamiltonian cache data.
    pipeline : :class:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline`
        The Hamiltonian pipeline to populate with the deserialized data.
        
    Notes
    -----
    This function reconstructs the Hamiltonian cache from serialized data,
    creating Hamiltonian objects and storing them in the pipeline's cache.
    """
    from hiten.system.hamiltonians.base import Hamiltonian

    cache_data: Dict[str, Dict[str, Any]] = pickle.loads(blob)
    for form, info in cache_data.items():
        ham = Hamiltonian(info["poly_H"], info["degree"], ndof=info["ndof"], name=info["name"])
        pipeline._hamiltonian_cache[form] = ham


def save_center_manifold(
    cm: "CenterManifold",
    dir_path: str | Path,
    *,
    include_hamiltonians: bool = False,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialize center manifold and optionally cached Hamiltonians.

    Parameters
    ----------
    cm : :class:`~hiten.system.center.CenterManifold`
        The center manifold object to serialize.
    dir_path : str or pathlib.Path
        Directory path where to save the center manifold data.
    include_hamiltonians : bool, default False
        If True, the cached Hamiltonian objects stored in the internal
        pipeline are pickled and saved alongside the core data. This speeds
        up subsequent loads at the cost of larger files.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The function creates an HDF5 file containing the center manifold data
    and optionally saves associated Poincare maps to separate files.
    
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.center import CenterManifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> cm = CenterManifold(L2, degree=10)
    >>> save_center_manifold(cm, "my_center_manifold")
    """
    from hiten.utils.io.common import _write_dataset

    dir_path = Path(dir_path)
    _ensure_dir(dir_path)

    main_file = dir_path / "manifold.h5"
    with h5py.File(main_file, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = cm.__class__.__name__
        f.attrs["degree"] = int(cm.degree)

        # Serialize point via pickle (small)
        point_blob = pickle.dumps(cm.point, protocol=pickle.HIGHEST_PROTOCOL)
        f.create_dataset("point_pickle", data=np.frombuffer(point_blob, dtype=np.uint8))

        if include_hamiltonians and cm.pipeline._hamiltonian_cache:
            ham_blob = _serialize_hamiltonians(cm.pipeline)
            _write_dataset(f, "hamiltonians_pickle", np.frombuffer(ham_blob, dtype=np.uint8))

    if cm._poincare_maps:
        maps_dir = dir_path / "maps"
        maps_dir.mkdir(exist_ok=True)
        keys_info = []
        for idx, (key, pmap) in enumerate(cm._poincare_maps.items()):
            filename = f"map_{idx}.h5"
            save_poincare_map(pmap, maps_dir / filename, compression=compression, level=level)
            keys_info.append({"key": list(key), "file": filename})
        (dir_path / "poincare_maps_keys.json").write_text(json.dumps(keys_info))
    else:
        # Remove stale files if present
        key_file = dir_path / "poincare_maps_keys.json"
        if key_file.exists():
            key_file.unlink()
        maps_dir = dir_path / "maps"
        if maps_dir.exists() and not any(maps_dir.iterdir()):
            maps_dir.rmdir()


def load_center_manifold(dir_path: str | Path) -> "CenterManifold":
    """Load a center manifold from a directory.
    
    Parameters
    ----------
    dir_path : str or pathlib.Path
        Directory path containing the center manifold data.
        
    Returns
    -------
    :class:`~hiten.system.center.CenterManifold`
        The reconstructed center manifold object.
        
    Raises
    ------
    FileNotFoundError
        If the required HDF5 file is not found in the directory.
        
    Notes
    -----
    The function reconstructs the center manifold from serialized data,
    including any cached Hamiltonians and associated Poincare maps.
    
    Examples
    --------
    >>> cm = load_center_manifold("my_center_manifold")
    >>> print(f"Loaded center manifold with degree {cm.degree}")
    """
    from hiten.system.center import CenterManifold

    dir_path = Path(dir_path)
    main_file = dir_path / "manifold.h5"
    if not main_file.exists():
        raise FileNotFoundError(main_file)

    with h5py.File(main_file, "r") as f:
        degree = int(f.attrs["degree"])
        point_blob = f["point_pickle"][()]
        point = pickle.loads(point_blob.tobytes())

        cm = CenterManifold(point, degree)

        if "hamiltonians_pickle" in f:
            ham_blob = f["hamiltonians_pickle"][()]
            _deserialize_hamiltonians(ham_blob.tobytes(), cm.pipeline)
            # Refresh internal Hamiltonian system reference to the deserialized one
            cm._hamsys = cm.pipeline.get_hamiltonian("center_manifold_real").hamsys

    maps_key_file = dir_path / "poincare_maps_keys.json"
    maps_dir = dir_path / "maps"
    if maps_key_file.exists():
        keys_info = json.loads(maps_key_file.read_text())
        for info in keys_info:
            key_list = info["key"]
            energy = key_list[0]
            config_tuple = tuple(tuple(item) for item in key_list[1])
            cache_key = (energy, config_tuple)

            pmap = load_poincare_map(maps_dir / info["file"], cm)
            cm._poincare_maps[cache_key] = pmap

    return cm
