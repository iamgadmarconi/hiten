"""Input/output utilities for manifold data.

This module provides functions for serializing and deserializing manifold
objects and their associated data to/from HDF5 files. It includes utilities
for saving and loading manifolds, their generating orbits, and manifold
computation results.

Notes
-----
All data is stored in HDF5 format with version tracking. The module supports
compression and handles both stable and unstable manifolds.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset
from hiten.utils.io.orbits import _read_orbit_group, _write_orbit_group

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold


HDF5_VERSION = "1.0"
"""HDF5 format version for manifold data."""

def save_manifold(
    manifold: "Manifold",
    path: str | Path,
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialize manifold to HDF5 file.

    Parameters
    ----------
    manifold : :class:`~hiten.system.manifold.Manifold`
        The manifold object to serialize.
    path : str or pathlib.Path
        File path where to save the manifold data.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The function saves the manifold's generating orbit, stability properties,
    and computation results (if available) to an HDF5 file. Trajectory data
    is stored in separate groups for each trajectory.
    
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.manifold import Manifold
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> orbit = L2.create_orbit('halo', amplitude_z=0.3)
    >>> manifold = orbit.manifold(stable=True, direction='positive')
    >>> save_manifold(manifold, "my_manifold.h5")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = manifold.__class__.__name__
        f.attrs["stable"] = bool(manifold.stable == 1)
        f.attrs["direction"] = "positive" if manifold.direction == 1 else "negative"

        ggrp = f.create_group("generating_orbit")
        _write_orbit_group(ggrp, manifold._generating_orbit, compression=compression, level=level)

        if manifold.result is not None:
            mr = manifold.result
            rgrp = f.create_group("result")
            _write_dataset(rgrp, "ysos", np.asarray(mr[0]))
            _write_dataset(rgrp, "dysos", np.asarray(mr[1]))
            rgrp.attrs["_successes"] = int(mr[4])
            rgrp.attrs["_attempts"] = int(mr[5])

            tgrp = rgrp.create_group("trajectories")
            for i, (states, times) in enumerate(zip(mr[2], mr[3])):
                sub = tgrp.create_group(str(i))
                _write_dataset(sub, "states", np.asarray(states), compression=compression, level=level)
                _write_dataset(sub, "times", np.asarray(times), compression=compression, level=level)


def load_manifold(path: str | Path) -> "Manifold":
    """Load a manifold from an HDF5 file.
    
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
        
    Notes
    -----
    The function reconstructs the manifold from serialized data, including
    the generating orbit, stability properties, and computation results
    (if available).
    
    Examples
    --------
    >>> manifold = load_manifold("my_manifold.h5")
    >>> print(f"Loaded manifold: stable={manifold.stable}, direction={manifold.direction}")
    """
    from hiten.system.manifold import Manifold

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        stable_flag = bool(f.attrs.get("stable", True))
        direction_str = f.attrs.get("direction", "positive")

        ggrp = f["generating_orbit"]
        gen_orbit = _read_orbit_group(ggrp)

        man = Manifold(
            generating_orbit=gen_orbit,
            stable=stable_flag,
            direction=direction_str,
        )

        if "result" in f:
            rgrp = f["result"]
            ysos = rgrp["ysos"][()] if "ysos" in rgrp else []
            dysos = rgrp["dysos"][()] if "dysos" in rgrp else []
            succ = int(rgrp.attrs.get("_successes", 0))
            attm = int(rgrp.attrs.get("_attempts", 0))

            states_list, times_list = [], []
            if "trajectories" in rgrp:
                tgrp = rgrp["trajectories"]
                for key in tgrp.keys():
                    sub = tgrp[key]
                    states_list.append(sub["states"][()])
                    times_list.append(sub["times"][()])

            man.dynamics._manifold_result = (
                float(ysos),
                float(dysos),
                states_list,
                times_list,
                succ,
                attm,
            )

    return man
