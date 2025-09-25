
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
from typing import TYPE_CHECKING, Tuple

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset
from hiten.utils.io.orbits import _read_orbit_group, _write_orbit_group

if TYPE_CHECKING:
    from hiten.system.family import OrbitFamily
    from hiten.system.orbits.base import PeriodicOrbit


HDF5_VERSION = "1.0"
"""HDF5 format version for orbit family data."""


def save_family(
    family: "OrbitFamily",
    filepath: str | Path,
    **kwargs,
) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    compression = kwargs.get("compression", "gzip")
    level = kwargs.get("level", 4)

    with h5py.File(path, "w") as h5:
        h5.attrs["class"] = "OrbitFamily"
        h5.attrs["format_version"] = "1.0"
        h5.attrs["parameter_name"] = family.parameter_name
        h5.create_dataset("parameter_values", data=family.parameter_values)

        grp = h5.create_group("orbits")
        for idx, orbit in enumerate(family.orbits):
            subgrp = grp.create_group(str(idx))
            _write_orbit_group(subgrp, orbit, compression=compression, level=level)

    return None

def load_family(
    self,
    filepath: str | Path,
    **kwargs,
) -> Tuple[list["PeriodicOrbit"], str, list[float]]:
    path = Path(filepath)
    with h5py.File(path, "r") as h5:
        if str(h5.attrs.get("class", "")) != "OrbitFamily":
            raise ValueError("File does not contain an OrbitFamily object")

        parameter_name = str(h5.attrs["parameter_name"])
        parameter_values = h5["parameter_values"][()]

        orbits: list["PeriodicOrbit"] = []
        for key in sorted(h5["orbits"], key=lambda k: int(k)):
            grp = h5["orbits"][key]
            orbits.append(_read_orbit_group(grp))

    return orbits, parameter_name, parameter_values

def load_family_inplace(
    family: "OrbitFamily",
    filepath: str | Path,
    **kwargs,
) -> None:
    path = Path(filepath)
    with h5py.File(path, "r") as h5:
        orbits: list["PeriodicOrbit"] = []
        for key in sorted(h5["orbits"], key=lambda k: int(k)):
            grp = h5["orbits"][key]
            orbits.append(_read_orbit_group(grp))