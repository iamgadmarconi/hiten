"""Input/output utilities for invariant torus data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset
from hiten.utils.io.orbits import _read_orbit_group, _write_orbit_group

if TYPE_CHECKING:  # pragma: no cover - import-guard for typing
    from hiten.system.torus import InvariantTori, Torus

HDF5_VERSION = "1.0"
"""HDF5 format version for invariant torus data."""


def save_torus(torus: "InvariantTori", path: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
    """Serialize an invariant torus facade and cached grid to HDF5."""

    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = torus.__class__.__name__
        f.attrs["orbit_class"] = torus.orbit.__class__.__name__
        f.attrs["mu"] = float(torus.orbit.mu)
        f.attrs["period"] = float(torus.orbit.period)
        f.attrs["family"] = torus.orbit.family

        orbit_grp = f.create_group("generating_orbit")
        _write_orbit_group(orbit_grp, torus.orbit, compression=compression, level=level)

        if torus.rotation_number is not None:
            f.attrs["rotation_number"] = float(torus.rotation_number)

        if torus.grid is not None:
            _write_dataset(f, "grid", np.asarray(torus.grid), compression=compression, level=level)


def load_torus(path: str | Path) -> "InvariantTori":
    """Load an invariant torus facade from HDF5."""

    from hiten.system.torus import InvariantTori

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        orbit_grp = f["generating_orbit"]
        orbit = _read_orbit_group(orbit_grp)
        torus = InvariantTori(orbit)

        if "grid" in f:
            torus._grid = f["grid"][()]

        if "theta1" in f:
            torus._theta1 = f["theta1"][()]
        if "ubar" in f:
            torus._ubar = f["ubar"][()]
        if "y_series" in f:
            torus._y_series = f["y_series"][()]

        rot = float(f.attrs.get("rotation_number", np.nan))
        if not np.isnan(rot):
            torus._rotation_number = rot

    return torus


def load_torus_inplace(obj: "InvariantTori", path: str | Path) -> None:
    """Populate an existing invariant torus facade from HDF5."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if obj.__class__.__name__ != f.attrs.get("class", obj.__class__.__name__):
            raise ValueError("Mismatch between stored class and target object.")

        if "grid" in f:
            obj._grid = f["grid"][()]

        if "theta1" in f:
            obj._theta1 = f["theta1"][()]
        if "ubar" in f:
            obj._ubar = f["ubar"][()]
        if "y_series" in f:
            obj._y_series = f["y_series"][()]

        rot = float(f.attrs.get("rotation_number", np.nan))
        obj._rotation_number = None if np.isnan(rot) else rot

