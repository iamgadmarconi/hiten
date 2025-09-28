"""Input/output utilities for Body objects.

This module provides helpers to serialize and deserialize Body instances
to/from HDF5 files, mirroring the approach used for systems and orbits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.body import Body


HDF5_VERSION = "1.0"
"""HDF5 format version for Body data."""


def save_body(body: "Body", path: str | Path) -> None:
    """Serialize a Body to an HDF5 file.

    Parameters
    ----------
    body : :class:`~hiten.system.body.Body`
        Body instance to serialize.
    path : str or pathlib.Path
        File path where to save the body.
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = body.__class__.__name__
        f.attrs["name"] = body.name
        f.attrs["mass"] = float(body.mass)
        f.attrs["radius"] = float(body.radius)
        f.attrs["color"] = body.color

        # For secondaries, store parent name (optional metadata)
        parent_name = body.parent.name if body.parent is not None and body.parent is not body else None
        if parent_name is not None:
            f.attrs["parent_name"] = parent_name


def load_body(path: str | Path) -> "Body":
    """Load a Body from an HDF5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the body description.

    Returns
    -------
    :class:`~hiten.system.body.Body`
        The reconstructed Body instance.
    """
    from hiten.system.body import Body

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        name = str(f.attrs.get("name", "Body"))
        mass = float(f.attrs.get("mass", 1.0))
        radius = float(f.attrs.get("radius", 1.0))
        color = str(f.attrs.get("color", "#000000"))

    return Body(name, mass, radius, color)


def load_body_inplace(obj: "Body", path: str | Path) -> None:
    """Populate an existing Body object from an HDF5 file.

    Parameters
    ----------
    obj : :class:`~hiten.system.body.Body`
        Body instance to populate.
    path : str or pathlib.Path
        File path with serialized body.
    """
    tmp = load_body(path)
    obj.__dict__.update(tmp.__dict__)


