"""Input/output utilities for LibrationPoint objects.

This module provides helpers to serialize and deserialize libration points
to/from HDF5 files. Data stored includes class name, mass parameter, optional
cached position/energy/Jacobi values, optional stability information, and
system context to aid reconstruction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.base import System


HDF5_VERSION = "1.0"
"""HDF5 format version for LibrationPoint data."""


def save_libration_point(lp: "LibrationPoint", path: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
    """Serialize a LibrationPoint to an HDF5 file.

    Parameters
    ----------
    lp : :class:`~hiten.system.libration.base.LibrationPoint`
        Libration point instance to serialize.
    path : str or pathlib.Path
        File path where to save the libration point.
    compression : str, optional
        Compression algorithm to use for HDF5 datasets.
    level : int, optional
        Compression level (0-9) for HDF5 datasets.
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = lp.__class__.__name__
        f.attrs["mu"] = float(lp.mu)

        # System context for convenience (non-binding during load)
        sys = lp.system
        try:
            f.attrs["primary"] = sys.primary.name
            f.attrs["secondary"] = sys.secondary.name
            f.attrs["distance_km"] = float(sys.distance)
        except Exception:
            pass

        # Optional cached values from cache store
        try:
            pos = lp.cache_get(("position",))
            if pos is not None:
                _write_dataset(f, "position", np.asarray(pos), compression=compression, level=level)
        except Exception:
            pass
        try:
            energy = lp.cache_get(("energy",))
            if energy is not None:
                f.attrs["energy"] = float(energy)
        except Exception:
            pass
        try:
            jacobi = lp.cache_get(("jacobi_constant",))
            if jacobi is not None:
                f.attrs["jacobi"] = float(jacobi)
        except Exception:
            pass

        if getattr(lp, "_stability_info", None) is not None:
            sgrp = f.create_group("stability")
            sn, un, cn, Ws, Wu, Wc = lp._stability_info
            _write_dataset(sgrp, "sn", np.asarray(sn))
            _write_dataset(sgrp, "un", np.asarray(un))
            _write_dataset(sgrp, "cn", np.asarray(cn))
            _write_dataset(sgrp, "Ws", np.asarray(Ws))
            _write_dataset(sgrp, "Wu", np.asarray(Wu))
            _write_dataset(sgrp, "Wc", np.asarray(Wc))


def _construct_lp_by_class(class_name: str, system: "System") -> "LibrationPoint":
    """Construct a LibrationPoint subclass instance by class name."""
    from importlib import import_module

    # Try known modules
    for mod_name in (
        "hiten.system.libration.collinear",
        "hiten.system.libration.triangular",
    ):
        try:
            mod = import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if hasattr(mod, class_name):
            cls = getattr(mod, class_name)
            return cls(system)
    # Fallback: raise
    raise ImportError(f"LibrationPoint class '{class_name}' not found in expected modules.")


def load_libration_point(path: str | Path, system: "System") -> "LibrationPoint":
    """Load a LibrationPoint from an HDF5 file given a System.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the libration point data.
    system : :class:`~hiten.system.base.System`
        The system to attach the libration point to.

    Returns
    -------
    :class:`~hiten.system.libration.base.LibrationPoint`
        The reconstructed libration point instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        class_name = str(f.attrs.get("class", "L1Point"))

    lp = _construct_lp_by_class(class_name, system)
    load_libration_point_inplace(lp, path)
    return lp


def load_libration_point_inplace(obj: "LibrationPoint", path: str | Path) -> None:
    """Populate an existing LibrationPoint object from an HDF5 file.

    Parameters
    ----------
    obj : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point instance to populate.
    path : str or pathlib.Path
        File path containing the libration point data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        file_cls = str(f.attrs.get("class", obj.__class__.__name__))
        if file_cls != obj.__class__.__name__:
            raise ValueError(
                f"Mismatch between file ({file_cls}) and object ({obj.__class__.__name__}) classes."
            )

        # Cached scalars via cache
        if "energy" in f.attrs:
            obj.cache_set(("energy",), float(f.attrs["energy"]))
        if "jacobi" in f.attrs:
            obj.cache_set(("jacobi_constant",), float(f.attrs["jacobi"]))

        # Position
        if "position" in f:
            obj.cache_set(("position",), f["position"][()])

        # Stability info
        if "stability" in f:
            sgrp = f["stability"]
            sn = sgrp["sn"][()] if "sn" in sgrp else np.array([])
            un = sgrp["un"][()] if "un" in sgrp else np.array([])
            cn = sgrp["cn"][()] if "cn" in sgrp else np.array([])
            Ws = sgrp["Ws"][()] if "Ws" in sgrp else np.empty((0, 0))
            Wu = sgrp["Wu"][()] if "Wu" in sgrp else np.empty((0, 0))
            Wc = sgrp["Wc"][()] if "Wc" in sgrp else np.empty((0, 0))
            obj._stability_info = (sn, un, cn, Ws, Wu, Wc)
        else:
            obj._stability_info = None

        # Clear normal-form heavy caches; they can be recomputed
        obj._linear_data = None
        obj._cm_registry = {}


