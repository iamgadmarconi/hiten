from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit

HDF5_VERSION = "1.0"


def _write_orbit_group(
    grp: h5py.Group,
    orbit: "PeriodicOrbit",
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialise *orbit* into *grp*.

    The HDF5 *grp* may be the root file object or a subgroup - the helper does
    not make any assumptions about hierarchy, making it re-usable for nested
    structures (e.g. manifolds that embed a generating orbit).
    """

    grp.attrs["format_version"] = HDF5_VERSION
    grp.attrs["class"] = orbit.__class__.__name__
    grp.attrs["family"] = orbit.family
    grp.attrs["mu"] = float(orbit.mu)
    grp.attrs["period"] = -1.0 if orbit.period is None else float(orbit.period)

    _write_dataset(grp, "initial_state", np.asarray(orbit._initial_state))

    if getattr(orbit, "_system", None) is not None:
        grp.attrs["primary"] = orbit._system.primary.name
        grp.attrs["secondary"] = orbit._system.secondary.name
        grp.attrs["distance_km"] = float(orbit._system.distance)

    if getattr(orbit, "libration_point", None) is not None:
        grp.attrs["libration_index"] = int(orbit.libration_point.idx)

    if orbit._trajectory is not None:
        _write_dataset(grp, "trajectory", np.asarray(orbit._trajectory), compression=compression, level=level)
        _write_dataset(grp, "times", np.asarray(orbit._times), compression=compression, level=level)

    if orbit._stability_info is not None:
        sgrp = grp.create_group("stability")
        indices, eigvals, eigvecs = orbit._stability_info
        _write_dataset(sgrp, "indices", np.asarray(indices))
        _write_dataset(sgrp, "eigvals", np.asarray(eigvals))
        _write_dataset(sgrp, "eigvecs", np.asarray(eigvecs))


_ORBIT_CLASSES: dict[str, type] = {}


def register_orbit_class(cls):
    """Decorator that registers *cls* for deserialisation"""
    _ORBIT_CLASSES[cls.__name__] = cls
    return cls


def _read_orbit_group(grp: h5py.Group):
    """Reconstruct and return a PeriodicOrbit instance from *grp*."""
    cls_name = grp.attrs.get("class", "GenericOrbit")
    orbit_cls = _ORBIT_CLASSES.get(cls_name)

    if orbit_cls is None:
        from importlib import import_module

        for mod_name in (
            "hiten.system.orbits.halo",
            "hiten.system.orbits.lyapunov",
            "hiten.system.orbits.base",
        ):
            try:
                mod = import_module(mod_name)
            except ModuleNotFoundError:
                continue
            if hasattr(mod, cls_name):
                orbit_cls = getattr(mod, cls_name)
                _ORBIT_CLASSES[cls_name] = orbit_cls  # cache for next time
                break

    if orbit_cls is None:
        raise ImportError(
            f"Orbit class '{cls_name}' not found. Ensure the class is defined and imported correctly."
        )
    orbit: "PeriodicOrbit" = orbit_cls.__new__(orbit_cls)

    orbit._family = str(grp.attrs.get("family", orbit._family))
    orbit._mu = float(grp.attrs.get("mu", np.nan))

    period_val = float(grp.attrs.get("period", -1.0))
    orbit.period = None if period_val < 0 else period_val

    orbit._initial_state = grp["initial_state"][()]

    try:
        primary_name = grp.attrs.get("primary")
        secondary_name = grp.attrs.get("secondary")
        distance_km = float(grp.attrs.get("distance_km", -1.0))
        lib_idx = int(grp.attrs.get("libration_index", -1))

        if primary_name and secondary_name and distance_km > 0:
            from hiten.system.base import System
            from hiten.system.body import Body
            from hiten.utils.constants import Constants

            p_key, s_key = str(primary_name).lower(), str(secondary_name).lower()
            try:
                primary = Body(primary_name.capitalize(), Constants.get_mass(p_key), Constants.get_radius(p_key))
                secondary = Body(secondary_name.capitalize(), Constants.get_mass(s_key), Constants.get_radius(s_key), _parent_input=primary)
            except Exception:
                primary = Body(primary_name.capitalize(), 1.0, 1.0)
                secondary = Body(secondary_name.capitalize(), 1.0, 1.0, _parent_input=primary)

            system = System(primary, secondary, distance_km)
            orbit._system = system
            if 1 <= lib_idx <= 5:
                orbit._libration_point = system.get_libration_point(lib_idx)
    except Exception:
        pass

    if "trajectory" in grp:
        orbit._trajectory = grp["trajectory"][()]
        orbit._times = grp["times"][()]
    else:
        orbit._trajectory = None
        orbit._times = None

    if "stability" in grp:
        sgrp = grp["stability"]
        orbit._stability_info = (
            sgrp["indices"][()],
            sgrp["eigvals"][()],
            sgrp["eigvecs"][()],
        )
    else:
        orbit._stability_info = None

    if getattr(orbit, "_libration_point", None) is None:
        orbit._libration_point = None
    if getattr(orbit, "_system", None) is None:
        orbit._system = None

    orbit._cached_dynsys = None

    return orbit


def save_periodic_orbit(
    orbit: "PeriodicOrbit",
    path: str | Path,
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialise *orbit* to *path* (HDF5)."""
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        _write_orbit_group(f, orbit, compression=compression, level=level)


def load_periodic_orbit_inplace(
    obj: "PeriodicOrbit",
    path: str | Path,
) -> None:
    """In-place deserialisation: patch *obj* with data stored at *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        cls_name = f.attrs.get("class", "<unknown>")
        if cls_name != obj.__class__.__name__:
            raise ValueError(
                f"Mismatch between file ({cls_name}) and object ({obj.__class__.__name__}) classes."
            )
        tmp = _read_orbit_group(f)
        obj.__dict__.update(tmp.__dict__)


def load_periodic_orbit(path: str | Path):
    """Return a *new* :class:`PeriodicOrbit` loaded from *path*."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        return _read_orbit_group(f)
