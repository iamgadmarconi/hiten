"""Input/output utilities for periodic orbit data.

This module provides functions for serializing and deserializing periodic orbit
objects and their associated data to/from HDF5 files. It includes utilities
for saving and loading orbits, their trajectories, stability information, and
system context.

Notes
-----
All data is stored in HDF5 format with version tracking. The module supports
compression and handles both in-place loading and new object creation. Orbit
classes are automatically registered for deserialization.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit

HDF5_VERSION = "1.0"
"""HDF5 format version for periodic orbit data."""


def _write_orbit_group(
    grp: h5py.Group,
    orbit: "PeriodicOrbit",
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialize periodic orbit data into HDF5 group.

    Parameters
    ----------
    grp : h5py.Group
        The HDF5 group to write orbit data to.
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The periodic orbit object to serialize.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The HDF5 group may be the root file object or a subgroup - the helper does
    not make any assumptions about hierarchy, making it re-usable for nested
    structures (e.g. manifolds that embed a generating orbit).
    
    The function serializes orbit attributes, trajectory data, stability
    information, and system context if available.
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
"""Registry of orbit classes for deserialization."""


def register_orbit_class(cls):
    """Decorator that registers orbit class for deserialization.
    
    Parameters
    ----------
    cls : type
        The orbit class to register.
        
    Returns
    -------
    type
        The same class (for use as decorator).
        
    Notes
    -----
    This decorator registers orbit classes so they can be automatically
    deserialized from HDF5 files. Classes are identified by their name
    and stored in the global registry.
    
    Examples
    --------
    >>> @register_orbit_class
    ... class MyOrbit(PeriodicOrbit):
    ...     pass
    """
    _ORBIT_CLASSES[cls.__name__] = cls
    return cls


def _read_orbit_group(grp: h5py.Group) -> "PeriodicOrbit":
    """Reconstruct and return a PeriodicOrbit instance from HDF5 group.
    
    Parameters
    ----------
    grp : h5py.Group
        The HDF5 group containing orbit data.
        
    Returns
    -------
    :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The reconstructed periodic orbit object.
        
    Raises
    ------
    ImportError
        If the orbit class cannot be found or imported.
        
    Notes
    -----
    This function reconstructs a periodic orbit from serialized data,
    including trajectory, stability information, and system context.
    The orbit class is determined from the stored class name and
    automatically imported if not already registered.
    """
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
    """Serialize periodic orbit to HDF5 file.

    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The periodic orbit object to serialize.
    path : str or pathlib.Path
        File path where to save the orbit data.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The function saves the orbit's attributes, trajectory data, stability
    information, and system context to an HDF5 file.
    
    Examples
    --------
    >>> from hiten.system import System
    >>> from hiten.system.orbits.halo import HaloOrbit
    >>> system = System.from_bodies("earth", "moon")
    >>> L2 = system.get_libration_point(2)
    >>> orbit = L2.create_orbit('halo', amplitude_z=0.3)
    >>> save_periodic_orbit(orbit, "my_orbit.h5")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        _write_orbit_group(f, orbit, compression=compression, level=level)


def load_periodic_orbit_inplace(
    obj: "PeriodicOrbit",
    path: str | Path,
) -> None:
    """Load periodic orbit data into existing object.
    
    Parameters
    ----------
    obj : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        The periodic orbit object to populate with data.
    path : str or pathlib.Path
        File path containing the orbit data.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file contains data for a different orbit class.
        
    Notes
    -----
    This function loads data into an existing periodic orbit object,
    replacing all its attributes with the loaded data. The object's
    class must match the class stored in the file.
    
    Examples
    --------
    >>> from hiten.system.orbits.halo import HaloOrbit
    >>> orbit = HaloOrbit()
    >>> load_periodic_orbit_inplace(orbit, "my_orbit.h5")
    """
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


def load_periodic_orbit(path: str | Path) -> "PeriodicOrbit":
    """Load a periodic orbit from an HDF5 file.
    
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
    ImportError
        If the orbit class cannot be found or imported.
        
    Notes
    -----
    This function creates a new periodic orbit object and loads data into it.
    The orbit class is determined from the stored class name and
    automatically imported if not already registered.
    
    Examples
    --------
    >>> orbit = load_periodic_orbit("my_orbit.h5")
    >>> print(f"Loaded orbit: {orbit.family}, period={orbit.period}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        return _read_orbit_group(f)
