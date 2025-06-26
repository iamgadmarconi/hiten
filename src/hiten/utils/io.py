from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

if TYPE_CHECKING:
    from hiten.algorithms.poincare.base import _PoincareMap
    from hiten.system.center import CenterManifold
    from hiten.system.manifold import Manifold, ManifoldResult
    from hiten.system.orbits.base import PeriodicOrbit


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_dataset(group: h5py.Group, name: str, data: Optional[np.ndarray], *, compression: str = "gzip", level: int = 4) -> None:
    if data is None:
        return
    if isinstance(data, np.ndarray):
        group.create_dataset(name, data=data, compression=compression, compression_opts=level)


def _save_periodic_orbit(orbit: "PeriodicOrbit", filepath: str, *, compression: str = "gzip", level: int = 4) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with h5py.File(filepath, "w") as f:
        f.attrs["class"] = orbit.__class__.__name__
        f.attrs["format_version"] = "1.0"
        f.attrs["family"] = orbit.family
        f.attrs["mu"] = float(orbit.mu)
        f.attrs["period"] = -1.0 if orbit.period is None else float(orbit.period)

        _write_dataset(f, "initial_state", np.asarray(orbit._initial_state))

        # ------------------------------------------------------------------
        # Lightweight metadata needed to reconstruct the dynamical context
        # (System + LibrationPoint) without serialising numba objects.
        # ------------------------------------------------------------------
        if getattr(orbit, "_system", None) is not None:
            f.attrs["primary"] = orbit._system.primary.name
            f.attrs["secondary"] = orbit._system.secondary.name
            f.attrs["distance_km"] = float(orbit._system.distance)

        if getattr(orbit, "libration_point", None) is not None:
            f.attrs["libration_index"] = int(orbit.libration_point.idx)

        if orbit._trajectory is not None:
            _write_dataset(f, "trajectory", np.asarray(orbit._trajectory), compression=compression, level=level)
            _write_dataset(f, "times", np.asarray(orbit._times), compression=compression, level=level)

        if orbit._stability_info is not None:
            grp = f.create_group("stability")
            indices, eigvals, eigvecs = orbit._stability_info
            _write_dataset(grp, "indices", np.asarray(indices))
            _write_dataset(grp, "eigvals", np.asarray(eigvals))
            _write_dataset(grp, "eigvecs", np.asarray(eigvecs))


def _load_periodic_orbit_inplace(obj: "PeriodicOrbit", filepath: str) -> None:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Orbit file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        # Sanity check - only warn on mismatch to allow subclass <-> base use
        cls_name = f.attrs.get("class", "<unknown>")
        if cls_name != obj.__class__.__name__:
            raise ValueError(f"Mismatch between file and object class: {cls_name} != {obj.__class__.__name__}")
        obj._family = str(f.attrs.get("family", obj._family))
        # Retrieve stored mass ratio (mu) and ensure consistency with current object.
        stored_mu = float(f.attrs["mu"])
        # If the object already has a _mu attribute, keep it unless it is uninitialised or differs significantly
        if not hasattr(obj, "_mu") or obj._mu is None:
            obj._mu = stored_mu
        elif abs(obj._mu - stored_mu) > 1e-12:
            import warnings
            warnings.warn(
                f"Loaded mu ({stored_mu}) differs from object mu ({obj._mu}); keeping existing value.")

        period_val = float(f.attrs["period"])
        obj.period = None if period_val < 0 else period_val

        obj._initial_state = f["initial_state"][()]

        try:
            primary_name = f.attrs.get("primary", None)
            secondary_name = f.attrs.get("secondary", None)
            distance_km = float(f.attrs.get("distance_km", -1.0))
            lib_idx = int(f.attrs.get("libration_index", -1))

            if primary_name and secondary_name and distance_km > 0:
                from hiten.system.base import System
                from hiten.system.body import Body
                from hiten.utils.constants import Constants

                p_key = str(primary_name).lower()
                s_key = str(secondary_name).lower()

                try:
                    primary = Body(primary_name.capitalize(), Constants.get_mass(p_key), Constants.get_radius(p_key))
                    secondary = Body(secondary_name.capitalize(), Constants.get_mass(s_key), Constants.get_radius(s_key), _parent_input=primary)
                except Exception:
                    # Fall back to generic body parameters if constants missing
                    primary = Body(primary_name.capitalize(), 1.0, 1.0)
                    secondary = Body(secondary_name.capitalize(), 1.0, 1.0, _parent_input=primary)

                system = System(primary, secondary, distance_km)
                obj._system = system
                if 1 <= lib_idx <= 5:
                    obj._libration_point = system.get_libration_point(lib_idx)
        except Exception as exc:
            # Silent failure - context reconstruction is best effort only.
            import warnings
            warnings.warn(f"Could not reconstruct System from metadata: {exc}")

        if "trajectory" in f and "times" in f:
            obj._trajectory = f["trajectory"][()]
            obj._times = f["times"][()]
        else:
            obj._trajectory = None
            obj._times = None

        if "stability" in f:
            g = f["stability"]
            indices = g["indices"][()]
            eigvals = g["eigvals"][()]
            eigvecs = g["eigvecs"][()]
            obj._stability_info = (indices, eigvals, eigvecs)
        else:
            obj._stability_info = None

    if getattr(obj, "_libration_point", None) is None:
        obj._libration_point = None
    if getattr(obj, "_system", None) is None:
        obj._system = None

    obj._cached_dynsys = None


def _load_periodic_orbit(filepath: str) -> "PeriodicOrbit":
    """Loads a periodic orbit from a file, creating a new object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Orbit file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        return _read_periodic_orbit_group(f)


def _write_periodic_orbit_group(grp: h5py.Group, orbit: "PeriodicOrbit", *, compression: str = "gzip", level: int = 4) -> None:
    grp.attrs["class"] = orbit.__class__.__name__
    grp.attrs["family"] = orbit.family
    grp.attrs["mu"] = float(orbit.mu)
    grp.attrs["period"] = -1.0 if orbit.period is None else float(orbit.period)

    # -- Essential state --
    _write_dataset(grp, "initial_state", np.asarray(orbit._initial_state))

    # Context metadata (see _save_periodic_orbit above)
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


def _save_manifold(manifold: "Manifold", filepath: str, *, compression: str = "gzip", level: int = 4) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

    with h5py.File(filepath, "w") as f:
        f.attrs["class"] = manifold.__class__.__name__
        f.attrs["format_version"] = "1.0"
        f.attrs["stable"] = bool(manifold._stable == 1)
        f.attrs["direction"] = "positive" if manifold._direction == 1 else "negative"
        f.attrs["method"] = manifold._method
        f.attrs["order"] = manifold._order

        ggrp = f.create_group("generating_orbit")
        _write_periodic_orbit_group(ggrp, manifold._generating_orbit, compression=compression, level=level)

        if manifold._manifold_result is not None:
            mr: ManifoldResult = manifold._manifold_result  
            rgrp = f.create_group("result")
            _write_dataset(rgrp, "ysos", np.asarray(mr.ysos))
            _write_dataset(rgrp, "dysos", np.asarray(mr.dysos))
            rgrp.attrs["_successes"] = mr._successes
            rgrp.attrs["_attempts"] = mr._attempts
            # Variable-length list of trajectories
            tgrp = rgrp.create_group("trajectories")
            for i, (states, times) in enumerate(zip(mr.states_list, mr.times_list)):
                sub = tgrp.create_group(str(i))
                _write_dataset(sub, "states", np.asarray(states), compression=compression, level=level)
                _write_dataset(sub, "times", np.asarray(times), compression=compression, level=level)


def _read_periodic_orbit_group(grp: h5py.Group) -> "PeriodicOrbit":
    from hiten.system.orbits.base import GenericOrbit

    cls_name = grp.attrs.get("class", "GenericOrbit")
    orbit_cls = None
    for mod_name in [
        "hiten.system.orbits.halo",
        "hiten.system.orbits.lyapunov",
        "hiten.system.orbits.base",  # GenericOrbit
    ]:
        try:
            mod = import_module(mod_name)
            if hasattr(mod, cls_name):
                orbit_cls = getattr(mod, cls_name)
                break
        except ModuleNotFoundError:
            continue

    if orbit_cls is None:
        orbit_cls = GenericOrbit

    orbit = orbit_cls.__new__(orbit_cls)

    # Patch basic attributes
    orbit._family = str(grp.attrs.get("family", "generic"))
    stored_mu = float(grp.attrs["mu"])
    orbit._mu = stored_mu
    period_val = float(grp.attrs["period"])
    orbit.period = None if period_val < 0 else period_val

    orbit._initial_state = grp["initial_state"][()]
    # Context metadata reconstruction (primary, secondary, distance, libration index)
    try:
        primary_name = grp.attrs.get("primary", None)
        secondary_name = grp.attrs.get("secondary", None)
        distance_km = float(grp.attrs.get("distance_km", -1.0))
        lib_idx = int(grp.attrs.get("libration_index", -1))

        if primary_name and secondary_name and distance_km > 0:
            from hiten.system.base import System
            from hiten.system.body import Body
            from hiten.utils.constants import Constants

            p_key = str(primary_name).lower(); s_key = str(secondary_name).lower()
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
        indices = sgrp["indices"][()]
        eigvals = sgrp["eigvals"][()]
        eigvecs = sgrp["eigvecs"][()]
        orbit._stability_info = (indices, eigvals, eigvecs)
    else:
        orbit._stability_info = None

    if getattr(orbit, "_libration_point", None) is None:
        orbit._libration_point = None  
    if getattr(orbit, "_system", None) is None:
        orbit._system = None  

    orbit._cached_dynsys = None

    return orbit


def _load_manifold(filepath: str) -> "Manifold":
    from hiten.system.manifold import Manifold, ManifoldResult

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Manifold file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        stable_flag = bool(f.attrs.get("stable", True))
        direction_str = f.attrs.get("direction", "positive")
        method = f.attrs.get("method", "scipy")
        order = int(f.attrs.get("order", 6))

        ggrp = f["generating_orbit"]
        gen_orbit: PeriodicOrbit = _read_periodic_orbit_group(ggrp)

        man = Manifold(
            generating_orbit=gen_orbit,
            stable=stable_flag,
            direction=direction_str,
            method=method,
            order=order,
        )

        if "result" in f:
            rgrp = f["result"]
            ysos = rgrp["ysos"][()] if "ysos" in rgrp else []
            dysos = rgrp["dysos"][()] if "dysos" in rgrp else []
            succ = int(rgrp.attrs.get("_successes", 0))
            attm = int(rgrp.attrs.get("_attempts", 0))
            states_list, times_list = [], []
            tgrp = rgrp.get("trajectories")
            if tgrp is not None:
                for key in tgrp.keys():
                    sub = tgrp[key]
                    states_list.append(sub["states"][()])
                    times_list.append(sub["times"][()])
            man._manifold_result = ManifoldResult(
                ysos=list(ysos),
                dysos=list(dysos),
                states_list=states_list,
                times_list=times_list,
                _successes=succ,
                _attempts=attm,
            )

    return man


def _save_poincare_map(pmap: "_PoincareMap", filepath: str, *, compression: str = "gzip", level: int = 4) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

    with h5py.File(filepath, "w") as f:
        f.attrs["class"] = pmap.__class__.__name__
        f.attrs["format_version"] = "1.0"
        f.attrs["energy"] = float(pmap.energy)
        f.attrs["config_json"] = json.dumps(asdict(pmap.config))

        if pmap._section is not None:
            _write_dataset(f, "points", np.asarray(pmap._section.points), compression=compression, level=level)
            f.attrs["labels_json"] = json.dumps(list(pmap._section.labels))

        if pmap._grid is not None:
            _write_dataset(f, "grid", np.asarray(pmap._grid), compression=compression, level=level)
            f.attrs["grid_labels_json"] = json.dumps(list(pmap._section.labels))


def _load_poincare_map_inplace(obj: "_PoincareMap", filepath: str) -> None:
    from hiten.algorithms.poincare.base import _PoincareMapConfig
    from hiten.algorithms.poincare.map import _PoincareSection

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Poincaré-map file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        obj.energy = float(f.attrs["energy"])
        cfg_json = f.attrs.get("config_json", "{}")
        try:
            cfg_dict = json.loads(cfg_json)
            obj.config = _PoincareMapConfig(**cfg_dict)
        except Exception as exc:
            obj.config = _PoincareMapConfig()

        obj._use_symplectic = obj.config.method.lower() == "symplectic"

        if "points" in f:
            pts = f["points"][()]
            labels_json = f.attrs.get("labels_json")
            labels = tuple(json.loads(labels_json)) if labels_json else ("q2", "p2")
            obj._section = _PoincareSection(pts, labels)
        else:
            obj._section = None

        if "grid" in f:
            obj._grid = f["grid"][()]
            grid_labels_json = f.attrs.get("grid_labels_json")
            grid_labels = tuple(json.loads(grid_labels_json)) if grid_labels_json else ("q2", "p2")
            obj._grid_labels = grid_labels
        else:
            obj._grid = None


def _load_poincare_map(filepath: str, cm: "CenterManifold") -> "_PoincareMap":
    from hiten.algorithms.poincare.base import _PoincareMap

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Poincaré-map file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        energy = float(f.attrs["energy"])
        pmap = _PoincareMap(cm, energy)
        _load_poincare_map_inplace(pmap, filepath)
    return pmap


def _save_center_manifold(cm: "CenterManifold", dir_path: str | os.PathLike, *, compression: str = "gzip", level: int = 4) -> None:
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    main_file = dir_path / "manifold.h5"
    with h5py.File(main_file, "w") as f:
        f.attrs["class"] = cm.__class__.__name__
        f.attrs["format_version"] = "1.0"
        f.attrs["max_degree"] = int(cm._max_degree)

        point_blob = pickle.dumps(cm._point, protocol=pickle.HIGHEST_PROTOCOL)
        f.create_dataset("point_pickle", data=np.frombuffer(point_blob, dtype=np.uint8))

        clean_cache = cm.__class__._sanitize_cache(cm._cache)
        cache_blob = pickle.dumps(clean_cache, protocol=pickle.HIGHEST_PROTOCOL)
        f.create_dataset("cache_pickle", data=np.frombuffer(cache_blob, dtype=np.uint8), compression=compression, compression_opts=level)

    maps_dir = dir_path / "maps"
    keys_file = dir_path / "poincare_maps_keys.json"
    if cm._poincare_maps:
        maps_dir.mkdir(exist_ok=True)
        map_keys = []
        for i, (key, pmap) in enumerate(cm._poincare_maps.items()):
            filename = f"map_{i}.h5"
            pmap.save(maps_dir / filename)
            map_keys.append({"key": list(key), "file": filename})
        with open(keys_file, "w") as fh:
            json.dump(map_keys, fh)
    else:
        if keys_file.exists():
            keys_file.unlink()


def _load_center_manifold(dir_path: str | os.PathLike) -> "CenterManifold":
    from pathlib import Path

    from hiten.algorithms.poincare.base import _PoincareMap
    from hiten.algorithms.polynomial.base import (
        _create_encode_dict_from_clmo, _init_index_tables)
    from hiten.system.center import CenterManifold

    dir_path = Path(dir_path)
    main_file = dir_path / "manifold.h5"
    if not main_file.exists():
        raise FileNotFoundError(main_file)

    with h5py.File(main_file, "r") as f:
        max_deg = int(f.attrs["max_degree"])
        point_blob = f["point_pickle"][()]
        cache_blob = f["cache_pickle"][()]
        point = pickle.loads(point_blob.tobytes())
        cache = pickle.loads(cache_blob.tobytes())

    # Build blank instance and patch attributes
    cm = CenterManifold.__new__(CenterManifold)  
    cm._point = point
    cm._max_degree = max_deg
    cm._psi, cm._clmo = _init_index_tables(max_deg)  
    cm._encode_dict_list = _create_encode_dict_from_clmo(cm._clmo)  
    cm._cache = cache
    cm._poincare_maps = {}

    keys_file = dir_path / "poincare_maps_keys.json"
    maps_dir = dir_path / "maps"
    if keys_file.exists():
        with open(keys_file, "r") as fh:
            map_keys_info = json.load(fh)
        for info in map_keys_info:
            key_list = info["key"]
            energy = key_list[0]
            config_tuple = tuple(tuple(item) for item in key_list[1])
            original_key = (energy, config_tuple)
            pmap = _PoincareMap(cm, energy)  # dummy to allocate
            pmap.load(maps_dir / info["file"])
            pmap.cm = cm
            cm._poincare_maps[original_key] = pmap

    return cm
