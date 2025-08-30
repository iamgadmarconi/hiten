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


def save_poincare_map(
    pmap: "CenterManifoldMap",
    path: str | Path,
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialise *pmap* (all cached sections) to *path*."""

    if not pmap._sections:
        pmap.compute()

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
    """Populate *obj* with data stored at *path*."""

    from hiten.algorithms.poincare.centermanifold.config import _CenterManifoldMapConfig
    from hiten.algorithms.poincare.core.base import _Section

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
            obj._sections[str(coord)] = _Section(pts, sts, labels)

        obj._section = obj._sections[obj.config.section_coord]


def load_poincare_map(path: str | Path, cm: "CenterManifold"):
    """Return a new :class:`CenterManifoldMap` loaded from *path*."""
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        energy = float(f.attrs["energy"])

    pmap = CenterManifoldMap(cm, energy)
    load_poincare_map_inplace(pmap, path)
    return pmap
