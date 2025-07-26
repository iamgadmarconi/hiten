from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.system.hamiltonians.base import Hamiltonian

HDF5_VERSION = "1.0"

_HAM_CLASSES: Dict[str, Type["Hamiltonian"]] = {}


def _resolve_class(class_name: str):
    if class_name in _HAM_CLASSES:
        return _HAM_CLASSES[class_name]
    # fallback, import base module
    from importlib import import_module

    for mod_name in (
        "hiten.system.hamiltonians.base",
    ):
        try:
            mod = import_module(mod_name)
        except ModuleNotFoundError:
            continue
        if hasattr(mod, class_name):
            cls = getattr(mod, class_name)
            _HAM_CLASSES[class_name] = cls
            return cls
    from hiten.system.hamiltonians.base import Hamiltonian as _Default

    return _Default


def save_hamiltonian(ham: "Hamiltonian", path: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = ham.__class__.__name__
        f.attrs["degree"] = int(ham.degree)
        f.attrs["ndof"] = int(ham.ndof)
        f.attrs["name"] = ham.name

        grp = f.create_group("poly")
        for idx, block in enumerate(ham.poly_H):
            # Skip empty blocks to save space
            if block.size == 0:
                continue
            _write_dataset(grp, str(idx), block, compression=compression, level=level)


def load_hamiltonian(path: str | Path, **kwargs):
    from hiten.system.hamiltonians.base import Hamiltonian

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        cls_name = f.attrs.get("class", "Hamiltonian")
        degree = int(f.attrs["degree"])
        ndof = int(f.attrs.get("ndof", 3))
        name = str(f.attrs.get("name", cls_name))

        cls = _resolve_class(cls_name)

        poly_grp = f["poly"]
        max_idx = max(int(k) for k in poly_grp.keys()) if poly_grp.keys() else -1
        poly_H = [np.zeros((0,)) for _ in range(max_idx + 1)]
        for key in poly_grp.keys():
            idx = int(key)
            poly_H[idx] = poly_grp[key][()]

    ham_obj: Hamiltonian = cls(poly_H, degree, ndof=ndof, name=name)
    return ham_obj 