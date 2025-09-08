"""Input/output utilities for Hamiltonian data.

This module provides functions for serializing and deserializing Hamiltonian
objects to/from HDF5 files. It includes utilities for saving and loading
Hamiltonian polynomial coefficients and metadata.

Notes
-----
All data is stored in HDF5 format with version tracking. The module supports
compression and handles class resolution for different Hamiltonian types.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Type

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset

if TYPE_CHECKING:
    from hiten.system.hamiltonians.base import Hamiltonian

HDF5_VERSION = "1.0"
"""HDF5 format version for Hamiltonian data."""

_HAM_CLASSES: Dict[str, Type["Hamiltonian"]] = {}
"""Cache for resolved Hamiltonian classes."""


def _resolve_class(class_name: str) -> Type["Hamiltonian"]:
    """Resolve a Hamiltonian class by name.
    
    Parameters
    ----------
    class_name : str
        Name of the Hamiltonian class to resolve.
        
    Returns
    -------
    Type[:class:`~hiten.system.hamiltonians.base.Hamiltonian`]
        The resolved Hamiltonian class.
        
    Notes
    -----
    This function first checks the class cache, then attempts to import
    the class from known modules. If the class cannot be found, it returns
    the base Hamiltonian class as a fallback.
    
    Examples
    --------
    >>> cls = _resolve_class("Hamiltonian")
    >>> cls.__name__
    'Hamiltonian'
    """
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
    """Save a Hamiltonian object to an HDF5 file.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonians.base.Hamiltonian`
        The Hamiltonian object to serialize.
    path : str or pathlib.Path
        File path where to save the Hamiltonian data.
    compression : str, default "gzip"
        Compression algorithm to use for HDF5 files.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    The function saves the Hamiltonian's polynomial coefficients, degree,
    number of degrees of freedom, and name to an HDF5 file. Empty polynomial
    blocks are skipped to save space.
    
    Examples
    --------
    >>> from hiten.system.hamiltonians.base import Hamiltonian
    >>> import numpy as np
    >>> ham = Hamiltonian([np.array([1.0, 2.0])], degree=2, ndof=3, name="test")
    >>> save_hamiltonian(ham, "my_hamiltonian.h5")
    """
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


def load_hamiltonian(path: str | Path, **kwargs) -> "Hamiltonian":
    """Load a Hamiltonian object from an HDF5 file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the Hamiltonian data.
    **kwargs
        Additional keyword arguments (currently unused).
        
    Returns
    -------
    :class:`~hiten.system.hamiltonians.base.Hamiltonian`
        The reconstructed Hamiltonian object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Notes
    -----
    The function reconstructs the Hamiltonian from serialized data, including
    polynomial coefficients, degree, number of degrees of freedom, and name.
    The class type is resolved using the stored class name.
    
    Examples
    --------
    >>> ham = load_hamiltonian("my_hamiltonian.h5")
    >>> print(f"Loaded Hamiltonian: {ham.name}")
    """
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