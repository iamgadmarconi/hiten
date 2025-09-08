"""Common input/output utilities for the hiten package.

This module provides shared utilities for file and directory operations,
HDF5 dataset management, and other common I/O tasks used throughout
the hiten package.

Notes
-----
All functions are designed to be robust and handle common edge cases
in file operations and data serialization.
"""

import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _ensure_dir(path: str | os.PathLike) -> None:
    """Create directory and any parent directories if they do not exist.

    Parameters
    ----------
    path : str or os.PathLike
        Directory path that should exist after this call. Accepts any object
        supported by pathlib.Path.
        
    Notes
    -----
    This function is safe to call multiple times - it will not raise an error
    if the directory already exists. Parent directories are created recursively
    as needed.
    
    Examples
    --------
    >>> _ensure_dir("path/to/directory")
    >>> _ensure_dir(Path("another/path"))
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_dataset(
    group: h5py.Group,
    name: str,
    data: Optional[np.ndarray],
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Write data into HDF5 group under dataset name if data is not None.

    Parameters
    ----------
    group : h5py.Group
        Open h5py group or file handle.
    name : str
        Name of the dataset to create.
    data : numpy.ndarray or None
        Array to store. If None, the function is a no-op.
    compression : str, default "gzip"
        Compression backend passed to h5py.Group.create_dataset.
    level : int, default 4
        Compression level (0-9, higher means better compression).
        
    Notes
    -----
    This function safely handles None data by performing no operation.
    Only numpy arrays are written to the HDF5 group.
    
    Examples
    --------
    >>> import h5py
    >>> import numpy as np
    >>> with h5py.File("test.h5", "w") as f:
    ...     _write_dataset(f, "my_data", np.array([1, 2, 3]))
    >>> with h5py.File("test.h5", "w") as f:
    ...     _write_dataset(f, "empty", None)  # No-op
    """
    if data is None:
        return

    if isinstance(data, np.ndarray):
        group.create_dataset(name, data=data, compression=compression, compression_opts=level)
