import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _ensure_dir(path: str | os.PathLike) -> None:
    """Create *path* and any parent directories if they do not exist.

    Parameters
    ----------
    path
        Directory path that should exist after this call.  Accepts any object
        supported by :class:`pathlib.Path`.
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
    """Write *data* into *group* under dataset *name* if *data* is not None.

    Parameters
    ----------
    group
        Open h5py group or file handle.
    name
        Name of the dataset to create.
    data
        Array to store.  If *None* the function is a no-op.
    compression, level
        Compression backend and level passed straight to
        :meth:`h5py.Group.create_dataset`.
    """
    if data is None:
        return

    if isinstance(data, np.ndarray):
        group.create_dataset(name, data=data, compression=compression, compression_opts=level)
