"""Input/output utilities for Body objects.

This module provides helpers to serialize and deserialize Body instances
using pickle, which preserves object relationships automatically.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.body import Body


def save_body(body: "Body", path: str | Path) -> None:
    """Serialize a Body to a pickle file.

    Parameters
    ----------
    body : :class:`~hiten.system.body.Body`
        Body instance to serialize.
    path : str or pathlib.Path
        File path where to save the body.
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(body, f)


def load_body(path: str | Path) -> "Body":
    """Load a Body from a pickle file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the body description.

    Returns
    -------
    :class:`~hiten.system.body.Body`
        The reconstructed Body instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_body_inplace(obj: "Body", path: str | Path) -> None:
    """Populate an existing Body object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.body.Body`
        Body instance to populate.
    path : str or pathlib.Path
        File path with serialized body.
    """
    tmp = load_body(path)
    obj.__dict__.update(tmp.__dict__)


