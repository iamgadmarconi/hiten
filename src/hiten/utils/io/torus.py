"""Input/output utilities for invariant torus data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:  # pragma: no cover - import-guard for typing
    from hiten.system.torus import InvariantTori, Torus


def save_torus(torus: "InvariantTori", path: str | Path, **kwargs) -> None:
    """Serialize an invariant torus to a pickle file."""

    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(torus, f)


def load_torus(path: str | Path) -> "InvariantTori":
    """Load an invariant torus from a pickle file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_torus_inplace(obj: "InvariantTori", path: str | Path) -> None:
    """Populate an existing invariant torus from a pickle file."""

    tmp = load_torus(path)
    obj.__dict__.update(tmp.__dict__)

