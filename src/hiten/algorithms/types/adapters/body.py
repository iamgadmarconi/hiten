"""Adapters supporting persistence for `hiten.system.body` objects."""

from __future__ import annotations

from pathlib import Path

from hiten.algorithms.types.adapters.base import _PersistenceAdapterMixin
from hiten.utils.io.body import load_body, load_body_inplace, save_body


class _BodyPersistenceAdapter(_PersistenceAdapterMixin):
    """Encapsulate IO helpers for bodies to simplify testing and substitution."""

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda body, path, **kw: save_body(body, Path(path), **kw),
            load_fn=lambda path, **kw: load_body(Path(path), **kw),
            load_inplace_fn=lambda body, path, **kw: load_body_inplace(body, Path(path), **kw),
        )
