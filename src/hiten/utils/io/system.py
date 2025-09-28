from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.base import System


def save_system(system: "System", path: str | Path) -> None:
    """Serialize a System to a pickle file.

    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system instance to serialize.
    path : str or pathlib.Path
        File path where to save the system description.
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(system, f)


def load_system(path: str | Path) -> "System":
    """Load a System from a pickle file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the system description.

    Returns
    -------
    :class:`~hiten.system.base.System`
        The reconstructed system instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def load_system_inplace(obj: "System", path: str | Path) -> None:
    """Populate an existing System object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.base.System`
        The System instance to populate.
    path : str or pathlib.Path
        File path containing the serialized system description.
    """
    tmp = load_system(path)
    obj.__dict__.update(tmp.__dict__)