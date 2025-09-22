from pathlib import Path
from typing import TYPE_CHECKING

import h5py

from hiten.system.body import Body
from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.base import System


HDF5_VERSION = "1.0"
"""HDF5 format version for System data."""


def save_system(system: "System", path: str | Path) -> None:
    """Serialize a System to an HDF5 file.

    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system instance to serialize.
    path : str or pathlib.Path
        File path where to save the system description.
    """

    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = system.__class__.__name__
        f.attrs["distance_km"] = float(system.distance)
        f.attrs["mu"] = float(system.mu)

        bodies_grp = f.create_group("bodies")

        p = bodies_grp.create_group("primary")
        p.attrs["name"] = system.primary.name
        p.attrs["mass"] = float(system.primary.mass)
        p.attrs["radius"] = float(system.primary.radius)
        p.attrs["color"] = system.primary.color

        s = bodies_grp.create_group("secondary")
        s.attrs["name"] = system.secondary.name
        s.attrs["mass"] = float(system.secondary.mass)
        s.attrs["radius"] = float(system.secondary.radius)
        s.attrs["color"] = system.secondary.color


def load_system(path: str | Path) -> "System":
    """Load a System from an HDF5 file.

    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the system description.

    Returns
    -------
    :class:`~hiten.system.base.System`
        The reconstructed system instance.
    """

    from hiten.system.base import System

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        # Basic sanity check (non-fatal if missing)
        cls_name = f.attrs.get("class", "System")
        if str(cls_name) not in ("System",):
            raise ValueError(f"Unsupported class '{cls_name}' in file {path}.")

        distance = float(f.attrs["distance_km"]) if "distance_km" in f.attrs else 1.0

        bodies = f["bodies"] if "bodies" in f else None
        if bodies is None or "primary" not in bodies or "secondary" not in bodies:
            raise ValueError("Invalid system file: missing bodies/primary or bodies/secondary group.")

        p = bodies["primary"]
        s = bodies["secondary"]

        primary = Body(
            str(p.attrs.get("name", "Primary")),
            float(p.attrs.get("mass", 1.0)),
            float(p.attrs.get("radius", 1.0)),
            str(p.attrs.get("color", "#000000")),
        )
        secondary = Body(
            str(s.attrs.get("name", "Secondary")),
            float(s.attrs.get("mass", 1.0)),
            float(s.attrs.get("radius", 1.0)),
            str(s.attrs.get("color", "#000000")),
            parent=primary,
        )

    return System(primary, secondary, distance)


def load_system_inplace(obj: "System", path: str | Path) -> None:
    """Populate an existing System object from an HDF5 file.

    Parameters
    ----------
    obj : :class:`~hiten.system.base.System`
        The System instance to populate.
    path : str or pathlib.Path
        File path containing the serialized system description.
    """
    tmp = load_system(path)
    obj.__dict__.update(tmp.__dict__)