from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from hiten.utils.io.common import _ensure_dir, _write_dataset
from hiten.utils.io.orbits import (_read_orbit_group,
                                              _write_orbit_group)

if TYPE_CHECKING:
    from hiten.system.manifold import Manifold, ManifoldResult

__all__ = ["save_manifold", "load_manifold"]

HDF5_VERSION = "1.0"

def save_manifold(
    manifold: "Manifold",
    path: str | Path,
    *,
    compression: str = "gzip",
    level: int = 4,
) -> None:
    """Serialise *manifold* to *path* (HDF5)."""
    path = Path(path)
    _ensure_dir(path.parent)

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = HDF5_VERSION
        f.attrs["class"] = manifold.__class__.__name__
        f.attrs["stable"] = bool(manifold._stable == 1)
        f.attrs["direction"] = "positive" if manifold._direction == 1 else "negative"

        ggrp = f.create_group("generating_orbit")
        _write_orbit_group(ggrp, manifold._generating_orbit, compression=compression, level=level)

        if manifold._manifold_result is not None:
            mr: "ManifoldResult" = manifold._manifold_result
            rgrp = f.create_group("result")
            _write_dataset(rgrp, "ysos", np.asarray(mr.ysos))
            _write_dataset(rgrp, "dysos", np.asarray(mr.dysos))
            rgrp.attrs["_successes"] = int(mr._successes)
            rgrp.attrs["_attempts"] = int(mr._attempts)

            tgrp = rgrp.create_group("trajectories")
            for i, (states, times) in enumerate(zip(mr.states_list, mr.times_list)):
                sub = tgrp.create_group(str(i))
                _write_dataset(sub, "states", np.asarray(states), compression=compression, level=level)
                _write_dataset(sub, "times", np.asarray(times), compression=compression, level=level)


def load_manifold(path: str | Path):
    """Return a new :class:`Manifold` instance reconstructed from *path*."""
    from hiten.system.manifold import Manifold, ManifoldResult

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        stable_flag = bool(f.attrs.get("stable", True))
        direction_str = f.attrs.get("direction", "positive")

        ggrp = f["generating_orbit"]
        gen_orbit = _read_orbit_group(ggrp)

        man = Manifold(
            generating_orbit=gen_orbit,
            stable=stable_flag,
            direction=direction_str,
        )

        if "result" in f:
            rgrp = f["result"]
            ysos = rgrp["ysos"][()] if "ysos" in rgrp else []
            dysos = rgrp["dysos"][()] if "dysos" in rgrp else []
            succ = int(rgrp.attrs.get("_successes", 0))
            attm = int(rgrp.attrs.get("_attempts", 0))

            states_list, times_list = [], []
            if "trajectories" in rgrp:
                tgrp = rgrp["trajectories"]
                for key in tgrp.keys():
                    sub = tgrp[key]
                    states_list.append(sub["states"][()])
                    times_list.append(sub["times"][()])

            man._manifold_result = ManifoldResult(
                ysos=list(ysos),
                dysos=list(dysos),
                states_list=states_list,
                times_list=times_list,
                _successes=succ,
                _attempts=attm,
            )

    return man
