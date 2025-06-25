"""
hiten.system.family
=====================

Light-weight container that groups a *family* of periodic orbits obtained via a
continuation engine.

The class keeps together:
    • list of `PeriodicOrbit` instances in ascending continuation order,
    • the continuation parameter values (float array),
    • a short string identifying that parameter (for labelling / DataFrame).

It offers convenience helpers for iteration, random access, conversion to a
`pandas.DataFrame`, and basic serialisation to an HDF5 file leveraging the
existing utilities in :pymod:`hiten.utils.io`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List

import h5py
import numpy as np
import pandas as pd

from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io import (_ensure_dir, _read_periodic_orbit_group,
                            _write_periodic_orbit_group)
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_orbit_family


@dataclass
class OrbitFamily:
    """Container for an ordered family of periodic orbits."""

    orbits: List[PeriodicOrbit] = field(default_factory=list)
    parameter_name: str = "param"
    parameter_values: np.ndarray | None = None  # one value per orbit

    def __post_init__(self) -> None:
        if self.parameter_values is None:
            self.parameter_values = np.full(len(self.orbits), np.nan, dtype=float)
        else:
            self.parameter_values = np.asarray(self.parameter_values, dtype=float)
            if self.parameter_values.shape[0] != len(self.orbits):
                raise ValueError("Length of parameter_values must match number of orbits")

    @classmethod
    def from_engine(cls, engine, parameter_name: str | None = None):
        """Build an `OrbitFamily` from a finished continuation engine."""
        if parameter_name is None:
            parameter_name = "param"
        return cls(list(engine.family), parameter_name, np.asarray(engine.parameter_values))

    def __len__(self) -> int:
        return len(self.orbits)

    def __iter__(self) -> Iterator[PeriodicOrbit]:
        return iter(self.orbits)

    def __getitem__(self, idx):
        return self.orbits[idx]

    @property
    def periods(self) -> np.ndarray:
        """Array of orbit periods (NaN if not available)."""
        return np.array([o.period if o.period is not None else np.nan for o in self.orbits])

    @property
    def jacobi_constants(self) -> np.ndarray:
        return np.array([o.jacobi_constant for o in self.orbits])
    
    def propagate(self, **kwargs) -> None:
        """Propagate all orbits in the family."""
        for orb in self.orbits:
            orb.propagate(**kwargs)

    def save(self, filepath: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
        """Save the family to an HDF5 file (each orbit as a subgroup)."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.attrs["class"] = "OrbitFamily"
            f.attrs["format_version"] = "1.0"
            f.attrs["parameter_name"] = self.parameter_name
            f.create_dataset("parameter_values", data=self.parameter_values)

            grp = f.create_group("orbits")
            for i, orbit in enumerate(self.orbits):
                sub = grp.create_group(str(i))
                _write_periodic_orbit_group(sub, orbit, compression=compression, level=level)

        logger.info(f"Family saved to {filepath}")

    def to_csv(self, filepath: str, **kwargs) -> None:
        r"""
        Export the contents of the orbit family to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination CSV file.
        **kwargs
            Extra keyword arguments passed to :py:meth:`PeriodicOrbit.propagate`.

        Raises
        ------
        ValueError
            If no trajectory data is available to export.
        """
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))

        data = []
        for idx, orbit in enumerate(self.orbits):
            if orbit.trajectory is None or orbit.times is None:
                orbit.propagate(**kwargs)
            for t, state in zip(orbit.times, orbit.trajectory):
                data.append([idx, self.parameter_values[idx], t, *state])

        if not data:
            raise ValueError("No trajectory data available to export.")

        columns = [
            "orbit_id", self.parameter_name, "time",
            "x", "y", "z", "vx", "vy", "vz",
        ]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filepath, index=False)
        logger.info(f"Orbit family trajectories successfully exported to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path):
        """Load a family previously saved with `save_hdf`."""
        with h5py.File(filepath, "r") as f:
            if str(f.attrs.get("class", "")) != "OrbitFamily":
                raise ValueError("File does not contain an OrbitFamily object")
            param_name = str(f.attrs["parameter_name"])
            param_vals = f["parameter_values"][()]
            orbits: List[PeriodicOrbit] = []
            for key in sorted(f["orbits"], key=lambda s: int(s)):
                grp = f["orbits"][key]
                orbits.append(_read_periodic_orbit_group(grp))
        return cls(orbits, param_name, param_vals)

    def __repr__(self) -> str:
        return f"OrbitFamily(n_orbits={len(self)}, parameter='{self.parameter_name}')"

    def plot(self, *, dark_mode: bool = True, save: bool = False, filepath: str = "orbit_family.svg", **kwargs):
        """Visualise the family trajectories in rotating frame."""

        states_list = []
        times_list = []
        for orb in self.orbits:
            if orb.trajectory is None or orb.times is None:
                err = "Orbit has no trajectory data. Please call propagate() before plotting."
                logger.error(err)
                raise ValueError(err)

            states_list.append(orb.trajectory)
            times_list.append(orb.times)

        first_orbit = self.orbits[0]
        bodies = [first_orbit.system.primary, first_orbit.system.secondary]
        system_distance = first_orbit.system.distance

        return plot_orbit_family(
            states_list,
            times_list,
            np.asarray(self.parameter_values),
            bodies,
            system_distance,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )
