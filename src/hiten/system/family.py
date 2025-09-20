"""Light-weight container that groups a family of periodic orbits obtained via a
continuation engine.

It offers convenience helpers for iteration, random access, conversion to a
pandas.DataFrame, and basic serialisation to an HDF5 file leveraging the
existing utilities in :mod:`~hiten.utils.io`.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List

import h5py
import numpy as np
import pandas as pd

from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io.common import _ensure_dir
from hiten.utils.io.orbits import _read_orbit_group, _write_orbit_group
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_orbit_family


@dataclass
class OrbitFamily:
    """Container for an ordered family of periodic orbits.
    
    Parameters
    ----------
    orbits : list of :class:`~hiten.system.orbits.base.PeriodicOrbit`
        List of periodic orbits in ascending continuation order.
    parameter_name : str, default "param"
        Name of the continuation parameter for labelling.
    parameter_values : numpy.ndarray or None, optional
        Array of parameter values corresponding to each orbit.
        If None, will be initialized with NaN values.
    
    Attributes
    ----------
    orbits : list of :class:`~hiten.system.orbits.base.PeriodicOrbit`
        List of periodic orbits in the family.
    parameter_name : str
        Name of the continuation parameter.
    parameter_values : numpy.ndarray
        Array of parameter values for each orbit.
    """

    orbits: List[PeriodicOrbit] = field(default_factory=list)
    parameter_name: str = "param"
    parameter_values: np.ndarray | None = None  # one value per orbit

    def __post_init__(self) -> None:
        """Initialize parameter values after dataclass creation.
        
        This method ensures parameter_values is properly initialized and
        validates that the length matches the number of orbits.
        
        Raises
        ------
        ValueError
            If the length of parameter_values does not match the number of orbits.
        """
        if self.parameter_values is None:
            self.parameter_values = np.full(len(self.orbits), np.nan, dtype=float)
        else:
            self.parameter_values = np.asarray(self.parameter_values, dtype=float)
            if self.parameter_values.shape[0] != len(self.orbits):
                raise ValueError("Length of parameter_values must match number of orbits")

    @classmethod
    def from_result(cls, result, parameter_name: str | None = None):
        """Build an OrbitFamily from a ContinuationResult.

        Parameters
        ----------
        result : ContinuationResult
            Result object returned by the new continuation engine/facade.
        parameter_name : str or None, optional
            Name for the continuation parameter. If None, defaults to "param".

        Returns
        -------
        :class:`~hiten.system.family.OrbitFamily`
            A new OrbitFamily instance containing the orbits from the result.
        """
        if parameter_name is None:
            parameter_name = "param"

        orbits = list(result.family)

        # Coerce tuple of parameter vectors to 1D array (one value per orbit)
        param_vals_list: list[float] = []
        for vec in result.parameter_values:
            arr = np.asarray(vec, dtype=float)
            if arr.ndim == 0 or arr.size == 1:
                param_vals_list.append(float(arr.reshape(-1)[0]))
            else:
                # Use Euclidean norm for multi-parameter continuation by default
                param_vals_list.append(float(np.linalg.norm(arr)))
        param_vals = np.asarray(param_vals_list, dtype=float)

        return cls(orbits, parameter_name, param_vals)

    def __len__(self) -> int:
        return len(self.orbits)

    def __iter__(self) -> Iterator[PeriodicOrbit]:
        return iter(self.orbits)

    def __getitem__(self, idx):
        return self.orbits[idx]

    @property
    def periods(self) -> np.ndarray:
        """Array of orbit periods.
        
        Returns
        -------
        numpy.ndarray
            Array of orbit periods in nondimensional units (NaN if not available).
        """
        return np.array([o.period if o.period is not None else np.nan for o in self.orbits])

    @property
    def jacobi_constants(self) -> np.ndarray:
        """Array of Jacobi constants for all orbits.
        
        Returns
        -------
        numpy.ndarray
            Array of Jacobi constants (dimensionless).
        """
        return np.array([o.jacobi_constant for o in self.orbits])
    
    def propagate(self, **kwargs) -> None:
        """Propagate all orbits in the family.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to each orbit's propagate method.
        """
        for orb in self.orbits:
            orb.propagate(**kwargs)

    def save(self, filepath: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
        """Save the family to an HDF5 file.
        
        Each orbit is saved as a subgroup within the HDF5 file.
        
        Parameters
        ----------
        filepath : str or pathlib.Path
            Path where to save the HDF5 file.
        compression : str, default "gzip"
            Compression algorithm to use for the HDF5 file.
        level : int, default 4
            Compression level (0-9, higher means better compression).
        """
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
                _write_orbit_group(sub, orbit, compression=compression, level=level)

        logger.info(f"Family saved to {filepath}")

    def to_csv(self, filepath: str, **kwargs) -> None:
        """
        Export the contents of the orbit family to a CSV file.

        Parameters
        ----------
        filepath : str
            Destination CSV file path.
        **kwargs
            Extra keyword arguments passed to :meth:`~hiten.system.orbits.base.PeriodicOrbit.propagate`.

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
        """Load a family previously saved with save method.
        
        Parameters
        ----------
        filepath : str or pathlib.Path
            Path to the HDF5 file containing the saved family.
            
        Returns
        -------
        :class:`~hiten.system.family.OrbitFamily`
            The loaded OrbitFamily instance.
            
        Raises
        ------
        ValueError
            If the file does not contain a valid :class:`~hiten.system.family.OrbitFamily` object.
        """
        with h5py.File(filepath, "r") as f:
            if str(f.attrs.get("class", "")) != "OrbitFamily":
                raise ValueError("File does not contain an OrbitFamily object")
            param_name = str(f.attrs["parameter_name"])
            param_vals = f["parameter_values"][()]
            orbits: List[PeriodicOrbit] = []
            for key in sorted(f["orbits"], key=lambda s: int(s)):
                grp = f["orbits"][key]
                orbits.append(_read_orbit_group(grp))
        return cls(orbits, param_name, param_vals)

    def __repr__(self) -> str:
        return f"OrbitFamily(n_orbits={len(self)}, parameter='{self.parameter_name}')"

    def plot(self, *, dark_mode: bool = True, save: bool = False, filepath: str = "orbit_family.svg", **kwargs):
        """Visualise the family trajectories in rotating frame.
        
        Parameters
        ----------
        dark_mode : bool, default True
            Whether to use dark mode for the plot.
        save : bool, default False
            Whether to save the plot to a file.
        filepath : str, default "orbit_family.svg"
            Path where to save the plot if save=True.
        **kwargs
            Additional keyword arguments passed to the plotting function.
            
        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
            
        Raises
        ------
        ValueError
            If orbits have no trajectory data available.
        """

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
