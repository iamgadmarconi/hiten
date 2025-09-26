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
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.family import _OrbitFamilyServices
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io.common import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_orbit_family


class OrbitFamily(_HitenBase):
    """Container for an ordered family of periodic orbits."""

    def __init__(
        self,
        orbits: List[PeriodicOrbit] | None = None,
        parameter_name: str = "param",
        parameter_values: np.ndarray | None = None,
        *,
        services: _OrbitFamilyServices | None = None,
    ) -> None:
        super().__init__()
        self.orbits: List[PeriodicOrbit] = list(orbits) if orbits is not None else []
        self.parameter_name = parameter_name

        if parameter_values is None:
            self.parameter_values = np.full(len(self.orbits), np.nan, dtype=float)
        else:
            arr = np.asarray(parameter_values, dtype=float)
            if arr.shape[0] != len(self.orbits):
                raise ValueError("Length of parameter_values must match number of orbits")
            self.parameter_values = arr

        self._services: _OrbitFamilyServices = services or _OrbitFamilyServices.default()

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
        """Save the family to an HDF5 file."""
        self._services.persistence.save(self, filepath, compression=compression, level=level)

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

    def to_df(self, **kwargs) -> pd.DataFrame:
        """Return a DataFrame summarising the family."""
        return pd.DataFrame(
            {
                "orbit": self.orbits,
                self.parameter_name: self.parameter_values,
                "period": self.periods,
                "jacobi_constant": self.jacobi_constants,
            }
        )

    @classmethod
    def load(cls, filepath: str | Path):
        """Load a family previously saved with save method."""
        services = _OrbitFamilyServices.default()
        orbits, parameter_name, parameter_values = services.persistence.load(filepath)
        family = cls(orbits, parameter_name, parameter_values)
        family._services = services
        return family

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
