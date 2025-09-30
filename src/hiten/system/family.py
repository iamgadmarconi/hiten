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
from hiten.algorithms.types.services.family import (
    _OrbitFamilyPersistenceService, _OrbitFamilyServices)
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.io.common import _ensure_dir
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_orbit_family


class OrbitFamily(_HitenBase):
    """Container for an ordered family of periodic orbits.
    
    Parameters
    ----------
    orbits : List[:class:`~hiten.system.orbits.base.PeriodicOrbit`]
        The orbits in the family.
    parameter_name : str
        The name of the parameter that is varied.
    parameter_values : np.ndarray
        The values of the parameter that is varied.
    """

    def __init__(
        self,
        orbits: List[PeriodicOrbit] | None = None,
        parameter_name: str = "param",
        parameter_values: np.ndarray | None = None,
) -> None:

        services = _OrbitFamilyServices.default(self)
        super().__init__(services)
        self.orbits: List[PeriodicOrbit] = list(orbits) if orbits is not None else []
        self.parameter_name = parameter_name

        if parameter_values is None:
            self.parameter_values = np.full(len(self.orbits), np.nan, dtype=float)
        else:
            arr = np.asarray(parameter_values, dtype=float)
            if arr.shape[0] != len(self.orbits):
                raise ValueError("Length of parameter_values must match number of orbits")
            self.parameter_values = arr

    def __repr__(self) -> str:
        return f"OrbitFamily(n_orbits={len(self)}, parameter='{self.parameter_name}')"

    def __str__(self) -> str:
        return f"OrbitFamily(n_orbits={len(self)}, parameter='{self.parameter_name}')"

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
    def jacobis(self) -> np.ndarray:
        """Array of Jacobi constants for all orbits.
        
        Returns
        -------
        numpy.ndarray
            Array of Jacobi constants (dimensionless).
        """
        return np.array([o.jacobi for o in self.orbits])
    
    def propagate(self, **kwargs) -> None:
        """Propagate all orbits in the family.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to each orbit's propagate method.
        """
        for orb in self.orbits:
            orb.propagate(**kwargs)

    def to_csv(self, filepath: str, **kwargs) -> None:
        """
        Export the contents of the orbit family to a CSV file.

        Parameters
        ----------
        filepath : str or Path
            Destination CSV file path.
        **kwargs
            Extra keyword arguments passed to :meth:`~hiten.system.orbits.base.PeriodicOrbit.propagate`.

        Raises
        ------
        ValueError
            If no trajectory data is available to export.
        """
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        df = self.to_df(**kwargs)
        df.to_csv(filepath, index=False)
        logger.info(f"Orbit family trajectories successfully exported to {filepath}")

    def to_df(self, **kwargs) -> pd.DataFrame:
        """Return a DataFrame summarising the family.
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to :meth:`~hiten.system.orbits.base.PeriodicOrbit.propagate`.
        """
        data = []
        for idx, orbit in enumerate(self.orbits):
            # Check if trajectory is computed
            try:
                trajectory = orbit.trajectory
            except ValueError:
                # Trajectory not computed, propagate it
                orbit.propagate(**kwargs)
                trajectory = orbit.trajectory
            
            for t, state in zip(trajectory.times, trajectory.states):
                data.append([idx, self.parameter_values[idx], t, *state])

        if not data:
            raise ValueError("No trajectory data available to export.")

        columns = [
            "orbit_id", self.parameter_name, "time",
            "x", "y", "z", "vx", "vy", "vz",
        ]
        df = pd.DataFrame(data, columns=columns)
        return df

    def __getstate__(self):
        state = super().__getstate__()
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._setup_services(_OrbitFamilyServices.default(self))

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

    def save(self, filepath: str | Path, *, compression: str = "gzip", level: int = 4) -> None:
        """Save the family to an pickle file.
        
        Parameters
        ----------
        filepath : str or Path
            The path to the file to save the family to.
        compression : str, default "gzip"
            The compression algorithm to use.
        level : int, default 4
            The compression level to use.
        """
        self.persistence.save(self, filepath, compression=compression, level=level)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "OrbitFamily":
        """Load a OrbitFamily from a file (new instance).
        
        Parameters
        ----------
        filepath : str or Path
            The path to the file to load the family from.
        **kwargs
            Additional keyword arguments for the load operation.
        """
        return cls._load_with_services(
            filepath, 
            _OrbitFamilyPersistenceService(), 
            _OrbitFamilyServices.default, 
            **kwargs
        )


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
            try:
                trajectory = orb.trajectory
            except ValueError:
                raise ValueError("Orbit has no trajectory data. Please call propagate() before plotting.")

            states_list.append(trajectory.states)
            times_list.append(trajectory.times)

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
