"""Stable/unstable invariant manifolds of periodic orbits in the spatial circular
restricted three-body problem.

The module offers a high-level interface (:class:`~hiten.system.manifold.Manifold`) that, given a
generating :class:`~hiten.system.orbits.base.PeriodicOrbit`, launches trajectory
integrations along the selected eigen-directions, records their intersections
with the canonical Poincare section, provides quick 3-D visualisation, and
handles (de)serialisation through :meth:`~hiten.system.manifold.Manifold.save` and
:meth:`~hiten.system.manifold.Manifold.load`.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Koon, W. S., Lo, M. W., Marsden, J. E., & Ross, S. D. (2016). "Dynamical Systems, the Three-Body Problem
and Space Mission Design".
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Literal

import pandas as pd

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.manifold import (
    _ManifoldPersistenceService, _ManifoldServices)
from hiten.algorithms.types.states import Trajectory
from hiten.utils.io.common import _ensure_dir
from hiten.utils.plots import plot_manifold

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.orbits.base import PeriodicOrbit


class Manifold(_HitenBase):
    """
    Compute and cache the invariant manifold of a periodic orbit.

    Parameters
    ----------
    generating_orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        Orbit that seeds the manifold.
    stable : bool, default True
        True selects the stable manifold, False the unstable one.
    direction : {'positive', 'negative'}, default 'positive'
        Sign of the eigenvector used to initialise the manifold branch.

    Attributes
    ----------
    generating_orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        Orbit that seeds the manifold.
    libration_point : :class:`~hiten.system.libration.base.LibrationPoint`
        Libration point associated with generating_orbit.
    stable : int
        Encoded stability: 1 for stable, -1 for unstable.
    direction : int
        Encoded direction: 1 for 'positive', -1 for 'negative'.
    mu : float
        Mass ratio of the underlying CRTBP system (dimensionless).
    manifold_result : :class:`~hiten.system.manifold.ManifoldResult` or None
        Cached result returned by the last successful compute call.

    Notes
    -----
    Re-invoking compute after a successful run returns the cached
    :class:`~hiten.system.manifold.ManifoldResult` without recomputation.
    """

    def __init__(
            self, 
            generating_orbit: "PeriodicOrbit", 
            stable: bool = True, 
            direction: Literal["positive", "negative"] = "positive", 
        ):
        self._generating_orbit = generating_orbit
        self._stable = stable
        self._direction = direction

        services = _ManifoldServices.default(self)
        super().__init__(services)

    def __str__(self):
        return f"Manifold(stable={self.stable}, direction={self.direction}) of {self.generating_orbit}"
    
    def __repr__(self):
        return self.__str__()

    @property
    def generating_orbit(self) -> "PeriodicOrbit":
        """Orbit that seeds the manifold.
        
        Returns
        -------
        :class:`~hiten.system.orbits.base.PeriodicOrbit`
            The generating periodic orbit.
        """
        return self.dynamics.orbit

    @property
    def libration_point(self):
        """Libration point associated with the generating orbit.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point associated with the generating orbit.
        """
        return self.dynamics.libration_point

    @property
    def system(self) -> "System":
        """The system this manifold belongs to.
        
        Returns
        -------
        :class:`~hiten.system.base.System`
            The system this manifold belongs to.
        """
        return self.dynamics.system

    @property
    def mu(self) -> float:
        """Mass ratio of the system.
        
        Returns
        -------
        float
            The mass ratio (dimensionless).
        """
        return self.dynamics.mu

    @property
    def stable(self) -> int:
        """Encoded stability: 1 for stable, -1 for unstable.
        
        Returns
        -------
        int
            Encoded stability: 1 for stable, -1 for unstable.
        """
        return self.dynamics.stable

    @property
    def direction(self) -> int:
        """Encoded direction: 1 for 'positive', -1 for 'negative'.
        
        Returns
        -------
        int
            Encoded direction: 1 for 'positive', -1 for 'negative'.
        """
        return self.dynamics.direction

    @property
    def result(self):
        """Cached result from the last successful compute call.
        
        Returns
        -------
        :class:`~hiten.system.manifold.ManifoldResult` or None
            The cached manifold result, or None if not computed.
        """
        return self.dynamics.manifold_result

    @property
    def eigendecomposition_config(self):
        """Get the eigenvalue decomposition configuration.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            The eigendecomposition configuration.
        """
        return self.dynamics.eigendecomposition_config

    @eigendecomposition_config.setter
    def eigendecomposition_config(self, value):
        """Set the eigenvalue decomposition configuration.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            New eigendecomposition configuration.
        """
        self.dynamics.eigendecomposition_config = value

    @property
    def eigendecomposition_options(self):
        """Get the eigenvalue decomposition runtime options.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            The eigendecomposition options with reasonable defaults.
        """
        return self.dynamics.eigendecomposition_options

    @eigendecomposition_options.setter
    def eigendecomposition_options(self, value):
        """Set the eigenvalue decomposition runtime options.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            New eigendecomposition options.
        """
        self.dynamics.eigendecomposition_options = value

    @property
    def trajectories(self) -> List[Trajectory]:
        """The trajectories of the manifold.
        
        Returns
        -------
        List[:class:`~hiten.algorithms.types.states.Trajectory`]
            The trajectories of the manifold.
        """
        return self.dynamics.trajectories

    def compute(self, step: float = 0.02, integration_fraction: float = 0.75, NN: int = 1, displacement: float = 1e-6, dt: float = 1e-3, method: Literal["fixed", "adaptive", "symplectic"] = "adaptive", order: int = 8, **kwargs):
        """
        Generate manifold trajectories and build a Poincare map.

        The routine samples the generating orbit at equally spaced fractions
        of its period, displaces each point by displacement along the
        selected eigenvector and integrates the resulting initial condition
        for integration_fraction of one synodic period.

        Parameters
        ----------
        step : float, default 0.02
            Increment of the dimensionless fraction along the orbit (i.e. 50 samples per orbit).
        integration_fraction : float, default 0.75
            Portion of 2*pi nondimensional time units to integrate
            each trajectory.
        NN : int, default 1
            Index of the real eigenvector to follow (1-based).
        displacement : float, default 1e-6
            Dimensionless displacement applied along the eigenvector.
        method : {'fixed', 'adaptive', 'symplectic'}, default 'adaptive'
            Integration method to use.
        order : int, default 8
            Integration order for fixed-step methods.
        **kwargs
            Additional options:

            show_progress : bool, default True
                Display a tqdm progress bar.
            energy_tol : float, default 1e-6
                Maximum relative variation of the Jacobi constant allowed along a trajectory.
                Larger deviations indicate numerical error (often triggered by near-singular
                passages) and cause the trajectory to be discarded.
            safe_distance : float, default 2.0
                Safety multiplier applied to the physical radii of both primaries. A trajectory
                is rejected if it ever comes within safe_distance x radius of either body.

        Returns
        -------
        :class:`~hiten.system.manifold.ManifoldResult`
            The computed manifold result containing trajectories and Poincare section data.

        Raises
        ------
        ValueError
            If called after a previous run with incompatible settings or if requested
            eigenvector is not available.

        Examples
        --------
        >>> from hiten.system import System, Manifold
        >>> system = System.from_bodies("earth", "moon")
        >>> L2 = system.get_libration_point(2)
        >>> halo_L2 = L2.create_orbit('halo', amplitude_z=0.3, zenith='northern')
        >>> halo_L2.correct()
        >>> halo_L2.propagate()
        >>> manifold = halo_L2.manifold(stable=True, direction='positive')
        >>> result = manifold.compute(step=0.05)
        >>> print(f"Success rate: {result.success_rate:.0%}")
        """
        kwargs.setdefault("show_progress", True)
        kwargs.setdefault("energy_tol", 1e-6)
        kwargs.setdefault("safe_distance", 2.0)

        result = self.dynamics.compute_manifold(
            step=step,
            integration_fraction=integration_fraction,
            NN=NN,
            displacement=displacement,
            method=method,
            order=order,
            dt=dt,
            energy_tol=kwargs["energy_tol"],
            safe_distance=kwargs["safe_distance"],
            show_progress=kwargs["show_progress"],
        )

        return result

    def plot(self, dark_mode: bool = True, save: bool = False, filepath: str = 'manifold.svg', **kwargs):
        """
        Render a 3-D plot of the computed manifold.

        Parameters
        ----------
        dark_mode : bool, default True
            Apply a dark colour scheme.
        save : bool, default False
            Whether to save the plot to a file.
        filepath : str, default 'manifold.svg'
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
            If manifold_result is None.
        """
        if self.trajectories is None:
            raise ValueError("Manifold result not computed. Please compute the manifold first.")

        # Extract states and times from the list of trajectories
        states_list = [traj.states for traj in self.trajectories]
        times_list = [traj.times for traj in self.trajectories]

        return plot_manifold(
            states_list=states_list,
            times_list=times_list,
            bodies=[self.system.primary, self.system.secondary],
            system_distance=self.system.distance,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs
        )

    def to_csv(self, filepath: str, **kwargs):
        """
        Export manifold trajectory data to a CSV file.

        Each row in the CSV file represents a point in a trajectory,
        and includes a trajectory ID, timestamp, and the 6D state vector
        (x, y, z, vx, vy, vz).

        Parameters
        ----------
        filepath : str
            Path to the output CSV file. Parent directories are created if
            they do not exist.
        **kwargs
            Reserved for future use.

        Raises
        ------
        ValueError
            If manifold_result is None.
        """
        df = self.to_df(**kwargs)
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        df.to_csv(filepath, index=False)

    def to_df(self, **kwargs):
        """
        Export manifold trajectory data to a pandas DataFrame.
        """
        if self.trajectories is None:
            raise ValueError("Manifold result not computed. Please compute the manifold first.")

        data = []
        for i, traj in enumerate(self.trajectories):
            for j in range(traj.states.shape[0]):
                data.append(
                    [i, traj.times[j], traj.states[j, 0], traj.states[j, 1], traj.states[j, 2], traj.states[j, 3], traj.states[j, 4], traj.states[j, 5]]
                )
        
        return pd.DataFrame(data, columns=['trajectory_id', 'time', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    def __setstate__(self, state):
        """Restore the Manifold instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of stable and direction.
        secondary bodies.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the Manifold.
        """
        super().__setstate__(state)
        self._setup_services(_ManifoldServices.default(self))

    @classmethod
    def load(cls, filepath: str, **kwargs) -> "Manifold":
        """Load a manifold from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the file containing the saved manifold.
            
        Returns
        -------
        :class:`~hiten.system.manifold.Manifold`
            The loaded Manifold instance.
            
        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        return cls._load_with_services(
            filepath, 
            _ManifoldPersistenceService(), 
            _ManifoldServices.default, 
            **kwargs
        )