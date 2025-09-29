from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, List

import numpy as np

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.maps import (_MapPersistenceService,
                                                  _MapServices)
from hiten.system.manifold import Manifold
from hiten.system.orbits.base import PeriodicOrbit
from hiten.algorithms.types.states import Trajectory
from hiten.utils.plots import plot_poincare_map


class SynodicMap(_HitenBase):

    def __init__(self, domain_obj: Literal[PeriodicOrbit, Manifold, List[Trajectory]]):
        self._domain_obj = domain_obj
        services = _MapServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"SynodicMap(domain_obj={self._domain_obj})"
    
    def __repr__(self) -> str:
        return self.__str__()

    def plot(
        self,
        *,
        axes: Sequence[str] | None = None,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        **kwargs,
    ):
        """Render a 2D Poincare map for the last computed synodic section.

        Parameters
        ----------
        axes : sequence of str, optional
            Coordinate axes to plot. If None, uses the default
            plane coordinates from the section configuration.
        dark_mode : bool, default True
            Whether to use dark mode for the plot.
        save : bool, default False
            Whether to save the plot to a file.
        filepath : str, default "poincare_map.svg"
            File path for saving the plot (only used if save=True).
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.

        Raises
        ------
        ValueError
            If no synodic section has been computed yet.

        Notes
        -----
        This method renders a 2D Poincare map for the most recently
        computed synodic section. It requires that `from_orbit`,
        `from_manifold`, or `from_trajectories` has been called to
        populate the cached section.

        The method supports custom axis selection and automatically
        handles the projection of the section data to 2D coordinates
        for visualization.

        The plot shows the Poincare section points in the specified
        coordinate system, providing a visual representation of the
        section's structure.
        """
        if self._section is None:
            raise ValueError("No synodic section cached. Compute from orbit or manifold first.")

        section = self._section

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            cols = []
            for ax in axes:
                if ax in section.labels:
                    idx = section.labels.index(ax)
                    cols.append(section.points[:, idx])
                else:
                    idx = self._section_iface.coordinate_index(ax)
                    cols.append(section.states[:, idx])
            pts = np.column_stack(cols)
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )

    def __setstate__(self, state):
        """Restore the SynodicMap instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of domain_obj.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the SynodicMap.
        """
        super().__setstate__(state)
        self._setup_services(_MapServices.default(self))

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load orbit data from a file in place."""
        self.persistence.load_inplace(self, filepath)
        self.dynamics.reset()
        return self

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "SynodicMap":
        """Load a CenterManifoldMap from a file (new instance)."""
        return cls._load_with_services(
            filepath, 
            _MapPersistenceService(), 
            _MapServices.default, 
            **kwargs
        )