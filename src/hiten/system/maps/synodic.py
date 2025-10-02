from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.core.types import _Section
from hiten.algorithms.poincare.synodic.types import SynodicMapResults
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.maps import (_MapPersistenceService,
                                                  _MapServices)
from hiten.algorithms.types.states import Trajectory
from hiten.system.manifold import Manifold
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.plots import plot_poincare_map

if TYPE_CHECKING:
    from hiten.algorithms.poincare.synodic.options import SynodicMapOptions


class SynodicMap(_HitenBase):
    """Poincare map for a synodic section.
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.orbits.base.PeriodicOrbit` or :class:`~hiten.system.manifold.Manifold`
        The domain object.

    Attributes
    ----------
    trajectories : List[:class:`~hiten.algorithms.types.states.Trajectory`]
        The trajectories.
    source : :class:`~hiten.system.orbits.base.PeriodicOrbit` or :class:`~hiten.system.manifold.Manifold`
        The source.
    """

    def __init__(self, domain_obj: Literal[PeriodicOrbit, Manifold]):
        self._trajectories = domain_obj.dynamics.trajectories
        self._source = domain_obj
        self._last_results: SynodicMapResults | None = None
        services = _MapServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"SynodicMap(source={self._source.__class__.__name__})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def trajectories(self) -> list[Trajectory]:
        """The trajectories."""
        return self.dynamics.trajectories

    @property
    def source(self) -> Literal[PeriodicOrbit, Manifold]:
        """The source."""
        return self.dynamics.source

    @property
    def sections(self) -> list[str]:
        """The sections."""
        return self.dynamics.list_sections()

    @property
    def config(self):
        """The map configuration."""
        return self.dynamics.map_config

    @config.setter
    def config(self, value):
        """Set the map configuration."""
        self.dynamics.map_config = value

    @property
    def options(self) -> "SynodicMapOptions":
        """Get the map runtime options.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            The map options with reasonable defaults.
        """
        return self.dynamics.map_options

    @options.setter
    def options(self, value: "SynodicMapOptions"):
        """Set the map runtime options.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`
            New map options.
        """
        self.dynamics.map_options = value

    def get_section(self, section_coord: str) -> _Section:
        """Get the section.
        
        Parameters
        ----------
        section_coord : str
            The section coordinate.
        """
        return self.dynamics.get_section(section_coord)
    
    def has_section(self, section_coord: str) -> bool:
        """Check if the section exists.
        
        Parameters
        ----------
        section_coord : str
            The section coordinate.
        """
        return self.dynamics.has_section(section_coord)
    
    def clear_sections(self) -> None:
        """Clear the sections."""
        return self.dynamics.clear()

    def compute(self, *, section_axis: str, section_offset: float, plane_coords: tuple[str, str], direction: Optional[Literal[1, -1, None]] = None, options: Optional["SynodicMapOptions"] = None) -> SynodicMapResults:
        """Compute the Poincare map.
        
        Parameters
        ----------
        section_axis : str
            The section axis.
        section_offset : float
            The section offset.
        plane_coords : tuple[str, str]
            The plane coordinates.
        direction : Literal[1, -1, None], optional
            The direction.
        options : :class:`~hiten.algorithms.poincare.synodic.options.SynodicMapOptions`, optional
            Runtime options for the map computation. If None, uses defaults.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.synodic.types.SynodicMapResults`
            The results of the Poincare map.
        """
        return self.dynamics.compute(section_axis=section_axis, section_offset=section_offset, plane_coords=plane_coords, direction=direction, options=options)

    def get_points(self, axes: Optional[tuple[str, str]] = None) -> np.ndarray:
        """Get points from the Poincare map.

        Parameters
        ----------
        axes : tuple[str, str], optional
            Axes to project onto. If None, uses the section plane coordinates.

        Returns
        -------
        ndarray, shape (n, 2)
            Array of 2D points in the section plane.
        """
        return self.dynamics.get_points(axes=axes)

    def plot(
        self,
        *,
        axes: Optional[Sequence[str]] = None,
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
        # Get points using dynamics service
        if axes is None:
            pts = self.dynamics.get_points()
            section = self.dynamics.get_section()
            lbls = section.labels
        else:
            pts = self.dynamics.get_points(axes=tuple(axes))
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