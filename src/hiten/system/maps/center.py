from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence

import numpy as np

from hiten.algorithms.poincare.centermanifold.types import \
    CenterManifoldMapResults
from hiten.algorithms.poincare.core.types import _Section
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.maps import (_MapPersistenceService,
                                                  _MapServices)
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive

if TYPE_CHECKING:
    from hiten.algorithms.poincare.centermanifold.options import \
        CenterManifoldMapOptions
    from hiten.system.center import CenterManifold

class CenterManifoldMap(_HitenBase):
    """Poincare map for a center manifold.
    
    Parameters
    ----------
    center_manifold : :class:`~hiten.system.center.CenterManifold`
        The center manifold.
    energy : float
        The energy of the center manifold.
    
    Attributes
    ----------
    center_manifold : :class:`~hiten.system.center.CenterManifold`
        The center manifold.
    energy : float
        The energy of the center manifold.
    """

    def __init__(self, center_manifold: "CenterManifold", energy: float):
        self._center_manifold = center_manifold
        self._energy = energy
        self._last_map: CenterManifoldMapResults | None = None
        services = _MapServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"CenterManifoldMap(center_manifold={self._center_manifold}, energy={self._energy})"
    
    def __repr__(self) -> str:
        return self.__str__()

    @property
    def center_manifold(self) -> "CenterManifold":
        """The center manifold."""
        return self.dynamics.center_manifold
    
    @property
    def energy(self) -> float:
        """The energy of the center manifold."""
        return self.dynamics.energy

    @property
    def config(self):
        """Get the map configuration."""
        return self.dynamics.map_config

    @config.setter
    def config(self, value):
        self.dynamics.map_config = value

    @property
    def options(self) -> "CenterManifoldMapOptions":
        """Get the map runtime options.
        
        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            The map options with reasonable defaults.
        """
        return self.dynamics.map_options

    @options.setter
    def options(self, value: "CenterManifoldMapOptions"):
        """Set the map runtime options.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`
            New map options.
        """
        self.dynamics.map_options = value

    @property
    def sections(self) -> list[str]:
        """The sections of the center manifold."""
        return self.dynamics.list_sections()

    def get_section(self, section_coord: str) -> _Section:
        """Get the section of the center manifold."""
        return self.dynamics.get_section(section_coord)
    
    def has_section(self, section_coord: str) -> bool:
        """Check if the center manifold has a section."""
        return self.dynamics.has_section(section_coord)
    
    def clear_sections(self) -> None:
        """Clear the sections of the center manifold."""
        return self.dynamics.clear()

    def compute(self, section_coord: Optional[str] = "q3", options: Optional["CenterManifoldMapOptions"] = None) -> CenterManifoldMapResults:
        """Compute the Poincare map.
        
        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        options : :class:`~hiten.algorithms.poincare.centermanifold.options.CenterManifoldMapOptions`, optional
            Runtime options for the map computation. If None, uses defaults.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.types.CenterManifoldMapResults`
            The results of the Poincare map.
        """
        return self.dynamics.compute(section_coord=section_coord, options=options)

    def get_states(self, section_coord: Optional[str] = "q3", axes: Optional[tuple[str, str]] = None) -> np.ndarray:
        """Get the states of the Poincare map.
        
        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        axes : tuple[str, str], optional
            Axes to project onto. If None, uses the section plane coordinates.

        Returns
        -------
        np.ndarray, shape (n, 4)
            Array of 4D states in the section plane.
        """
        return self.dynamics.get_points_with_4d_states(section_coord=section_coord, axes=axes)

    def get_points(self, section_coord: Optional[str] = None, axes: Optional[tuple[str, str]] = None) -> np.ndarray:
        """Get points from the Poincare map.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        axes : tuple[str, str], optional
            Axes to project onto. If None, uses the section plane coordinates.

        Returns
        -------
        ndarray, shape (n, 2)
            Array of 2D points in the section plane.
        """
        return self.dynamics.get_points(section_coord=section_coord, axes=axes)

    def to_synodic(self, pt: np.ndarray, *, section_coord: Optional[str] = None, tol: float = 1e-12) -> np.ndarray:
        """Convert a plane point to initial conditions for integration.

        Parameters
        ----------
        pt : ndarray, shape (2,)
            Point on the Poincare section plane.
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default
            section coordinate from configuration.
        tol : float, optional
            Tolerance for root finding. Default is 1e-12.

        Returns
        -------
        ndarray, shape (6,)
            Initial conditions [q1, q2, q3, p1, p2, p3] for integration.
        """
        return self.dynamics.to_synodic(pt, section_coord=section_coord, tol=tol)

    def plot(
        self,
        section_coord: Optional[str] = None,
        *,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        axes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """Plot the Poincare map.

        Parameters
        ----------
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.
        dark_mode : bool, default=True
            If True, use dark mode styling.
        save : bool, default=False
            If True, save the plot to file.
        filepath : str, default='poincare_map.svg'
            File path for saving the plot.
        axes : Sequence[str], optional
            Axes to plot. If None, uses the section plane coordinates.
        **kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot figure.
        """
        # Get points using dynamics service
        if axes is None:
            pts = self.dynamics.get_points(section_coord=section_coord)
            section = self.dynamics.get_section(section_coord)
            lbls = section.labels
        else:
            pts = self.dynamics.get_points_with_4d_states(section_coord=section_coord, axes=tuple(axes))
            lbls = tuple(axes)

        return plot_poincare_map(
            points=pts,
            labels=lbls,
            dark_mode=dark_mode,
            save=save,
            filepath=filepath,
            **kwargs,
        )

    def plot_interactive(
        self,
        *,
        steps=1000,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order=8,
        frame="rotating",
        dark_mode: bool = True,
        axes: Optional[Sequence[str]] = None,
        section_coord: Optional[str] = None,
    ) -> tuple[Any, dict]:
        """Create an interactive plot of the Poincare map.

        Parameters
        ----------
        steps : int, default=1000
            Number of integration steps for trajectory propagation.
        method : {'fixed', 'symplectic', 'adaptive'}, default='adaptive'
            Integration method for trajectory propagation.
        order : int, default=6
            Integration order for Runge-Kutta methods.
        frame : str, default='rotating'
            Reference frame for trajectory visualization.
        dark_mode : bool, default=True
            If True, use dark mode styling.
        axes : Sequence[str], optional
            Axes to plot. If None, uses the section plane coordinates.
        section_coord : str, optional
            Section coordinate identifier. If None, uses the default section.

        Returns
        -------
        tuple[matplotlib.figure.Figure, dict]
            A tuple containing:
            - The interactive plot figure
            - A dictionary with key 'orbit' that will contain the latest 
              computed orbit after clicking on a point (initially None)

        Notes
        -----
        Clicking on points in the plot will propagate trajectories from
        those points and display the resulting orbits. The latest orbit
        can be accessed via the returned dictionary.
        
        Examples
        --------
        >>> fig, orbit_container = pm.plot_interactive()
        >>> # After clicking on a point in the plot
        >>> latest_orbit = orbit_container['orbit']
        """
        # Container to store the latest orbit
        orbit_container = {'orbit': None}
        
        def _on_select(pt_np: np.ndarray):
            if axes is None:
                section_pt = pt_np
            else:
                proj_pts = self.dynamics.get_points_with_4d_states(section_coord=section_coord, axes=tuple(axes))
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                section = self.dynamics.get_section(section_coord)
                section_pt = section.points[np.argmin(distances)]

            orbit = self.dynamics._propagate_from_point(
                section_pt,
                self.energy,
                steps=steps,
                method=method,
                order=order,
            )

            orbit.plot(frame=frame, dark_mode=dark_mode, block=False, close_after=False)
            
            # Store the latest orbit in the container
            orbit_container['orbit'] = orbit

            return orbit

        # Get points using dynamics service
        if axes is None:
            pts = self.dynamics.get_points(section_coord=section_coord)
            section = self.dynamics.get_section(section_coord)
            lbls = section.labels
        else:
            pts = self.dynamics.get_points_with_4d_states(section_coord=section_coord, axes=tuple(axes))
            lbls = tuple(axes)

        fig = plot_poincare_map_interactive(
            points=pts,
            labels=lbls,
            on_select=_on_select,
            dark_mode=dark_mode,
        )
        
        return fig, orbit_container

    def __setstate__(self, state):
        """Restore the CenterManifoldMap instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of center_manifold and energy.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the CenterManifoldMap.
        """
        super().__setstate__(state)
        self._setup_services(_MapServices.default(self))

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load orbit data from a file in place."""
        self.persistence.load_inplace(self, filepath)
        self.dynamics.reset()
        return self

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "CenterManifoldMap":
        """Load a CenterManifoldMap from a file (new instance)."""
        return cls._load_with_services(
            filepath, 
            _MapPersistenceService(), 
            _MapServices.default, 
            **kwargs
        )
