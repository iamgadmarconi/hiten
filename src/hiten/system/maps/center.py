from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.maps import (_MapPersistenceService,
                                                  _MapServices)
from hiten.system.center import CenterManifold
from hiten.utils.plots import plot_poincare_map, plot_poincare_map_interactive


class CenterManifoldMap(_HitenBase):
    """Poincare map for a center manifold."""

    def __init__(self, center_manifold: CenterManifold, energy: float):
        self._center_manifold = center_manifold
        self._energy = energy
        services = _MapServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"CenterManifoldMap(center_manifold={self._center_manifold}, energy={self._energy})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def center_manifold(self) -> CenterManifold:
        return self.dynamics.center_manifold
    
    @property
    def energy(self) -> float:
        return self.dynamics.energy

    def plot(
        self,
        section_coord: str | None = None,
        *,
        dark_mode: bool = True,
        save: bool = False,
        filepath: str = "poincare_map.svg",
        axes: Sequence[str] | None = None,
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
        # Determine section
        if section_coord is not None:
            if not self.has_section(section_coord):
                self._solve_and_cache(section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self._solve_and_cache(None)
            section = self._section

        # Decide projection
        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
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
        order=6,
        frame="rotating",
        dark_mode: bool = True,
        axes: Sequence[str] | None = None,
        section_coord: str | None = None,
    ):
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
        matplotlib.figure.Figure
            The interactive plot figure.

        Notes
        -----
        Clicking on points in the plot will propagate trajectories from
        those points and display the resulting orbits.
        """
        # Ensure section exists
        if section_coord is not None:
            if not self.has_section(section_coord):
                self._solve_and_cache(section_coord)
            section = self.get_section(section_coord)
        else:
            if self._section is None:
                self._solve_and_cache(None)
            section = self._section

        def _on_select(pt_np: np.ndarray):
            if axes is None:
                section_pt = pt_np
            else:
                prev_sec = self._section
                self._section = section
                try:
                    proj_pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
                finally:
                    self._section = prev_sec
                distances = np.linalg.norm(proj_pts - pt_np, axis=1)
                section_pt = section.points[np.argmin(distances)]

            orbit = self._propagate_from_point(
                section_pt,
                self.energy,
                steps=steps,
                method=method,
                order=order,
            )

            orbit.plot(frame=frame, dark_mode=dark_mode, block=False, close_after=False)

            return orbit

        if axes is None:
            pts = section.points
            lbls = section.labels
        else:
            prev_sec = self._section
            self._section = section
            try:
                pts = self.get_points(section_coord=section_coord, axes=tuple(axes))
            finally:
                self._section = prev_sec
            lbls = tuple(axes)

        return plot_poincare_map_interactive(
            points=pts,
            labels=lbls,
            on_select=_on_select,
            dark_mode=dark_mode,
        )

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
