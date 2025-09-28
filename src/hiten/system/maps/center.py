

class CenterManifoldMap:

    def __init__(self, center_manifold: CenterManifold, energy: float):
        pass


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
                logger.debug("Section %s not cached - computing now...", section_coord)
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
                logger.debug("Section %s not cached - computing now...", section_coord)
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


    def save(self, filepath: str, **kwargs) -> None:
        """Save the Poincare map to file.

        Parameters
        ----------
        filepath : str
            Path to save the map data.
        **kwargs
            Additional keyword arguments passed to the save function.
        """
        save_poincare_map(self, filepath, **kwargs)

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load Poincare map data from file in place.

        Parameters
        ----------
        filepath : str
            Path to load the map data from.
        **kwargs
            Additional keyword arguments passed to the load function.
        """
        load_poincare_map_inplace(self, filepath, **kwargs)

    @classmethod
    def load(
        cls,
        filepath: str,
        cm: CenterManifold,
        **kwargs,
    ) -> "CenterManifoldMap":
        """Load a Poincare map from file.

        Parameters
        ----------
        filepath : str
            Path to load the map data from.
        cm : :class:`~hiten.system.center.CenterManifold`
            Center manifold object for the loaded map.
        **kwargs
            Additional keyword arguments passed to the load function.

        Returns
        -------
        :class:`~hiten.algorithms.poincare.centermanifold.base.CenterManifoldMap`
            Loaded center manifold map instance.
        """
        return load_poincare_map(filepath, cm, **kwargs)
