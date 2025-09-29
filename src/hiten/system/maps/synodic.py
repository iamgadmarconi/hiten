

class SynodicMap:


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

