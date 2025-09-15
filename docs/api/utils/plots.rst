Plotting Module
===============

The plots module provides a wide range of visualization functions for orbits, manifolds, Poincare maps, and other dynamical system visualizations in the Circular Restricted Three-Body Problem (CR3BP).

This module provides a comprehensive set of plotting utilities for visualizing dynamical systems, including orbit trajectories, invariant manifolds, Poincare maps, and invariant tori. All plotting functions support both light and dark modes, and most functions include options for saving plots to files. Coordinate systems are clearly labeled with appropriate units.

.. currentmodule:: hiten.utils.plots

Animation Functions
-------------------

Functions for creating animated visualizations.

.. autofunction:: animate_trajectories

Orbit Plotting Functions
------------------------

Functions for plotting orbit trajectories.

.. autofunction:: plot_rotating_frame

.. autofunction:: plot_inertial_frame

.. autofunction:: plot_orbit_family

Manifold Plotting Functions
---------------------------

Functions for plotting invariant manifolds.

.. autofunction:: plot_manifold

.. autofunction:: plot_manifolds

Poincare Map Functions
----------------------

Functions for plotting Poincare maps.

.. autofunction:: plot_poincare_map

.. autofunction:: plot_poincare_connections_map

.. autofunction:: plot_poincare_map_interactive

Invariant Torus Functions
-------------------------

Functions for plotting invariant tori.

.. autofunction:: plot_invariant_torus

Helper Functions
----------------

Internal helper functions for plotting.

.. autofunction:: _get_body_color

.. autofunction:: _plot_body

.. autofunction:: _set_axes_equal

.. autofunction:: _set_dark_mode
