from typing import Optional

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def _plot_body(ax, center, radius, color, name, u_res=40, v_res=15):
    """
    Helper method to plot a celestial body as a sphere.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D axes to plot on.
    center : array-like
        The (x, y, z) coordinates of the body center.
    radius : float
        The radius of the body in canonical units.
    color : str
        The color to use for the body.
    name : str
        The name of the body to use in the label.
    u_res : int, optional
        Resolution around the circumference (longitude). Default is 40.
    v_res : int, optional
        Resolution from pole to pole (latitude). Default is 30.
    """
    u, v = np.mgrid[0:2*np.pi:u_res*1j, 0:np.pi:v_res*1j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=0.9)
    
    # Add a small marker for the center of the body
    ax.scatter(center[0], center[1], center[2], color=color, s=20)
    
    # Add high contrast text label
    text_obj = ax.text(center[0], center[1], center[2] + 1.5*radius, name, 
                       color='white',
                       fontweight='bold',
                       fontsize=12,
                       ha='center')
    
    # Add a subtle outline for even better contrast
    text_obj.set_path_effects([
        patheffects.withStroke(linewidth=1.5, foreground='black')
    ])


def _set_axes_equal(ax):
    """
    Set 3D plot axes to equal scale.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The 3D axes to adjust.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def _set_dark_mode(fig: plt.Figure, ax: plt.Axes, title: Optional[str] = None):
    """
    Apply dark mode styling to the figure and axes.
    Handles both 2D and 3D axes.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to apply dark mode styling to.
    ax : matplotlib.axes.Axes
        The 2D or 3D axes to apply dark mode styling to.
    title : str, optional
        The title to set with appropriate dark mode styling.
    """
    text_color = 'white'
    grid_color = '#555555'  # A medium-dark gray for grid lines

    # Set dark background for the entire figure
    fig.patch.set_facecolor('black')

    # Set dark background for the specific axes object
    ax.set_facecolor('black')

    # Common text and tick color settings
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis='x', colors=text_color, which='both') # Apply to major and minor ticks
    ax.tick_params(axis='y', colors=text_color, which='both')

    if isinstance(ax, Axes3D):
        # 3D specific settings
        ax.zaxis.label.set_color(text_color)
        ax.tick_params(axis='z', colors=text_color, which='both')

        # Make panes transparent and set edge color
        for axis_obj in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis_obj.pane.fill = False
            axis_obj.pane.set_edgecolor('black') 

        # Style grid for 3D plots
        ax.grid(True, color=grid_color, linestyle=':', linewidth=0.5)
    else:
        # 2D specific settings
        # Style grid for 2D plots
        ax.grid(True, color=grid_color, linestyle=':', linewidth=0.5)
        
        # Set spine colors for 2D plots to make them visible
        for spine_key in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine_key].set_color(text_color)
            ax.spines[spine_key].set_linewidth(0.5)

    # Set title if provided, with dark mode color
    if title:
        ax.set_title(title, color=text_color)
    
    # Style legend if it exists
    if ax.get_legend():
        legend = ax.get_legend()
        frame = legend.get_frame()
        frame.set_facecolor('#111111')  # Dark background for legend
        frame.set_edgecolor(text_color)   # White border for legend
        
        for text_obj in legend.get_texts():
            text_obj.set_color(text_color) # White text for legend
