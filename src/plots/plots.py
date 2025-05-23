import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Any, Dict, Sequence, List
import matplotlib.patheffects as patheffects
import os

from utils.log_config import logger

from utils.coordinates import rotating_to_inertial


def plot_orbit_rotating_frame(trajectory: np.ndarray, 
                              mu: float,
                              system: Any,
                              libration_point: Any,
                              family: str,
                              show: bool = True, 
                              figsize: Tuple[float, float] = (10, 8),
                              dark_mode: bool = True,
                              **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the orbit trajectory in the rotating reference frame.
    
    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory array of shape (steps, 6) containing state vectors.
    mu : float
        Mass parameter of the system.
    system : System
        The CR3BP system containing the bodies.
    libration_point : LibrationPoint
        The libration point object.
    family : str
        The family name of the orbit.
    show : bool, optional
        Whether to call plt.show() after creating the plot. Default is True.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 8).
    dark_mode : bool, optional
        Whether to use dark mode theme to resemble space. Default is True.
    **kwargs
        Additional keyword arguments for plot customization.
        
    Returns
    -------
    tuple
        (fig, ax) containing the figure and axis objects for further customization
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get trajectory data
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    
    # Plot orbit trajectory
    orbit_color = kwargs.get('orbit_color', 'cyan')
    ax.plot(x, y, z, label=f'{family.capitalize()} Orbit', color=orbit_color)
    
    # Plot primary body (canonical position: -mu, 0, 0)
    primary_pos = np.array([-mu, 0, 0])
    primary_radius = system.primary.radius / system.distance  # Convert to canonical units
    plot_body(ax, primary_pos, primary_radius, system.primary.color, system.primary.name)
    
    # Plot secondary body (canonical position: 1-mu, 0, 0)
    secondary_pos = np.array([1-mu, 0, 0])
    secondary_radius = system.secondary.radius / system.distance  # Convert to canonical units
    plot_body(ax, secondary_pos, secondary_radius, system.secondary.color, system.secondary.name)
    
    # Plot libration point
    ax.scatter(*libration_point.position, color='#FF00FF', marker='o', 
              s=5, label=f'{libration_point}')
    
    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')
    
    # Create legend and apply styling
    ax.legend()
    set_axes_equal(ax)
    
    # Apply dark mode if requested
    if dark_mode:
        set_dark_mode(fig, ax, title=f'{family.capitalize()} Orbit in Rotating Frame')
    else:
        ax.set_title(f'{family.capitalize()} Orbit in Rotating Frame')
    
    if show:
        plt.show()
        
    return fig, ax


def plot_orbit_inertial_frame(trajectory: np.ndarray,
                             times: np.ndarray,
                             mu: float,
                             system: Any,
                             family: str,
                             show: bool = True,
                             figsize: Tuple[float, float] = (10, 8),
                             dark_mode: bool = True,
                             **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the orbit trajectory in the primary-centered inertial reference frame.
    
    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory array of shape (steps, 6) containing state vectors.
    times : np.ndarray
        Array of time points corresponding to the trajectory.
    mu : float
        Mass parameter of the system.
    system : System
        The CR3BP system containing the bodies.
    family : str
        The family name of the orbit.
    show : bool, optional
        Whether to call plt.show() after creating the plot. Default is True.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 8).
    dark_mode : bool, optional
        Whether to use dark mode theme to resemble space. Default is True.
    **kwargs
        Additional keyword arguments for plot customization.
        
    Returns
    -------
    tuple
        (fig, ax) containing the figure and axis objects for further customization
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get trajectory data and convert to inertial frame
    traj_inertial = []
    
    for state, t in zip(trajectory, times):
        # Convert rotating frame to inertial frame (canonical units)
        inertial_state = rotating_to_inertial(state, t, mu)
        traj_inertial.append(inertial_state)
    
    traj_inertial = np.array(traj_inertial)
    x, y, z = traj_inertial[:, 0], traj_inertial[:, 1], traj_inertial[:, 2]
    
    # Plot orbit trajectory
    orbit_color = kwargs.get('orbit_color', 'red')
    ax.plot(x, y, z, label=f'{family.capitalize()} Orbit', color=orbit_color)
    
    # Plot primary body at origin
    primary_pos = np.array([0, 0, 0])
    primary_radius = system.primary.radius / system.distance  # Convert to canonical units
    plot_body(ax, primary_pos, primary_radius, system.primary.color, system.primary.name)
    
    # Plot secondary's orbit and position
    theta = times  # Time is angle in canonical units
    secondary_x = (1-mu) * np.cos(theta)
    secondary_y = (1-mu) * np.sin(theta)
    secondary_z = np.zeros_like(theta)
    
    # Plot secondary's orbit
    ax.plot(secondary_x, secondary_y, secondary_z, '--', color=system.secondary.color, 
            alpha=0.5, label=f'{system.secondary.name} Orbit')
    
    # Plot secondary at final position
    secondary_pos = np.array([secondary_x[-1], secondary_y[-1], secondary_z[-1]])
    secondary_radius = system.secondary.radius / system.distance  # Convert to canonical units
    plot_body(ax, secondary_pos, secondary_radius, system.secondary.color, system.secondary.name)
    
    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')
    
    # Create legend and apply styling
    ax.legend()
    set_axes_equal(ax)
    
    # Apply dark mode if requested
    if dark_mode:
        set_dark_mode(fig, ax, title=f'{family.capitalize()} Orbit in Inertial Frame')
    else:
        ax.set_title(f'{family.capitalize()} Orbit in Inertial Frame')
    
    if show:
        plt.show()
        
    return fig, ax


def plot_body(ax, center, radius, color, name, u_res=40, v_res=15):
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


def set_axes_equal(ax):
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


def set_dark_mode(fig: plt.Figure, ax: plt.Axes, title: Optional[str] = None):
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


def plot_poincare_map(pts_list: list, h0_levels: Sequence[float], dark_mode: bool = True, output_dir: Optional[str] = None, filename: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the Poincaré map for multiple energy levels.
    Dynamically adjusts the number of subplots based on the number of energy levels.
    
    Parameters
    ----------
    pts_list : list
        List of numpy arrays containing points for each energy level.
        Each array should have shape (n_points, 2) with columns [q2, p2].
    h0_levels : Sequence[float]
        Energy levels corresponding to each set of points.
    dark_mode : bool, optional
        Whether to use dark mode styling. Default is True.
    output_dir : str, optional
        Directory to save the plot. If None, plot is not saved.
    filename : str, optional
        Filename for the saved plot. If None, plot is not saved.
        
    Returns
    -------
    tuple
        (fig, active_axs) containing the figure and a list of actively used axis objects.
    """
    
    num_levels = len(h0_levels)

    if num_levels == 0:
        logger.info("No H0 levels to plot for Poincaré map.")
        fig, ax = plt.subplots(figsize=(5,5)) 
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', color='red' if not dark_mode else 'white')
        ax.set_xticks([])
        ax.set_yticks([])
        title_text = "Poincaré Map (No Data)"
        if dark_mode:
            set_dark_mode(fig, ax, title=title_text)
        else:
            ax.set_title(title_text)
        plt.show()
        return fig, [ax]

    # Determine layout for subplots
    if num_levels == 1:
        nrows, ncols = 1, 1
    elif num_levels == 2:
        nrows, ncols = 1, 2
    elif num_levels == 3:
        nrows, ncols = 1, 3
    else:  # num_levels >= 4
        ncols = int(np.ceil(np.sqrt(num_levels)))
        nrows = int(np.ceil(num_levels / float(ncols)))

    # squeeze=False ensures ax_or_axs_array is always a 2D numpy array.
    fig, ax_or_axs_array = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5), squeeze=False)
    axs_list = list(ax_or_axs_array.flatten())

    for i, (pts, h0) in enumerate(zip(pts_list, h0_levels)):
        current_ax = axs_list[i]
        
        if pts.shape[0] == 0: 
            logger.info(f"No points to plot for h0={h0:.3f}. Plotting empty subplot.")
            current_ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=9,
                            color='gray' if not dark_mode else '#aaaaaa')
            current_ax.set_xlim(-1, 1) 
            current_ax.set_ylim(-1, 1)
        else:
            current_ax.scatter(pts[:, 0], pts[:, 1], s=1)
            max_val_q2 = max(abs(pts[:, 0].max()), abs(pts[:, 0].min())) if pts.shape[0] > 0 else 1e-9
            max_val_p2 = max(abs(pts[:, 1].max()), abs(pts[:, 1].min())) if pts.shape[0] > 0 else 1e-9
            max_abs_val = max(max_val_q2, max_val_p2, 1e-9) # Ensure not zero for limits
            
            current_ax.set_xlim(-max_abs_val * 1.1, max_abs_val * 1.1)
            current_ax.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)

        current_ax.set_aspect("equal", adjustable="box")
        current_ax.set_xlabel(r"$q_2'$")
        current_ax.set_ylabel(r"$p_2'$")
        
        title_text = f"Poincaré Map h={h0:.3f}"
        if dark_mode:
            set_dark_mode(fig, current_ax, title=title_text)
        else:
            current_ax.set_title(title_text)

    # Hide unused subplots
    for i in range(num_levels, nrows * ncols):
        fig.delaxes(axs_list[i])

    plt.tight_layout()

    if output_dir and filename:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        try:
            fig.savefig(filepath)
            logger.info(f"Poincaré map saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving Poincaré map to {filepath}: {e}")

    plt.show()
    
    active_axs = axs_list[:num_levels]
    return fig, active_axs
