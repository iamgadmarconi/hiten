import os
from typing import Optional

import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from hiten.utils.coordinates import (_get_angular_velocity, rotating_to_inertial,
                               si_time, to_si_units)
from hiten.utils.files import _ensure_dir


def animate_trajectories(states, times, bodies, system_distance, interval=10, figsize=(14, 6), save=False, dark_mode: bool = True, filepath: str = 'trajectory.mp4'):
    """
    Create an animated comparison of trajectories in rotating and inertial frames.
    
    Parameters
    ----------
    sol : integration result
    bodies : list
        List of celestial body objects with properties like mass, radius, and name.
    system_distance : float
        Characteristic distance of the system in meters.
    interval : int, default=20
        Time interval between animation frames in milliseconds.
    figsize : tuple, default=(14, 6)
        Figure size in inches (width, height).
    save : bool, default=False
        Whether to save the animation as an MP4 file.
        
    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object.
        
    Notes
    -----
    This function creates a side-by-side animation showing the trajectory in both
    rotating and inertial frames, with consistent axis scaling to maintain proper
    proportions. The animation shows the motion of celestial bodies and the particle
    over time, with time displayed in days.
    """

    fig = plt.figure(figsize=figsize)
    ax_rot = fig.add_subplot(121, projection='3d')
    ax_inert = fig.add_subplot(122, projection='3d')

    mu = bodies[1].mass / (bodies[0].mass + bodies[1].mass)
    omega_real = _get_angular_velocity(bodies[0].mass, bodies[1].mass, system_distance)
    t_si = si_time(times, bodies[0].mass, bodies[1].mass, system_distance)

    traj_rot = np.array([
        to_si_units(s, bodies[0].mass, bodies[1].mass, system_distance)[:3]
        for s in states
    ])
    
    traj_inert = []
    for s_dimless, t_dimless in zip(states, times):
        s_inert_dimless = rotating_to_inertial(s_dimless, t_dimless, mu)
        s_inert_si = to_si_units(s_inert_dimless, bodies[0].mass, bodies[1].mass, system_distance)
        traj_inert.append(s_inert_si[:3])
    traj_inert = np.array(traj_inert)
    
    secondary_x = system_distance * np.cos(omega_real * t_si)
    secondary_y = system_distance * np.sin(omega_real * t_si)
    secondary_z = np.zeros_like(secondary_x)

    primary_rot_center = np.array([-mu*system_distance, 0, 0])
    secondary_rot_center = np.array([(1.0 - mu)*system_distance, 0, 0])
    
    primary_inert_center = np.array([0, 0, 0])

    # ------------------------------------------------------------------
    # Pre-compute global axis limits so that zoom/scale is persistent.
    # ------------------------------------------------------------------
    coords_list = [
        traj_rot,
        traj_inert,
        np.stack([secondary_x, secondary_y, secondary_z], axis=1),
        primary_rot_center[None, :],
        secondary_rot_center[None, :],
        primary_inert_center[None, :],
    ]
    all_coords = np.vstack(coords_list)
    xyz_min = all_coords.min(axis=0)
    xyz_max = all_coords.max(axis=0)
    span = np.max(xyz_max - xyz_min)
    half_span = 0.55 * span  # slight padding (10%)
    center = 0.5 * (xyz_max + xyz_min)
    x_limits = (center[0] - half_span, center[0] + half_span)
    y_limits = (center[1] - half_span, center[1] + half_span)
    z_limits = (center[2] - half_span, center[2] + half_span)

    # Store initial view angles to keep orientation persistent
    init_elev_rot, init_azim_rot = ax_rot.elev, ax_rot.azim
    init_elev_inert, init_azim_inert = ax_inert.elev, ax_inert.azim

    # ----- Persisted view state (updated every frame, used by init on repeats) -----
    view_state = {
        'rot_elev': ax_rot.elev,
        'rot_azim': ax_rot.azim,
        'inert_elev': ax_inert.elev,
        'inert_azim': ax_inert.azim,
        'rot_xlim': x_limits,
        'rot_ylim': y_limits,
        'rot_zlim': z_limits,
        'inert_xlim': x_limits,
        'inert_ylim': y_limits,
        'inert_zlim': z_limits,
    }

    def init():
        """
        Initialize the animation (also called at every repeat).
        Uses the most recently stored `view_state` so the view chosen by the
        user persists across loops.
        """
        for ax, elev, azim, xl, yl, zl in (
            (ax_rot, view_state['rot_elev'], view_state['rot_azim'], view_state['rot_xlim'], view_state['rot_ylim'], view_state['rot_zlim']),
            (ax_inert, view_state['inert_elev'], view_state['inert_azim'], view_state['inert_xlim'], view_state['inert_ylim'], view_state['inert_zlim']),
        ):
            ax.clear()
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_zlim(zl)
            ax.view_init(elev=elev, azim=azim)
        
        ax_rot.set_title("Rotating Frame (SI Distances)", color='white' if dark_mode else 'black')
        ax_inert.set_title("Inertial Frame (Real Time, Real Ω)", color='white' if dark_mode else 'black')
        if dark_mode:
            _set_dark_mode(fig, ax_rot, title=ax_rot.get_title())
            _set_dark_mode(fig, ax_inert, title=ax_inert.get_title())
        return fig,
    
    def update(frame):
        """
        Update the animation for each frame.
        
        Parameters
        ----------
        frame : int
            The current frame number.
            
        Returns
        -------
        tuple
            A tuple containing the figure and the axes.

        Notes
        -----
        Updates the plot for the current frame, clearing the axes and
        setting the title and labels.
        """
        # Capture existing user view/zoom before clearing (ensures persistence)
        elev_rot_prev, azim_rot_prev = ax_rot.elev, ax_rot.azim
        elev_inert_prev, azim_inert_prev = ax_inert.elev, ax_inert.azim

        xlim_rot_prev, ylim_rot_prev, zlim_rot_prev = ax_rot.get_xlim(), ax_rot.get_ylim(), ax_rot.get_zlim()
        xlim_inert_prev, ylim_inert_prev, zlim_inert_prev = ax_inert.get_xlim(), ax_inert.get_ylim(), ax_inert.get_zlim()

        ax_rot.cla()
        ax_inert.cla()

        # Restore view/limits captured this frame
        for ax, elev, azim, xl, yl, zl in (
                (ax_rot, elev_rot_prev, azim_rot_prev, xlim_rot_prev, ylim_rot_prev, zlim_rot_prev),
                (ax_inert, elev_inert_prev, azim_inert_prev, xlim_inert_prev, ylim_inert_prev, zlim_inert_prev),
            ):
            ax.clear()
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_zlim(zl)
            ax.view_init(elev=elev, azim=azim)
        
        # Save these settings for the init() function on the next repeat
        view_state['rot_elev'] = elev_rot_prev
        view_state['rot_azim'] = azim_rot_prev
        view_state['inert_elev'] = elev_inert_prev
        view_state['inert_azim'] = azim_inert_prev
        view_state['rot_xlim'] = xlim_rot_prev
        view_state['rot_ylim'] = ylim_rot_prev
        view_state['rot_zlim'] = zlim_rot_prev
        view_state['inert_xlim'] = xlim_inert_prev
        view_state['inert_ylim'] = ylim_inert_prev
        view_state['inert_zlim'] = zlim_inert_prev

        current_t_days = t_si[frame] / 86400.0
        fig.suptitle(f"Time = {current_t_days:.2f} days", color='white' if dark_mode else 'black')
        
        ax_rot.plot(traj_rot[:frame+1, 0],
                    traj_rot[:frame+1, 1],
                    traj_rot[:frame+1, 2],
                    color='red', label='Particle')
        
        _plot_body(ax_rot, primary_rot_center, bodies[0].radius, 'blue', bodies[0].name)
        _plot_body(ax_rot, secondary_rot_center, bodies[1].radius, 'gray', bodies[1].name)
        
        ax_rot.set_title("Rotating Frame (SI Distances)", color='white' if dark_mode else 'black')
        ax_rot.legend()
        
        ax_inert.plot(traj_inert[:frame+1, 0],
                      traj_inert[:frame+1, 1],
                      traj_inert[:frame+1, 2],
                      color='red', label='Particle')
        
        _plot_body(ax_inert, primary_inert_center, bodies[0].radius, 'blue', bodies[0].name)
        
        ax_inert.plot(secondary_x[:frame+1], secondary_y[:frame+1], secondary_z[:frame+1],
                      '--', color='gray', alpha=0.5, label=f'{bodies[1].name} orbit')
        secondary_center_now = np.array([secondary_x[frame], secondary_y[frame], secondary_z[frame]])
        _plot_body(ax_inert, secondary_center_now, bodies[1].radius, 'gray', bodies[1].name)
        
        ax_inert.set_title("Inertial Frame (Real Time, Real Ω)", color='white' if dark_mode else 'black')
        ax_inert.legend()
        
        # Ensure dark-mode styling (including legend) is applied for this frame
        if dark_mode:
            _set_dark_mode(fig, ax_rot, title=ax_rot.get_title())
            _set_dark_mode(fig, ax_inert, title=ax_inert.get_title())
        
        return fig,
    
    total_frames = len(times)
    frames_to_use = range(0, total_frames, 30)  # e.g. step by 5

    ani = animation.FuncAnimation(
        fig, update,
        frames=frames_to_use,
        init_func=init,
        interval=interval,
        blit=False
    )
    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        ani.save(filepath, fps=60, dpi=500)
    plt.show()
    plt.close()
    return ani

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
    
    ax.scatter(center[0], center[1], center[2], color=color, s=20)
    
    text_obj = ax.text(center[0], center[1], center[2] + 1.5*radius, name, 
                       color='white',
                       fontweight='bold',
                       fontsize=12,
                       ha='center')
    
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
