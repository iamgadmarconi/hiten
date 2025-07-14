import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from hiten.algorithms.utils.coordinates import (_get_angular_velocity,
                                                _get_mass_parameter,
                                                _rotating_to_inertial,
                                                _si_time, _to_si_units)
from hiten.system.body import Body
from hiten.utils.io import _ensure_dir


def animate_trajectories(
        states: np.ndarray, 
        times: np.ndarray, 
        bodies: List[Body], 
        system_distance: float, 
        interval: int = 10, 
        figsize: Tuple[int, int] = (14, 6), 
        save: bool = False, 
        dark_mode: bool = True, 
        filepath: str = 'trajectory.mp4'
    ):
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
    t_si = _si_time(times, bodies[0].mass, bodies[1].mass, system_distance)

    traj_rot = np.array([
        _to_si_units(s, bodies[0].mass, bodies[1].mass, system_distance)[:3]
        for s in states
    ])
    
    traj_inert = []
    for s_dimless, t_dimless in zip(states, times):
        s_inert_dimless = _rotating_to_inertial(s_dimless, t_dimless, mu)
        s_inert_si = _to_si_units(s_inert_dimless, bodies[0].mass, bodies[1].mass, system_distance)
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
        
        primary_color = _get_body_color(bodies[0], 'royalblue')
        secondary_color = _get_body_color(bodies[1], 'slategray')
        _plot_body(ax_rot, primary_rot_center, bodies[0].radius, primary_color, bodies[0].name)
        _plot_body(ax_rot, secondary_rot_center, bodies[1].radius, secondary_color, bodies[1].name)
        
        ax_rot.set_title("Rotating Frame (SI Distances)", color='white' if dark_mode else 'black')
        ax_rot.legend()
        
        ax_inert.plot(traj_inert[:frame+1, 0],
                      traj_inert[:frame+1, 1],
                      traj_inert[:frame+1, 2],
                      color='red', label='Particle')
        
        _plot_body(ax_inert, primary_inert_center, bodies[0].radius, primary_color, bodies[0].name)
        
        ax_inert.plot(secondary_x[:frame+1], secondary_y[:frame+1], secondary_z[:frame+1],
                      '--', color='gray', alpha=0.5, label=f'{bodies[1].name} orbit')
        secondary_center_now = np.array([secondary_x[frame], secondary_y[frame], secondary_z[frame]])
        _plot_body(ax_inert, secondary_center_now, bodies[1].radius, secondary_color, bodies[1].name)
        
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

def plot_rotating_frame(
        states: np.ndarray,
        times: np.ndarray,
        bodies: List[Body],
        system_distance: float,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = 'rotating_frame.svg',
        *,
        block: bool = True,
        close_after: bool = True,
        **kwargs):
    r"""
    Plot the orbit trajectory in the rotating reference frame.
    
    Parameters
    ----------
    states : array-like
        The states to plot.
    times : array-like
        The times corresponding to the states.
    bodies : list
        The bodies to plot.
    system_distance : float
        The distance between the bodies.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 8).
        
    Returns
    -------
    tuple
        (fig, ax) containing the figure and axis objects for further customization
    """
    mu = _get_mass_parameter(bodies[0].mass, bodies[1].mass)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get trajectory data
    x = states[:, 0]
    y = states[:, 1]
    z = states[:, 2]
    
    # Plot orbit trajectory
    orbit_color = kwargs.get('orbit_color', 'cyan')
    ax.plot(x, y, z, label=f'Orbit', color=orbit_color)
    
    # Plot primary body (canonical position: -mu, 0, 0)
    primary_pos = np.array([-mu, 0, 0])
    primary_radius = bodies[0].radius / system_distance  # Convert to canonical units
    primary_color = _get_body_color(bodies[0], 'royalblue')
    _plot_body(ax, primary_pos, primary_radius, primary_color, bodies[0].name)
    
    # Plot secondary body (canonical position: 1-mu, 0, 0)
    secondary_pos = np.array([1-mu, 0, 0])
    secondary_radius = bodies[1].radius / system_distance  # Convert to canonical units
    secondary_color = _get_body_color(bodies[1], 'slategray')
    _plot_body(ax, secondary_pos, secondary_radius, secondary_color, bodies[1].name)
    
    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')
    
    # Create legend and apply styling
    ax.legend()
    _set_axes_equal(ax)
    
    # Apply dark mode if requested
    if dark_mode:
        _set_dark_mode(fig, ax, title=f'Orbit in Rotating Frame')
    else:
        ax.set_title(f'Orbit in Rotating Frame')
    
    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)

    plt.show(block=block)
    if close_after:
        plt.close(fig)

    return fig, ax

    
def plot_inertial_frame(
        states: np.ndarray,
        times: np.ndarray,
        bodies: List[Body],
        system_distance: float,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = 'inertial_frame.svg',
        *,
        block: bool = True,
        close_after: bool = True,
        **kwargs):
    r"""
    Plot the orbit trajectory in the primary-centered inertial reference frame.
    
    Parameters
    ----------
    show : bool, optional
        Whether to call plt.show() after creating the plot. Default is True.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 8).
    **kwargs
        Additional keyword arguments for plot customization.
        
    Returns
    -------
    tuple
        (fig, ax) containing the figure and axis objects for further customization
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    mu = _get_mass_parameter(bodies[0].mass, bodies[1].mass)
    # Get trajectory data and convert to inertial frame
    traj_inertial = []
    
    for state, t in zip(states, times):
        # Convert rotating frame to inertial frame (canonical units)
        inertial_state = _rotating_to_inertial(state, t, mu)
        traj_inertial.append(inertial_state)
    
    traj_inertial = np.array(traj_inertial)
    x, y, z = traj_inertial[:, 0], traj_inertial[:, 1], traj_inertial[:, 2]
    
    # Plot orbit trajectory
    orbit_color = kwargs.get('orbit_color', 'red')
    ax.plot(x, y, z, label=f'Orbit', color=orbit_color)
    
    # Plot primary body at origin
    primary_pos = np.array([0, 0, 0])
    primary_radius = bodies[0].radius / system_distance  # Convert to canonical units
    primary_color = _get_body_color(bodies[0], 'royalblue')
    _plot_body(ax, primary_pos, primary_radius, primary_color, bodies[0].name)
    
    # Plot secondary's orbit and position
    theta = times  # Time is angle in canonical units
    secondary_x = (1-mu)
    secondary_y = np.zeros_like(theta)
    secondary_z = np.zeros_like(theta)
    
    # Plot secondary's orbit
    ax.plot(secondary_x, secondary_y, secondary_z, '--', color=bodies[1].color, 
            alpha=0.5, label=f'{bodies[1].name} Orbit')
    
    # Plot secondary at final position
    secondary_pos = np.array([secondary_x[-1], secondary_y[-1], secondary_z[-1]])
    secondary_radius = bodies[1].radius / system_distance  # Convert to canonical units
    secondary_color = _get_body_color(bodies[1], 'slategray')
    _plot_body(ax, secondary_pos, secondary_radius, secondary_color, bodies[1].name)
    
    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')
    
    # Create legend and apply styling
    ax.legend()
    _set_axes_equal(ax)
    
    # Apply dark mode if requested
    if dark_mode:
        _set_dark_mode(fig, ax, title=f'Orbit in Inertial Frame')
    else:
        ax.set_title(f'Orbit in Inertial Frame')
    
    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)
    
    plt.show(block=block)
    if close_after:
        plt.close(fig)
        
    return fig, ax


def plot_orbit_family(
        states_list: List[np.ndarray],
        times_list: List[np.ndarray],
        parameter_values: np.ndarray,
        bodies: List[Body],
        system_distance: float,
        *,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = "orbit_family.svg",
        **kwargs):
    r"""
    Visualise a family of periodic orbits (rotating-frame trajectories).

    Parameters
    ----------
    states_list, times_list : list[array]
        Trajectory arrays for each family member.
    parameter_values : array-like
        Scalar parameter associated with each orbit (used for colour-coding).
    bodies : list[Body]
        Primary and secondary bodies of the system.
    system_distance : float
        Characteristic distance (km) - needed to scale body radii.
    figsize, save, dark_mode, filepath, cmap : see other plot helpers.
    elev, azim : float, optional
        Elevation and azimuth angles for 3D view.
    equal_axes : bool, optional
        Whether to use equal scaling for all axes. Default is True.
    """

    cmap_key = kwargs.get('cmap', 'plasma')
    elev = kwargs.get('elev', None)
    azim = kwargs.get('azim', None)
    equal_axes = kwargs.get('equal_axes', True)

    param_index = kwargs.get('param_index', None)

    param_arr = np.asarray(parameter_values, dtype=float)
    if param_arr.ndim == 2 and param_arr.shape[1] > 1:
        if param_index is not None:
            if not (0 <= param_index < param_arr.shape[1]):
                raise ValueError(
                    f"param_index={param_index} out of bounds for parameter_values with shape {param_arr.shape}"
                )
            scalar_param = param_arr[:, param_index]
        else:
            # Default: use Euclidean norm across components
            scalar_param = np.linalg.norm(param_arr, axis=1)
    else:
        # Already 1-D (or a list/tuple of scalars)
        scalar_param = param_arr.squeeze()

    if len(states_list) != len(scalar_param):
        raise ValueError("states_list and parameter_values length mismatch after flattening")

    mu = _get_mass_parameter(bodies[0].mass, bodies[1].mass)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Use matplotlib's colour normalisation utilities (not available via `plt`)
    norm = mpl.colors.Normalize(vmin=float(np.min(scalar_param)), vmax=float(np.max(scalar_param)))
    sm = plt.cm.ScalarMappable(cmap=cmap_key, norm=norm)

    for traj, val in zip(states_list, scalar_param):
        color = sm.to_rgba(float(val))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.2)

    # plot primaries in rotating frame (-mu,0,0) and (1-mu,0,0)
    primary_pos = np.array([-mu, 0, 0])
    secondary_pos = np.array([1 - mu, 0, 0])
    _plot_body(ax, primary_pos, bodies[0].radius / system_distance, _get_body_color(bodies[0], 'royalblue'), bodies[0].name)
    _plot_body(ax, secondary_pos, bodies[1].radius / system_distance, _get_body_color(bodies[1], 'slategray'), bodies[1].name)

    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')
    
    # Apply equal axes scaling only if requested
    if equal_axes:
        _set_axes_equal(ax)
    
    # Set custom view angle if provided
    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)
    
    ax.set_title('Orbit family')

    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Continuation parameter')

    if dark_mode:
        _set_dark_mode(fig, ax, title=ax.get_title())
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)

    plt.show()
    plt.close(fig)
    return fig, ax


def plot_manifold(
        states_list: List[np.ndarray], 
        times_list: List[np.ndarray], 
        bodies: List[Body], 
        system_distance: float, 
        figsize: Tuple[int, int] = (10, 8), 
        save: bool = False, 
        dark_mode: bool = True, 
        filepath: str = 'manifold.svg',
        **kwargs):
    r"""
    Plot the manifold.
    
    Parameters
    ----------
    states_list : list
        The states to plot.
    times_list : list
        The times corresponding to the states.
    bodies : list
        The bodies to plot.
    system_distance : float
        The distance between the bodies.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 8).
    save : bool, optional
        Whether to save the plot to a file. Default is False.
    dark_mode : bool, optional
        Whether to use dark mode for the plot. Default is True.

    Returns
    -------
    tuple
        (fig, ax) containing the figure and axis objects for further customization
    """

    cmap_key = kwargs.get('cmap', 'plasma')
    mu = _get_mass_parameter(bodies[0].mass, bodies[1].mass)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    num_traj = len(states_list)
    cmap = plt.get_cmap(cmap_key)

    for i, (xW, _) in enumerate(zip(states_list, times_list)):
        color = cmap(i / (num_traj - 1)) if num_traj > 1 else cmap(0.5)
        ax.plot(xW[:, 0], xW[:, 1], xW[:, 2], color=color, lw=2)

    primary_color = _get_body_color(bodies[0], 'royalblue')
    primary_center = np.array([-mu, 0, 0])
    primary_radius = bodies[0].radius
    _plot_body(ax, primary_center, primary_radius / system_distance, primary_color, bodies[0].name)

    secondary_color = _get_body_color(bodies[1], 'slategray')
    secondary_center = np.array([(1 - mu), 0, 0])
    secondary_radius = bodies[1].radius
    _plot_body(ax, secondary_center, secondary_radius / system_distance, secondary_color, bodies[1].name)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    _set_axes_equal(ax)
    ax.set_title('Manifold')

    if dark_mode:
        _set_dark_mode(fig, ax, title=ax.get_title())

    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)

    plt.show()
    plt.close()

    return fig, ax

def plot_poincare_map(
        points: np.ndarray,
        labels: List[str],
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = 'poincare_map.svg',
        **kwargs):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if points.shape[0] == 0:
        ax.text(0.5, 0.5, "No data to plot", ha='center', va='center', 
                color='red' if not dark_mode else 'white', fontsize=12)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        title_text = "Poincaré Map (No Data)"

    else:
        n_pts = points.shape[0]
        point_size = max(0.2, min(4.0, 4000.0 / max(n_pts, 1)))

        ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=0.7)
        
        max_val_0 = max(abs(points[:, 0].max()), abs(points[:, 0].min()))
        max_val_1 = max(abs(points[:, 1].max()), abs(points[:, 1].min()))
        max_abs_val = max(max_val_0, max_val_1, 1e-9)
        
        ax.set_xlim(-max_abs_val * 1.1, max_abs_val * 1.1)
        ax.set_ylim(-max_abs_val * 1.1, max_abs_val * 1.1)
        
        ax.set_xlabel(f"${labels[0]}'$")
        ax.set_ylabel(f"${labels[1]}'$")
        title_text = f"Poincaré Map"

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    
    if dark_mode:
        _set_dark_mode(fig, ax, title=title_text)
    else:
        ax.set_title(title_text)

    plt.tight_layout()

    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)

    plt.show()
    plt.close()

    return fig, ax

def plot_poincare_map_interactive(
        points: np.ndarray,
        labels: List[str],
        on_select: Optional[Callable[[np.ndarray], Any]] = None,
        figsize: Tuple[int, int] = (10, 8),
        dark_mode: bool = True,
        **kwargs):
    """Interactive Poincaré-map viewer.

    Parameters
    ----------
    points : numpy.ndarray, shape (n, 2)
        Collection of Poincaré-section points to visualise.
    labels : list[str]
        Axis labels corresponding to *points* (e.g. ["q2", "p2"]).
    on_select : callable, optional
        Callback executed when a point is selected with the left mouse button.
        It receives the coordinates of the selected point *(2,)* and may return
        an arbitrary object (e.g. a `GenericOrbit`).  The last returned value is
        also the return value of this function.
    figsize : tuple[int, int], default (10, 8)
        Figure size in inches.
    dark_mode : bool, default True
        Use a dark colour scheme.
    **kwargs
        Currently ignored.  Reserved for future extensions.
    """

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    pts = points  # Local alias inside closure

    # Adaptive point size for clarity on dense maps
    n_pts_int = pts.shape[0]
    adaptive_ps = max(0.2, min(4.0, 4000.0 / max(n_pts_int, 1)))

    # Scatter plot of the Poincaré set
    ax.scatter(pts[:, 0], pts[:, 1], s=adaptive_ps, alpha=0.7)
    ax.set_xlabel(f"${labels[0]}'$")
    ax.set_ylabel(f"${labels[1]}'$")
    ax.set_title("Select a point on the Poincaré Map\n(Press 'q' to quit)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if dark_mode:
        _set_dark_mode(fig, ax, title=ax.get_title())

    # Marker highlighting the selected point
    selected_marker = ax.scatter([], [], s=60, c='red', marker='x')
    selected_payload: Dict[str, Any] = {"orbit": None}

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _onclick(event):
        """Handle mouse-click events inside the axes."""
        if event.inaxes != ax or event.button != 1:  # Left click only
            return

        # Find the closest point to the click position
        click_xy = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(pts - click_xy, axis=1)
        idx = int(np.argmin(distances))
        pt = pts[idx]

        # Update marker position
        selected_marker.set_offsets(np.array([[pt[0], pt[1]]]))
        fig.canvas.draw_idle()

        # Execute user callback, if any
        if on_select is not None:
            try:
                selected_payload["orbit"] = on_select(pt)
            except Exception as exc:  # pragma: no cover
                raise Exception(f"Error in on_select callback: {exc}")

    def _onkey(event):
        """Close the figure when the user presses *q*."""
        if event.key == 'q':
            plt.close(fig)

    # Register the callbacks
    fig.canvas.mpl_connect('button_press_event', _onclick)
    fig.canvas.mpl_connect('key_press_event', _onkey)

    # Start the UI loop (blocking)
    plt.show()
    plt.close(fig)

    return selected_payload.get("orbit")

def plot_invariant_torus(
        u_grid: np.ndarray,
        bodies: List[Body],
        system_distance: float,
        *,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = "invariant_torus.svg",
        cmap: str = "plasma",
        elev: Optional[float] = None,
        azim: Optional[float] = None,
        equal_axes: bool = True,
        **kwargs):
    """Visualise an invariant torus in the rotating frame.

    Parameters
    ----------
    u_grid : numpy.ndarray
        Grid of state vectors with shape (n_theta1, n_theta2, 6) returned by
        :pyfunc:`_InvariantTori.sample_grid`.
    bodies : list[Body]
        Primary and secondary bodies of the CR3BP.
    system_distance : float
        Characteristic distance (km) - required to scale the body radii.
    figsize, save, dark_mode, filepath : see other plot helpers.
    cmap : str, optional
        Colormap name used to colour-code the major-angle \theta₁. Default ``"plasma"``.
    elev, azim : float, optional
        Elevation and azimuth view angles passed to ``Axes3D.view_init``.
    equal_axes : bool, default True
        If True, enforce identical scaling on all axes.
    **kwargs
        Additional keyword arguments reserved for future extensions.
    """
    n_theta1, n_theta2, _ = u_grid.shape

    # Mass parameter for canonical positions of the primaries
    mu = _get_mass_parameter(bodies[0].mass, bodies[1].mass)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    colour_map = plt.get_cmap(cmap)
    for i in range(n_theta1):
        traj = u_grid[i, :, :3]  # (n_theta2, 3) - use spatial components only
        color = colour_map(i / max(n_theta1 - 1, 1))
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, lw=1.0)

    # Plot primary and secondary in rotating-frame canonical positions
    primary_pos = np.array([-mu, 0, 0])
    secondary_pos = np.array([1 - mu, 0, 0])
    _plot_body(ax, primary_pos, bodies[0].radius / system_distance,
               _get_body_color(bodies[0], 'royalblue'), bodies[0].name)
    _plot_body(ax, secondary_pos, bodies[1].radius / system_distance,
               _get_body_color(bodies[1], 'slategray'), bodies[1].name)

    ax.set_xlabel('X [canonical]')
    ax.set_ylabel('Y [canonical]')
    ax.set_zlabel('Z [canonical]')

    if elev is not None or azim is not None:
        ax.view_init(elev=elev, azim=azim)

    if equal_axes:
        _set_axes_equal(ax)

    ax.set_title('Invariant Torus')

    if dark_mode:
        _set_dark_mode(fig, ax, title=ax.get_title())

    if save:
        _ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        plt.savefig(filepath)

    plt.show()
    plt.close(fig)
    return fig, ax

def _get_body_color(body: Body, default_color: str) -> str:
    """
    Determines the color for a celestial body in a plot.

    It returns the color specified in the `Body` object, unless the color is
    the default black ("#000000"), in which case it returns a specified default
    color. This ensures visibility in both light and dark modes.

    Parameters
    ----------
    body : Body
        The celestial body object.
    default_color : str
        The color to use if the body's color is the default black.

    Returns
    -------
    str
        The determined color string.
    """
    if body.color and body.color.upper() != '#000000':
        return body.color
    return default_color

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
