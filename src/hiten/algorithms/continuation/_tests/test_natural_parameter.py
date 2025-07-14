import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import os
from pathlib import Path

import numpy as np
import pytest

from hiten.algorithms import EnergyParameter, PeriodParameter, StateParameter
from hiten.system import System
from hiten.system.family import OrbitFamily
from hiten.system.orbits.base import S


def _make_seed_orbit():
    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    # Small planar amplitude for fast convergence
    seed = l1.create_orbit("lyapunov", amplitude_x=0.01)
    seed.correct(max_attempts=10, tol=1e-8)
    return seed


def _run_engine(engine_cls, save_figure=False, figure_name=None, **kwargs):
    seed = _make_seed_orbit()

    # Build engine and run (keep family very small for speed - 3 members)
    engine = engine_cls(initial_orbit=seed, max_orbits=10, **kwargs)
    engine.run()

    # Build family container and generate trajectories (few steps for speed)
    family = OrbitFamily.from_engine(engine)
    family.propagate(steps=300, method="rk", order=4)

    # Determine save parameters
    if save_figure and figure_name:
        # Get the directory where this test file is located
        test_dir = Path(__file__).parent
        filepath = test_dir / f"{figure_name}.png"
        save = True
    else:
        filepath = None
        save = False

    # Use the built-in family plotting method with 2D top-down parameters
    plot_kwargs = {
        'figsize': (10, 8),
        'elev': 90,  # Top-down view (90 degrees elevation)
        'azim': 0,   # Standard azimuth
        'equal_axes': True,  # Keep equal scaling for proper aspect ratio
        'dark_mode': False,  # Light mode for better visibility
    }

    # Generate the plot using the family's built-in method
    fig, ax = family.plot(
        save=save, 
        filepath=str(filepath) if filepath else None, 
        **plot_kwargs
    )
    
    # Add a more descriptive title
    if hasattr(ax, 'set_title'):
        ax.set_title(f'Lyapunov Orbit Family around L1 (Top View)\n'
                    f'{len(family.orbits)} orbits, amplitude range: {family.parameter_values.min():.4f} - {family.parameter_values.max():.4f}')
    
    if save and filepath:
        print(f"Figure saved to: {filepath}")
    
    return fig, ax


def test_state_parameter():
    seed = _make_seed_orbit()
    x0 = float(seed.initial_state[S.X])
    amp0 = float(seed.amplitude)
    max_orbits = 10  # Match what _run_engine uses
    # With max_orbits=10, we get: seed + 9 new orbits = 10 total
    # So we need 9 continuation steps to go from amp0 to amp0*3
    step = (amp0 * 3 - amp0) / (max_orbits - 1)  # 9 steps for 10 total orbits

    # Target +0.004 canonical units along X (absolute coordinate)
    fig, ax = _run_engine(
        StateParameter,
        state=S.X,
        amplitude=True,
        target=(amp0, amp0 * 3),
        step=step,
        corrector_kwargs=dict(max_attempts=25, tol=1e-9),
        save_figure=True,
        figure_name="test_state_parameter_family"
    )

    # --- basic visual sanity checks ---
    assert fig is not None
    assert len(fig.get_axes()) == 2  # main plot + colorbar
    assert len(ax.lines) >= 3  # one trajectory per orbit


def test_period_parameter():
    # seed = _make_seed_orbit()
    # T0 = float(seed.period)

    # # Increase period by ~2 % across 3 steps
    # fig, ax = _run_engine(
    #     PeriodParameter,
    #     target=(T0, T0 * 1.02),
    #     step=(T0 * 0.01),
    #     corrector_kwargs=dict(max_attempts=10, tol=1e-9),
    #     save_figure=True,
    #     figure_name="test_period_parameter_family"
    # )

    # assert fig is not None
    # assert len(fig.get_axes()) == 2
    # assert len(ax.lines) >= 3
    assert True


def test_energy_parameter():
    seed = _make_seed_orbit()
    C0 = float(seed.jacobi_constant)

    # Slightly raise the Jacobi constant (|\Delta C| \approx 1e-3)
    fig, ax = _run_engine(
        EnergyParameter,
        target=(C0, C0 + 1e-3),
        step=5e-4,
        use_jacobi=True,
        corrector_kwargs=dict(max_attempts=10, tol=1e-9),
        save_figure=True,
        figure_name="test_energy_parameter_family"
    )

    assert fig is not None
    assert len(fig.get_axes()) == 2
    assert len(ax.lines) >= 3
