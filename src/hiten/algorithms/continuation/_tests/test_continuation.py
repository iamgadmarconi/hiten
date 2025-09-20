import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for tests

import os
from pathlib import Path

import numpy as np
import pytest

from hiten.algorithms import StateParameter
from hiten.system import System
from hiten.system.family import OrbitFamily
from hiten.algorithms.utils.types import SynodicState


def _make_seed_orbit():
    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    # Small planar amplitude for fast convergence
    seed = l1.create_orbit("lyapunov", amplitude_x=0.01)
    seed.correct(max_attempts=10, tol=1e-8)
    return seed


def _run_facade(*, state, target, step, save_figure=False, figure_name=None, **kwargs):
    seed = _make_seed_orbit()

    result = StateParameter().solve(
        seed=seed,
        state=state,
        target=target,
        step=step,
        max_members=10,
        extra_params=kwargs.get('corrector_kwargs', {}),
    )

    family = OrbitFamily.from_result(result, parameter_name="state")
    family.propagate(steps=300, method="fixed", order=4)

    if save_figure and figure_name:
        test_dir = Path(__file__).parent
        filepath = test_dir / f"{figure_name}.png"
        save = True
    else:
        filepath = None
        save = False

    plot_kwargs = {
        'figsize': (10, 8),
        'elev': 90,
        'azim': 0,
        'equal_axes': True,
        'dark_mode': False,
    }

    fig, ax = family.plot(
        save=save,
        filepath=str(filepath) if filepath else None,
        **plot_kwargs
    )

    if hasattr(ax, 'set_title'):
        ax.set_title(
            f'Lyapunov/Halo Orbit Family around L1 (Top View)\n'
            f'{len(family.orbits)} orbits, param range: {family.parameter_values.min():.4f} - {family.parameter_values.max():.4f}'
        )

    if save and filepath:
        print(f"Figure saved to: {filepath}")

    return fig, ax


def test_state_parameter():
    seed = _make_seed_orbit()
    x0 = float(seed.initial_state[SynodicState.X])
    amp0 = float(seed.amplitude)
    max_orbits = 10  # Match what _run_engine uses
    # With max_orbits=10, we get: seed + 9 new orbits = 10 total
    # So we need 9 continuation steps to go from amp0 to amp0*3
    step = (amp0 * 3 - amp0) / (max_orbits - 1)  # 9 steps for 10 total orbits

    # Target +0.004 canonical units along X (absolute coordinate)
    fig, ax = _run_facade(
        state=SynodicState.X,
        target=(amp0, amp0 * 3),
        step=step,
        corrector_kwargs=dict(max_attempts=25, tol=1e-9),
        save_figure=True,
        figure_name="test_state_parameter_family",
    )

    # --- basic visual sanity checks ---
    assert fig is not None
    assert len(fig.get_axes()) == 2  # main plot + colorbar
    assert len(ax.lines) >= 3  # one trajectory per orbit
