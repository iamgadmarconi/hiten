"""Example script: continuation-based generation of a Halo-orbit halo_family.

Run with
    python examples/orbit_family.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import OrbitFamily, System
from hiten.algorithms import StateParameter
from hiten.algorithms.utils.types import SynodicState


def main() -> None:
    """Generate and save a small Halo halo_family around the Earth-Moon L1 point.
    
    This example demonstrates how to use the StateParameter predictor to
    generate a halo_family of Halo orbits around the Earth-Moon L1 point.
    """
    num_orbits = 5
    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)
    
    # --- Halo seed and state parameter engine ---
    halo_seed = l1.create_orbit('halo', amplitude_z= 0.2, zenith='southern')
    halo_seed.correct(max_attempts=25, max_delta=1e-3)
    # --- two-parameter continuation: vary absolute X (in-plane) and Z (out-of-plane) ---
    current_x = halo_seed.initial_state[SynodicState.X]
    current_z = halo_seed.initial_state[SynodicState.Z]  # 0 for planar Lyapunov halo_seed
    # --- Target displacements (CRTBP canonical units) ---
    target_x = current_x + 0.01   # small shift along X
    target_z = current_z + 0.3   # introduce out-of-plane Z
    step_x = (target_x - current_x) / (num_orbits - 1)
    step_z = (target_z - current_z) / (num_orbits - 1)

    # --- Build engine and run ---
    state_engine = StateParameter(
        initial_orbit=halo_seed,
        state=(SynodicState.X, SynodicState.Z),   # vary absolute coordinates, not amplitudes
        amplitude=False,
        target=(
            [current_x, current_z],
            [target_x, target_z],
        ),
        step=(step_x, step_z),
        corrector_kwargs=dict(max_attempts=50, tol=1e-12),
        max_orbits=num_orbits,
    )
    state_engine.run()

    # --- Build family and propagate ---
    halo_family = OrbitFamily.from_engine(state_engine)
    halo_family.propagate()
    halo_family.plot()

if __name__ == "__main__":
    main()
