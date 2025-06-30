"""Example script: continuation-based generation of a Halo-orbit family.

Run with
    python examples/orbit_family.py
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import OrbitFamily, System
from hiten.algorithms import NaturalParameter
from hiten.system.orbits.base import S


def main() -> None:
    """Generate and save a small Halo family around the Earth-Moon L1 point.
    
    This example demonstrates how to use the NaturalParameter predictor to
    generate a family of Halo orbits around the Earth-Moon L1 point.
    """

    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)

    seed = l1.create_orbit('halo', amplitude_z= 0.2, zenith='southern')
    seed.differential_correction(max_attempts=25)

    # --- two-parameter continuation: vary absolute X (in-plane) and Z (out-of-plane) ---
    current_x = seed.initial_state[S.X]
    current_z = seed.initial_state[S.Z]  # 0 for planar Lyapunov seed

    # Target displacements (CRTBP canonical units)
    target_x = current_x + 0.2   # moderate shift along X
    target_z = current_z + 0.3   # introduce small out-of-plane Z

    num_orbits = 20

    step_x = (target_x - current_x) / (num_orbits - 1)
    step_z = (target_z - current_z) / (num_orbits - 1)

    engine = NaturalParameter(
        initial_orbit=seed,
        state=(S.X, S.Z),   # vary absolute coordinates, not amplitudes
        amplitude=False,
        target=(
            [current_x, current_z],
            [target_x, target_z],
        ),
        step=(step_x, step_z),
        corrector_kwargs=dict(max_attempts=50, tol=1e-12),
        max_orbits=num_orbits,
    )
    engine.run()

    family = OrbitFamily.from_engine(engine)
    family.propagate()
    family.plot()


if __name__ == "__main__":
    main()
