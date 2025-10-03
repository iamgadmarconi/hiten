"""Example script: continuation-based generation of a Halo-orbit halo_family.

Run with
    python examples/orbit_family.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten.algorithms.continuation.options import OrbitContinuationOptions
from hiten.algorithms.types.states import SynodicState
from hiten.system.family import OrbitFamily


def main() -> None:
    """Generate and save a small Halo halo_family around the Earth-Moon L1 point.
    
    This example demonstrates how to use the ContinuationPipeline predictor to
    generate a halo_family of Halo orbits around the Earth-Moon L1 point.
    """
    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)
    
    halo_seed = l1.create_orbit('halo', amplitude_z= 0.2, zenith='southern')
    halo_seed.correct()
    halo_seed.propagate()

    options = OrbitContinuationOptions(
            target=(
                [halo_seed.initial_state[SynodicState.Z], halo_seed.initial_state[SynodicState.Y]],
                [halo_seed.initial_state[SynodicState.Z] + 2.0, halo_seed.initial_state[SynodicState.Y]-1.0]),
            step=(
                (1 - halo_seed.initial_state[SynodicState.Z]) / (100 - 1),
                (1 - halo_seed.initial_state[SynodicState.Y]) / (100 - 1),
            ),
            max_members=100,
            max_retries_per_step=50,
            step_min=1e-10,
            step_max=1.0,
            shrink_policy=None,
            extra_params=halo_seed.correction_options,
        )

    result = halo_seed.generate(options)

    print(result)

    family = OrbitFamily.from_result(result)
    family.propagate()
    family.plot()

if __name__ == "__main__":
    main()
