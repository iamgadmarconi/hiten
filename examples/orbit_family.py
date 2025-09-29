"""Example script: continuation-based generation of a Halo-orbit halo_family.

Run with
    python examples/orbit_family.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten.algorithms import ContinuationPipeline
from hiten.algorithms.types.states import SynodicState
from hiten.algorithms.continuation.config import _OrbitContinuationConfig
from hiten.system.family import OrbitFamily
from hiten.utils.log_config import logger


def main() -> None:
    """Generate and save a small Halo halo_family around the Earth-Moon L1 point.
    
    This example demonstrates how to use the ContinuationPipeline predictor to
    generate a halo_family of Halo orbits around the Earth-Moon L1 point.
    """
    num_orbits = 50
    system = System.from_bodies("earth", "moon")
    l1 = system.get_libration_point(1)
    
    halo_seed = l1.create_orbit('halo', amplitude_z= 0.2, zenith='southern')
    halo_seed.correct(max_attempts=25, max_delta=1e-3)

    current_z = halo_seed.initial_state[SynodicState.Z]  # 0 for planar Lyapunov halo_seed
    target_z = current_z + 5.0   # introduce out-of-plane Z
    step_z = (target_z - current_z) / (num_orbits - 1)

    config= _OrbitContinuationConfig(
        target=([current_z], [target_z]),
        step=((step_z),),
        state=(SynodicState.Z,),
        max_members=50,
        extra_params=dict(max_attempts=50, tol=1e-12),
        stepper="secant",
    )

    state_parameter = ContinuationPipeline.with_default_engine(config=config)

    result = state_parameter.generate(halo_seed)

    logger.info(f"Generated {len(result.family)} orbits (success rate {result.success_rate:.2%})")

    family = OrbitFamily.from_result(result)
    family.propagate()
    family.plot()

if __name__ == "__main__":
    main()
