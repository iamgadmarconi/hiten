"""Example script: continuation-based generation of a Halo-orbit halo_family.

Run with
    python examples/orbit_family.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten.algorithms.continuation.options import OrbitContinuationOptions
from hiten.algorithms.corrector.options import OrbitCorrectionOptions
from hiten.algorithms.types.options import (ConvergenceOptions,
                                            CorrectionOptions,
                                            IntegrationOptions,
                                            NumericalOptions)
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

    corrector_options = OrbitCorrectionOptions(
            base=CorrectionOptions(
                convergence=ConvergenceOptions(
                    max_attempts=50,
                    tol=1e-12,
                    max_delta=1e-2,
                ),
                integration=IntegrationOptions(
                    dt=1e-2,
                    order=8,
                    max_steps=2000,
                    c_omega_heuristic=20.0,
                    steps=500,
                ),
                numerical=NumericalOptions(
                    fd_step=1e-8,
                    line_search_alpha_reduction=0.5,
                    line_search_min_alpha=1e-4,
                    line_search_armijo_c=0.1,
                ),
            ),
            forward=1,
        )
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
            extra_params=corrector_options,
        )

    result = halo_seed.generate(options)

    family = OrbitFamily.from_result(result)
    family.propagate()
    family.plot()

if __name__ == "__main__":
    main()
