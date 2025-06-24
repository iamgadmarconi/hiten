"""Example script: generation of several families of periodic orbits (Vertical Lyapunov,
Halo, planar Lyapunov) around an Earth-Moon libration point, together with their
stable manifolds.

Run with
    python examples/periodic_orbits.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import (HaloOrbit, LyapunovOrbit, System,
                          VerticalLyapunovOrbit)
from hiten.utils.log_config import logger


def main() -> None:
    # Build system & centre manifold
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)
    cm = l_point.get_center_manifold(max_degree=10)
    cm.compute()

    ic_seed = cm.ic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM
    logger.info("Initial conditions (CM to physical coordinates): %s", ic_seed)

    # Specifications for each family we wish to generate
    orbit_specs = [
        {
            "cls": VerticalLyapunovOrbit,
            "name": "Vertical Lyapunov",
            "kwargs": {"initial_state": ic_seed},
            "diff_corr_attempts": 100,
        },
        {
            "cls": HaloOrbit,
            "name": "Halo",
            "kwargs": {"amplitude_z": 0.2, "zenith": "southern"},
            "diff_corr_attempts": 25,
        },
        {
            "cls": LyapunovOrbit,
            "name": "Planar Lyapunov",
            "kwargs": {"amplitude_x": 4e-3},
            "diff_corr_attempts": 25,
        },
    ]

    for spec in orbit_specs:

        orbit = l_point.create_orbit(spec["cls"], **spec["kwargs"])

        # Differential correction, propagation & basic visualisation
        orbit.differential_correction(max_attempts=spec["diff_corr_attempts"])
        orbit.propagate(steps=1000)
        orbit.plot("rotating")

        manifold = orbit.manifold(stable=True, direction="positive")
        manifold.compute()
        manifold.plot()


if __name__ == "__main__":
    main()