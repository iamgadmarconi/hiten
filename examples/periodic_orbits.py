"""Example script: generation of several families of periodic orbits (Vertical,
Halo, planar Lyapunov) around an Earth-Moon libration point.

Run with
    python examples/periodic_orbits.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import (HaloOrbit, LyapunovOrbit, System,
                          VerticalOrbit)
from hiten.utils.log_config import logger


def main() -> None:
    # Build system & centre manifold
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)
    cm = l_point.get_center_manifold(degree=6)
    cm.compute()

    ic_seed = cm.to_synodic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM
    logger.info("Initial conditions (CM to physical coordinates): %s", ic_seed)

    # Specifications for each family we wish to generate
    orbit_specs = [
        {
            "cls": VerticalOrbit,
            "name": "Vertical",
            "kwargs": {"initial_state": ic_seed},
        },
        {
            "cls": HaloOrbit,
            "name": "Halo",
            "kwargs": {"amplitude_z": 0.2, "zenith": "southern"},
        },
        {
            "cls": LyapunovOrbit,
            "name": "Planar Lyapunov",
            "kwargs": {"amplitude_x": 0.1},
        },
    ]

    for spec in orbit_specs:

        orbit = l_point.create_orbit(spec["cls"], **spec["kwargs"])

        orbit.correct()
        print(f"Corrected state: {orbit.initial_state} | Period: {orbit.period}")
        orbit.propagate(steps=1000)
        orbit.plot("rotating")

if __name__ == "__main__":
    main()