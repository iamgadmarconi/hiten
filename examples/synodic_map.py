"""Example script: Generating a Poincare map for the synodic section of a vertical orbit manifold.

Run with
    python examples/heteroclinic_connection.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.algorithms.poincare import SynodicMap, SynodicMapConfig
from hiten.system import System, VerticalOrbit


def main() -> None:
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)

    cm = l_point.get_center_manifold(degree=6)
    cm.compute()

    ic_seed = cm.ic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM

    orbit = VerticalOrbit(l_point, initial_state=ic_seed)
    orbit.correct(max_attempts=100, finite_difference=True)
    orbit.propagate(steps=1000)

    manifold = orbit.manifold(stable=True, direction="positive")
    manifold.compute(step=0.005)
    manifold.plot()

    section_cfg = SynodicMapConfig(
        section_axis="y",
        section_offset=0.0,
        plane_coords=("x", "z"),
        interp_kind="cubic",
        segment_refine=30,
        newton_max_iter=10,
    )
    synodic_map = SynodicMap(section_cfg)
    synodic_map.from_manifold(manifold)
    synodic_map.plot()


if __name__ == "__main__":
    main()