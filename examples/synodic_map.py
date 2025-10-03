"""Example script: Generating a Poincare map for the synodic section of a vertical orbit manifold.

Run with
    python examples/heteroclinic_connection.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import SynodicMap, System, VerticalOrbit


def main() -> None:
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)

    cm = l_point.get_center_manifold(degree=6)
    cm.compute()

    ic_seed = cm.to_synodic([0.0, 0.0], 0.6, "q3") # Good initial guess from CM

    orbit = VerticalOrbit(l_point, initial_state=ic_seed)
    orbit.correct()
    orbit.propagate(steps=1000)

    manifold = orbit.manifold(stable=True, direction="positive")
    manifold.compute(step=0.005)
    manifold.plot()

    synodic_map = SynodicMap(manifold)
    synodic_map.compute(section_axis="y", section_offset=0.0, plane_coords=("x", "z"), direction=-1)
    synodic_map.plot()


if __name__ == "__main__":
    main()