"""Example script: generation of several families of periodic orbits (Vertical,
Halo, planar Lyapunov) around an Earth-Moon libration point, together with their
stable manifolds.

Run with
    python examples/orbit_manifold.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import System


def main() -> None:
    # Build system & centre manifold
    system = System.from_bodies("sun", "earth")
    l_point = system.get_libration_point(2)
    halo_orbit = l_point.create_orbit('halo', amplitude_z=0.3, zenith='northern')
    halo_orbit.correct()
    halo_orbit.propagate()

    direction, stability = ['positive', 'negative'], [True, False]
    for d in direction:
        for s in stability:
            manifold = halo_orbit.manifold(stable=s, direction=d)
            manifold.compute(integration_fraction=1, dt=1e-3)
            manifold.plot()

if __name__ == "__main__":
    main()
