"""Example script: computing the invariant torus for the Earth-Moon halo orbit.

Run with
    python examples/invariant_tori.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten.algorithms import InvariantTori


def main() -> None:
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)

    orbit = l_point.create_orbit('halo', amplitude_z=0.3, zenith='southern')
    orbit.correct(max_attempts=25)
    orbit.propagate(steps=1000)

    torus = InvariantTori(orbit)
    torus.compute(scheme='gmos', epsilon=1e-3, n_theta1=32, n_theta2=128)
    torus.plot()

if __name__ == "__main__":
    main()