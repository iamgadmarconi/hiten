"""Example script: computing the invariant torus for the Earth-Moon halo orbit.

Run with
    python examples/invariant_tori.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten import InvariantTori


def main() -> None:
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)

    orbit = l_point.create_orbit('halo', amplitude_z=0.3, zenith='southern')
    correction_opts = orbit.correction_options.merge(
        base=orbit.correction_options.base.merge(
            convergence=orbit.correction_options.base.convergence.merge(max_attempts=25)
        )
    )
    orbit.correct(options=correction_opts)
    orbit.propagate(steps=1000)

    torus = InvariantTori(orbit)
    torus.compute(epsilon=1e-2, n_theta1=512, n_theta2=512)
    torus.plot()

if __name__ == "__main__":
    main()