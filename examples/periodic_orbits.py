"""Example script: generation of several families of periodic orbits (Vertical Lyapunov,
Halo, planar Lyapunov) around an Earth-Moon libration point, together with their
stable manifolds.

Run with
    python examples/periodic_orbits.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import (CenterManifold, HaloOrbit, LyapunovOrbit,
                          Manifold, System, VerticalLyapunovOrbit)
from hiten.utils.files import _ensure_dir
from hiten.utils.log_config import logger

_ensure_dir("results")
# Directory that will hold manifold pickle files
_MANIFOLD_DIR = os.path.join("results", "manifolds")


def main() -> None:
    _ensure_dir(_MANIFOLD_DIR)

    # Build system & centre manifold
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(1)

    cm = CenterManifold(l_point, 10)
    cm.compute()

    ic_seed = cm.ic([0.0, 0.0], 0.6, "q3")
    logger.info("Initial conditions (CM to physical coordinates): %s", ic_seed)

    # Specifications for each family we wish to generate
    orbit_specs = [
        {
            "cls": VerticalLyapunovOrbit,
            "name": "Vertical Lyapunov",
            "kwargs": {"initial_state": ic_seed},  # Good initial guess from CM
            "diff_corr_attempts": 100,
        },
        {
            "cls": HaloOrbit,
            "name": "Halo",
            "kwargs": {"Az": 0.2, "Zenith": "southern"},
            "diff_corr_attempts": 25,
        },
        {
            "cls": LyapunovOrbit,
            "name": "Planar Lyapunov",
            "kwargs": {"Ax": 4e-3},
            "diff_corr_attempts": 25,
        },
    ]

    for spec in orbit_specs:
        logger.info("\n================  Generating %s orbit  ================", spec["name"])

        # Build orbit object with the new direct parameter API
        orbit = spec["cls"](l_point, **spec["kwargs"])

        # Differential correction, propagation & basic visualisation
        orbit.differential_correction(max_attempts=spec["diff_corr_attempts"])
        orbit.propagate(steps=1000)
        orbit.plot("rotating")

        # ---- Stable manifold generation --------------------------------------------------
        manifold = Manifold(orbit, stable=True, direction="Positive")
        m_filepath = os.path.join(_MANIFOLD_DIR, f"{spec['name'].lower().replace(' ', '_')}_manifold.pkl")

        logger.info("Computing manifold for %s orbit", spec["name"])
        manifold.compute()
        manifold.save(m_filepath)

        manifold.plot()


if __name__ == "__main__":
    main()