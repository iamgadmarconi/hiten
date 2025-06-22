"""Example script: generation of several families of periodic orbits (Vertical Lyapunov,
Halo, planar Lyapunov) around an Earth-Moon libration point, together with their
stable manifolds.

Run with
    python examples/periodic_orbits.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import (Body, CenterManifold, HaloOrbit, LyapunovOrbit, Manifold,
                    System, VerticalLyapunovOrbit, manifoldConfig, orbitConfig,
                    systemConfig)
from hiten.utils import Constants
from hiten.utils.files import _ensure_dir
from hiten.utils.log_config import logger

_ensure_dir("results")
# Directory that will hold manifold pickle files
_MANIFOLD_DIR = os.path.join("results", "manifolds")


def build_system():
    """Utility to construct the Earth-Moon CRTBP system and its L point."""
    primary = Body(
        "Earth",
        Constants.bodies["earth"]["mass"],
        Constants.bodies["earth"]["radius"],
        "blue",
    )
    secondary = Body(
        "Moon",
        Constants.bodies["moon"]["mass"],
        Constants.bodies["moon"]["radius"],
        "gray",
        primary,
    )
    system = System(
        systemConfig(
            primary,
            secondary,
            Constants.get_orbital_distance("Earth", "Moon"),
        )
    )
    return system, system.get_libration_point(1)


def compute_center_manifold(l_point):
    cm = CenterManifold(l_point, 10)
    cm.compute()
    return cm


def initial_conditions_from_cm(cm):
    """Return a small set of initial conditions obtained from the CM Poincaré map."""
    pm = cm.poincare_map(
        0.6,
        seed_axis="q2",
        section_coord="q3",
        n_seeds=20,
        n_iter=25,
        use_gpu=False,
    )
    return pm.ic([0.0, 0.0])


def main() -> None:
    _ensure_dir(_MANIFOLD_DIR)

    # Build system & centre manifold
    _, l_point = build_system()
    logger.info("Preparing centre manifold for initial guesses…")
    cm = compute_center_manifold(l_point)
    ic_seed = initial_conditions_from_cm(cm)
    logger.info("Initial conditions (CM to physical coordinates): %s", ic_seed)

    # Specifications for each family we wish to generate
    orbit_specs = [
        {
            "cls": VerticalLyapunovOrbit,
            "name": "Vertical Lyapunov",
            "extra_params": {},
            "initial_state": ic_seed,  # Good initial guess from CM
            "diff_corr_attempts": 100,
        },
        {
            "cls": HaloOrbit,
            "name": "Halo",
            "extra_params": {"Az": 0.2, "Zenith": "southern"},
            "initial_state": None,
            "diff_corr_attempts": 25,
        },
        {
            "cls": LyapunovOrbit,
            "name": "Planar Lyapunov",
            "extra_params": {"Ax": 4e-3},
            "initial_state": None,
            "diff_corr_attempts": 25,
        },
    ]

    for spec in orbit_specs:
        logger.info("\n================  Generating %s orbit  ================", spec["name"])

        # Build orbit object
        cfg = orbitConfig(spec["name"], l_point, extra_params=spec["extra_params"])
        orbit = spec["cls"](cfg, spec["initial_state"])

        # Differential correction, propagation & basic visualisation
        orbit.differential_correction(max_attempts=spec["diff_corr_attempts"])
        orbit.propagate(steps=1000)
        orbit.plot("rotating")

        # Optionally animate (comment out if running headless)
        try:
            orbit.animate()
        except Exception as exc:
            logger.warning("Could not create animation: %s", exc)

        # ---- Stable manifold generation --------------------------------------------------
        m_cfg = manifoldConfig(orbit, stable=True, direction="Positive")
        manifold = Manifold(m_cfg)
        m_filepath = os.path.join(_MANIFOLD_DIR, f"{spec['name'].lower().replace(' ', '_')}_manifold.pkl")

        if os.path.exists(m_filepath):
            logger.info("Loading existing manifold from %s", m_filepath)
            manifold.load(m_filepath)
        else:
            logger.info("Computing manifold for %s orbit", spec["name"])
            manifold.compute()
            manifold.save(m_filepath)

        manifold.plot()


if __name__ == "__main__":
    main()