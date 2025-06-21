import os

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.base import PoincareMap, poincareMapConfig
from algorithms.center.utils import format_cm_table
from config import (C_OMEGA_HEURISTIC, DT, H0, INTEGRATOR_METHOD,
                    INTEGRATOR_ORDER, L_POINT, MAX_DEG, N_ITER, N_SEEDS,
                    PRIMARY, SECONDARY, USE_GPU)
from orbits.base import orbitConfig
from orbits.halo import HaloOrbit
from orbits.lyapunov import LyapunovOrbit, VerticalLyapunovOrbit
from system.base import System, systemConfig
from system.body import Body
from system.manifold import Manifold, manifoldConfig
from utils.constants import Constants
from utils.log_config import logger


def main() -> None:
    primary = Body(PRIMARY, Constants.bodies[PRIMARY.lower()]["mass"], Constants.bodies[PRIMARY.lower()]["radius"], "blue")
    secondary = Body(SECONDARY, Constants.bodies[SECONDARY.lower()]["mass"], Constants.bodies[SECONDARY.lower()]["radius"], "gray", primary)
    system = System(systemConfig(primary, secondary, Constants.get_orbital_distance(PRIMARY, SECONDARY)))
    l_point = system.get_libration_point(L_POINT)
    logger.info(f"Using {PRIMARY}-{SECONDARY} system with L{L_POINT} point")

    cm = CenterManifold(l_point, MAX_DEG)
    cm_H = cm.compute()

    logger.info(
        "\nCentre-manifold Hamiltonian (deg 2 to 5) in real NF variables (q2, p2, q3, p3)\n",
        MAX_DEG,
    )
    logger.info("\n\n%s\n\n", format_cm_table(cm_H, cm.clmo))

    logger.info("Starting Poincaré map generation process…")

    logger.info("Generating iterated Poincaré map for h0=%.3f", H0)
    pm_cfg = poincareMapConfig(dt=DT, method=INTEGRATOR_METHOD, n_seeds=N_SEEDS, n_iter=N_ITER, seed_axis="q2", section_coord="q3",
                                integrator_order=INTEGRATOR_ORDER, c_omega_heuristic=C_OMEGA_HEURISTIC, compute_on_init=False, use_gpu=USE_GPU)

    filepath = f"results/maps/poincare_map_{H0}.pkl"

    pm = PoincareMap(cm, energy=H0, config=pm_cfg)

    if os.path.exists(filepath):
        logger.info(f"Loading existing Poincaré map from {filepath}")
        pm.load(filepath)
    else:
        logger.info("Computing new Poincaré map")
        pm.compute()
        pm.save(filepath)

    pm.plot_interactive(system)

    logger.info("Converting Poincaré points to initial conditions")
    ic = cm.cm2ic([0.0, 0.0], energy=H0)
    logger.info(f"Initial conditions (CM→IC):\n\n{ic}\n\n")

    orbit_specs = [
        {
            "cls": VerticalLyapunovOrbit,
            "name": "Vertical Lyapunov",
            "extra_params": {},
            "initial_state": ic,               # Use CM seed
            "diff_corr_attempts": 100,
            "manifold_file": "results/manifolds/vertical_orbit_manifold.pkl",
        },
        {
            "cls": HaloOrbit,
            "name": "Halo",
            "extra_params": {"Az": 0.2, "Zenith": "southern"},
            "initial_state": None,
            "diff_corr_attempts": 25,
            "manifold_file": "results/manifolds/halo_orbit_manifold.pkl",
        },
        {
            "cls": LyapunovOrbit,
            "name": "Lyapunov",
            "extra_params": {"Ax": 4e-3},
            "initial_state": None,
            "diff_corr_attempts": 25,
            "manifold_file": "results/manifolds/lyapunov_orbit_manifold.pkl",
        },
    ]

    for spec in orbit_specs:
        logger.info("\n================  Generating %s orbit  ================", spec["name"])

        # Build orbit
        cfg = orbitConfig(spec["name"], l_point, extra_params=spec["extra_params"])
        orbit = spec["cls"](cfg, spec["initial_state"])

        # Differential correction & propagation
        orbit.differential_correction(max_attempts=spec["diff_corr_attempts"])
        orbit.propagate(steps=1000)
        orbit.plot("rotating")
        orbit.animate()

        # Build manifold configuration
        manifold_cfg = manifoldConfig(orbit, stable=True, direction="Positive")
        manifold = Manifold(manifold_cfg)
        m_filepath = spec["manifold_file"]

        # Load if available, else compute & save
        if os.path.exists(m_filepath):
            logger.info("Loading existing manifold from %s", m_filepath)
            manifold.load(m_filepath)
        else:
            logger.info("Computing manifold for %s orbit", spec["name"])
            manifold.compute()
            # manifold.save(m_filepath)

        manifold.plot()

if __name__ == "__main__":
    main()
