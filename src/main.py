import numpy as np

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.base import PoincareMap, PoincareMapConfig
from algorithms.center.utils import format_cm_table
from config import (C_OMEGA_HEURISTIC, DT, H0_LEVELS, INTEGRATOR_ORDER,
                    L_POINT, MAX_DEG, N_ITER, N_SEEDS, SYSTEM, USE_SYMPLECTIC)
from orbits.base import orbitConfig
from orbits.halo import HaloOrbit
from plots.plots import plot_poincare_map
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants
from utils.log_config import logger


def main() -> None:
    # ---------------- choose equilibrium point --------------------------
    Sun   = Body("Sun",   Constants.bodies["sun"  ]["mass"],   Constants.bodies["sun"  ]["radius"],   "yellow")
    Earth = Body("Earth", Constants.bodies["earth"]["mass"],   Constants.bodies["earth"]["radius"], "blue", Sun)
    Moon  = Body("Moon",  Constants.bodies["moon" ]["mass"],   Constants.bodies["moon" ]["radius"],  "gray",  Earth)

    d_EM = Constants.get_orbital_distance("earth", "moon")
    d_SE = Constants.get_orbital_distance("sun",   "earth")

    system_EM = System(systemConfig(Earth, Moon,  d_EM))
    system_SE = System(systemConfig(Sun,   Earth, d_SE))

    # Select system based on global configuration
    if SYSTEM == "EM":
        selected_system = system_EM
        system_name = "EM"
    elif SYSTEM == "SE":
        selected_system = system_SE
        system_name = "SE"
    else:
        raise ValueError(f"Unknown system: {SYSTEM}. Should be 'EM' or 'SE'")

    # Get the selected libration point
    selected_l_point = selected_system.get_libration_point(L_POINT)
    logger.info(f"Using {SYSTEM} system with L{L_POINT} point")

    # ---------------- centre-manifold (object) --------------------------
    cm = CenterManifold(selected_l_point, MAX_DEG)
    cm_H = cm.compute()  # triggers all internal caches

    logger.info(
        "\nCentre-manifold Hamiltonian (deg 2 to %d) in real NF variables (q2, p2, q3, p3)\n",
        MAX_DEG,
    )
    logger.info("\n\n%s\n\n", format_cm_table(cm_H, cm.clmo))

    logger.info("Starting Poincaré map generation process…")

    for H0 in H0_LEVELS:
        logger.info("Generating iterated Poincaré map for h0=%.3f", H0)
        pm_cfg = PoincareMapConfig(
            dt=DT,
            method="symplectic" if USE_SYMPLECTIC else "rk4",
            n_seeds=N_SEEDS,
            n_iter=N_ITER,
            integrator_order=INTEGRATOR_ORDER,
            c_omega_heuristic=C_OMEGA_HEURISTIC,
        )

        pm = PoincareMap(cm, energy=H0, config=pm_cfg)
        logger.info("Poincaré points:\n%s", pm.points)
        pm.plot()

    logger.info("Converting Poincaré points to initial conditions")
    ic = cm.cm2ic(pm.points[0], energy=H0_LEVELS[0])
    logger.info(f"Initial conditions:\n\n{ic}\n\n")

    orbit_config = orbitConfig(selected_system, "halo", L_POINT)
    orbit = HaloOrbit(orbit_config, ic)
    orbit.differential_correction()
    orbit.propagate(steps=1000, rtol=1e-12, atol=1e-12)
    orbit.plot()



if __name__ == "__main__":
    main()
