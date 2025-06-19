import os

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.base import PoincareMap, poincareMapConfig
from algorithms.center.utils import format_cm_table
from config import (C_OMEGA_HEURISTIC, DT, H0_LEVELS, INTEGRATOR_ORDER,
                    L_POINT, MAX_DEG, N_ITER, N_SEEDS, PRIMARY, SECONDARY,
                    USE_GPU, USE_SYMPLECTIC)
from orbits.base import GenericOrbit, S, correctionConfig, orbitConfig
from orbits.halo import HaloOrbit
from orbits.lyapunov import LyapunovOrbit, VerticalLyapunovOrbit
from system.base import System, systemConfig
from system.body import Body
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
        "\nCentre-manifold Hamiltonian (deg 2 to %d) in real NF variables (q2, p2, q3, p3)\n",
        MAX_DEG,
    )
    logger.info("\n\n%s\n\n", format_cm_table(cm_H, cm.clmo))

    logger.info("Starting Poincaré map generation process…")

    for H0 in H0_LEVELS:
        logger.info("Generating iterated Poincaré map for h0=%.3f", H0)
        pm_cfg = poincareMapConfig(
            dt=DT,
            method="symplectic" if USE_SYMPLECTIC else "rk",
            n_seeds=N_SEEDS,
            n_iter=N_ITER,
            integrator_order=INTEGRATOR_ORDER,
            c_omega_heuristic=C_OMEGA_HEURISTIC,
            compute_on_init=False,
            use_gpu=USE_GPU
        )

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
    ic = cm.cm2ic([0.0, 0.0], energy=H0_LEVELS[0])
    logger.info(f"Initial conditions:\n\n{ic}\n\n")

    orbit_config = orbitConfig(system, "Vertical Lyapunov", L_POINT)
    orbit = VerticalLyapunovOrbit(orbit_config, ic)

    orbit.differential_correction(max_attempts=100)
    orbit.propagate(steps=1000, method="rk8")
    orbit.plot("rotating")
    orbit.plot("inertial")


if __name__ == "__main__":
    main()
