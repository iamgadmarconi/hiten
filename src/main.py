import numpy as np

from algorithms.center.base import CenterManifold
from algorithms.center.poincare.base import PoincareMap, PoincareMapConfig
from algorithms.center.utils import format_cm_table
from algorithms.integrators.standard import propagate_crtbp
from config import (
    C_OMEGA_HEURISTIC,
    DT,
    H0_LEVELS,
    INTEGRATOR_ORDER,
    L_POINT,
    MAX_DEG,
    N_ITER,
    N_SEEDS,
    SYSTEM,
    USE_SYMPLECTIC,
)
from plots.plots import (plot_orbit_inertial_frame, plot_orbit_rotating_frame,
                         plot_poincare_map)
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

    all_pts = []
    output_directory = "results"

    for H0 in H0_LEVELS:
        logger.info("Generating iterated Poincaré map for h0=%.3f", H0)
        pm_cfg = PoincareMapConfig(
            dt=DT,
            method="symplectic" if USE_SYMPLECTIC else "rk4",
            use_iterated=True,
            n_seeds=N_SEEDS,
            n_iter=N_ITER,
            integrator_order=INTEGRATOR_ORDER,
            c_omega_heuristic=C_OMEGA_HEURISTIC,
        )

        pm = PoincareMap(cm, energy=H0, config=pm_cfg)
        logger.info("Poincaré points:\n%s", pm.points)
        all_pts.append(pm.points)

    logger.info("Converting Poincaré points to initial conditions")
    first_pm_point = all_pts[0][0]
    ic = cm.cm2ic(first_pm_point, energy=H0_LEVELS[0])
    logger.info(f"Initial conditions:\n\n{ic}\n\n")

    logger.info("Propagating initial conditions")
    sol = propagate_crtbp(ic, 0, 1.4*np.pi, selected_system.mu)
    traj = sol.y.T
    times = sol.t

    # Plot the orbit
    plot_orbit_rotating_frame(traj, selected_system.mu, selected_system, selected_l_point, "PM", show=True)
    plot_orbit_inertial_frame(traj, times, selected_system.mu, selected_system, "PM", show=True)

    # Construct filename
    if len(H0_LEVELS) == 1:
        energy_level_str = f"{H0_LEVELS[0]:.2f}".replace('.', 'p')
    else:
        energy_level_str = f"{min(H0_LEVELS):.2f}to{max(H0_LEVELS):.2f}".replace('.', 'p')

    symplectic_str = "symplectic" if USE_SYMPLECTIC else "nonsymplectic"

    filename = (
        f"{system_name}_{L_POINT}_PM_{MAX_DEG}_{energy_level_str}_{DT}_"
        f"{symplectic_str}_{N_ITER}.svg"
    )

    plot_poincare_map(all_pts, H0_LEVELS, output_dir=output_directory, filename=filename)


if __name__ == "__main__":
    main()
