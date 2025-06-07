import numpy as np
from numba import cuda

from algorithms.center.coordinates import poincare2ic
from algorithms.center.manifold import center_manifold_real
from algorithms.center.poincare.cuda.map import \
    generate_iterated_poincare_map_gpu
from algorithms.center.poincare.map import generate_iterated_poincare_map
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.utils import format_cm_table
from algorithms.integrators.standard import propagate_crtbp
from config import (C_OMEGA_HEURISTIC, DT, H0_LEVELS, INTEGRATOR_ORDER,
                    L_POINT, MAX_DEG, N_ITER, N_SEEDS, SYSTEM, USE_GPU,
                    USE_SYMPLECTIC)
from plots.plots import plot_orbit_rotating_frame, plot_poincare_map
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants
from utils.log_config import logger

if cuda.is_available() and USE_GPU:
    generate_poincare_map = generate_iterated_poincare_map_gpu
else:
    generate_poincare_map = generate_iterated_poincare_map


def main() -> None:
    # ---------------- lookup tables for polynomial indexing --------------
    psi, clmo = init_index_tables(MAX_DEG)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

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

    # ---------------- centre‑manifold reduction -------------------------
    H_cm_real_full = center_manifold_real(selected_l_point, psi, clmo, MAX_DEG)
    logger.info("\nCentre-manifold Hamiltonian (deg 2 to 5) in real NF variables (q2, p2, q3, p3)\n")
    logger.info(f"\n\n{format_cm_table(H_cm_real_full, clmo)}\n\n")

    logger.info("Starting Poincaré map generation process…")

    all_pts = []
    output_directory = "results"

    for H0 in H0_LEVELS:
        logger.info("Generating iterated Poincaré map for h0=%.3f", H0)
        pts = generate_poincare_map(
            h0=H0,
            H_blocks=H_cm_real_full,
            max_degree=MAX_DEG,
            psi_table=psi,
            clmo_table=clmo,
            encode_dict_list=encode_dict_list,
            n_seeds=N_SEEDS,
            n_iter=N_ITER,
            dt=DT,
            use_symplectic=USE_SYMPLECTIC,
            integrator_order=INTEGRATOR_ORDER,
            c_omega_heuristic=C_OMEGA_HEURISTIC,
            seed_axis="q2",
        )
        # Convert Poincaré points to initial conditions
        logger.info(f"Poincaré points:\n{pts}")
        all_pts.append(pts)  # Store points for this energy level

    logger.info("Converting Poincaré points to initial conditions")
    ic = poincare2ic([pts[0]], selected_l_point, psi, clmo, MAX_DEG, H0_LEVELS[0])
    logger.info(f"Initial conditions:\n\n{ic}\n\n")

    logger.info("Propagating initial conditions")
    traj = propagate_crtbp(ic[0], 0, 2*np.pi, selected_system.mu).y.T

    # Plot the orbit
    plot_orbit_rotating_frame(traj, selected_system.mu, selected_system, selected_l_point, "PM", show=True)

    # Construct filename
    if len(H0_LEVELS) == 1:
        energy_level_str = f"{H0_LEVELS[0]:.2f}".replace('.', 'p')
    else:
        energy_level_str = f"{min(H0_LEVELS):.2f}to{max(H0_LEVELS):.2f}".replace('.', 'p')

    symplectic_str = "symplectic" if USE_SYMPLECTIC else "nonsymplectic"
    
    filename = f"{system_name}_{L_POINT}_PM_{MAX_DEG}_{energy_level_str}_{DT}_{symplectic_str}_{N_ITER}.svg"

    plot_poincare_map(all_pts, H0_LEVELS, output_dir=output_directory, filename=filename)


if __name__ == "__main__":
    main()
