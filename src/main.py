from algorithms.center.manifold import center_manifold_rn
from algorithms.center.poincare.map import generate_iterated_poincare_map
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.utils import format_cm_table
from utils.log_config import logger
from plots.plots import plot_poincare_map
from config import (DT, H0_LEVELS, L_POINT, MAX_DEG, N_ITER, N_SEEDS, SYSTEM,
                        USE_SYMPLECTIC)
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants


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
    elif SYSTEM == "SE":
        selected_system = system_SE
    else:
        raise ValueError(f"Unknown system: {SYSTEM}. Should be 'EM' or 'SE'")
    
    # Get the selected libration point
    selected_l_point = selected_system.get_libration_point(L_POINT)
    logger.info(f"Using {SYSTEM} system with L{L_POINT} point")

    # ---------------- centre‑manifold reduction -------------------------
    H_cm_rn_full = center_manifold_rn(selected_l_point, psi, clmo, MAX_DEG)
    print("\n")
    print("Centre-manifold Hamiltonian (deg 2 to 5) in real NF variables (q2, p2, q3, p3)\n")
    print(format_cm_table(H_cm_rn_full, clmo))
    print("\n")

    logger.info("Starting Poincaré map generation process…")

    dt = DT

    all_pts = []
    output_directory = "results"

    for h0 in H0_LEVELS:
        logger.info("Generating iterated Poincaré map for h0=%.3f", h0)
        pts = generate_iterated_poincare_map(
            h0=h0,
            H_blocks=H_cm_rn_full,
            max_degree=MAX_DEG,
            psi_table=psi,
            clmo_table=clmo,
            encode_dict_list=encode_dict_list,
            n_seeds=N_SEEDS,
            n_iter=N_ITER,
            dt=dt,
            use_symplectic=USE_SYMPLECTIC,
            integrator_order=6,
            seed_axis="q2",
        )
        all_pts.append(pts)  # Store points for this energy level

    # Construct filename
    if len(H0_LEVELS) == 1:
        energy_level_str = f"{H0_LEVELS[0]:.2f}".replace('.', 'p')
    else:
        min_h0 = min(H0_LEVELS)
        max_h0 = max(H0_LEVELS)
        energy_level_str = f"{min_h0:.2f}to{max_h0:.2f}".replace('.', 'p')

    symplectic_str = "symplectic" if USE_SYMPLECTIC else "nonsymplectic"
    
    filename = f"PM_{MAX_DEG}_{energy_level_str}_{dt}_{symplectic_str}_{N_ITER}.svg"

    plot_poincare_map(all_pts, H0_LEVELS, output_dir=output_directory, filename=filename)


if __name__ == "__main__":
    main()
