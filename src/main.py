import matplotlib.pyplot as plt
import numpy as np

from algorithms.center.manifold import center_manifold_rn
from algorithms.center.poincare.generation.lindstedt_poincare import (build_LP,
                                                                      eval_lp)
from algorithms.center.poincare.map import generate_iterated_poincare_map
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.utils import format_cm_table
from log_config import logger
from plots.plots import plot_poincare_map
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants


# System configuration
SYSTEM = "SE"  # "EM" for Earth-Moon or "SE" for Sun-Earth
L_POINT = 1    # Libration point number (1 or 2)

# Algorithm parameters
MAX_DEG = 8
TOL     = 1e-14
# -------- LP parameters ----------------------------------------------------
LP_MAX_ORDER = 15          # i+j ≤ 15  (good compromise between speed & accuracy)
ALPHA        = 0.03        # in-plane amplitude
BETA         = 0.00        # out-of-plane amplitude = 0 → Lyapunov family


def build_three_body_system():
    """Return (System EM, System SE)."""
    Sun   = Body("Sun",   Constants.bodies["sun"  ]["mass"],   Constants.bodies["sun"  ]["radius"],   "yellow")
    Earth = Body("Earth", Constants.bodies["earth"]["mass"],   Constants.bodies["earth"]["radius"], "blue", Sun)
    Moon  = Body("Moon",  Constants.bodies["moon" ]["mass"],   Constants.bodies["moon" ]["radius"],  "gray",  Earth)

    d_EM = Constants.get_orbital_distance("earth", "moon")
    d_SE = Constants.get_orbital_distance("sun",   "earth")

    system_EM = System(systemConfig(Earth, Moon,  d_EM))
    system_SE = System(systemConfig(Sun,   Earth, d_SE))
    return system_EM, system_SE


def main() -> None:
    # ---------------- lookup tables for polynomial indexing --------------
    psi, clmo = init_index_tables(MAX_DEG)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    # ---------------- choose equilibrium point --------------------------
    system_EM, system_SE = build_three_body_system()
    
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

    logger.info("Computing Lindstedt-Poincaré expansion (order ≤ %d)…",
                LP_MAX_ORDER)

    c_series = [selected_l_point._cn(2)] + [
        selected_l_point._cn(n) for n in range(3, LP_MAX_ORDER + 3 + 1)
    ]

    X_arr, Y_arr, Z_arr, Omega_w, Omega_n = build_LP(c_series, LP_MAX_ORDER)
    logger.info("ω₀ = %.15g,  ν₀ = %.15g", Omega_w[0, 0], Omega_n[0, 0])

    x0, y0, z0 = eval_lp(ALPHA, BETA, X_arr, Y_arr, Z_arr, LP_MAX_ORDER)
    logger.info("LP initial position (alpha=%.3g, beta=%.3g)  =>  (x,y,z) = "
                "(%.6e, %.6e, %.6e)", ALPHA, BETA, x0, y0, z0)

    logger.info("Starting Poincaré map generation process…")

    H0_LEVELS = [0.20, 0.40, 0.60, 1.00]

    dt = 1e-1
    USE_SYMPLECTIC = False
    N_SEEDS = 10 # seeds along q2-axis
    N_ITER = 500 # iterations per seed

    all_pts = []

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

    plot_poincare_map(all_pts, H0_LEVELS)


if __name__ == "__main__":
    main()
