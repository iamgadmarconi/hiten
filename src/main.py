import numpy as np

from algorithms.center.manifold import center_manifold_rn
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)
from algorithms.center.utils import format_cm_table
from log_config import logger
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants

MAX_DEG = 8
TOL     = 1e-14


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
    L1_EM        = system_EM.get_libration_point(1)   # Earth‑Moon L₁
    L2_EM        = system_EM.get_libration_point(2)   # Earth‑Moon L₂
    L1_SE        = system_SE.get_libration_point(1)   # Sun‑Earth L₁
    L2_SE        = system_SE.get_libration_point(2)   # Sun‑Earth L₂

    # ---------------- centre‑manifold reduction -------------------------
    H_cm_rn_full = center_manifold_rn(L1_SE, psi, clmo, MAX_DEG)
    print("\n")
    print("Centre-manifold Hamiltonian (deg 2 to 5) in real NF variables (q2, p2, q3, p3)\n")
    print(format_cm_table(H_cm_rn_full, clmo))
    print("\n")
    # ---------------- Generate Poincare Map Points ------------------
    logger.info("Starting Poincaré map generation process...")

    H0_LEVELS = [0.20, 0.40, 0.60, 1.00]  # As in Fig. 3 for Sun–Earth L1


if __name__ == "__main__":
    main()
