import cProfile
import io
import pstats

import numpy as np

from algorithms.center.manifold import center_manifold_rn
from algorithms.center.poincare.flow import generate_hamiltonian_flow
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
    # H_cm_cn_full is the full list of coefficient arrays, indexed by degree
    # H_cm_cn_full = center_manifold_cn(L1_SE, psi, clmo, MAX_DEG)
    H_cm_rn_full = center_manifold_rn(L1_SE, psi, clmo, MAX_DEG)
    # ---------------- pretty print (Table 1 style) ----------------------
    # print("Centre-manifold Hamiltonian (deg 2 to 5) in complex NF variables\n")
    # print(format_cm_table(H_cm_cn_full, clmo))
    print("\n")
    print("Centre-manifold Hamiltonian (deg 2 to 5) in real NF variables (q2, p2, q3, p3)\n")
    print(format_cm_table(H_cm_rn_full, clmo))
    print("\n")

    # ---------------- Integrate Hamiltonian Flow ----------------------
    # Define initial conditions for the center manifold dynamics
    # Example: Q_cm1, Q_cm2, P_cm1, P_cm2
    # These values would typically be chosen based on a specific energy level
    # or a starting point for a Poincare section.
    initial_cm_state_4d = np.array([0.1, 0.0, 0.05, 0.0], dtype=np.float64)

    # Define time points for integration
    t_start = 0.0
    t_end = 10.0
    num_points = 1001
    t_values = np.linspace(t_start, t_end, num_points)

    # Integrator order (e.g., 4th order)
    integrator_order = 4 # Must be even and positive

    logger.info("Starting Hamiltonian flow integration...")
    trajectory = generate_hamiltonian_flow(
        hamiltonian_poly_coeffs=H_cm_rn_full,
        max_deg_hamiltonian=MAX_DEG,
        psi_table=psi,
        clmo_table=clmo,
        encode_dict_list=encode_dict_list,
        initial_cm_state_4d=initial_cm_state_4d,
        t_values=t_values,
        integrator_order=integrator_order
    )
    logger.info(f"Hamiltonian flow integration complete. Trajectory shape: {trajectory.shape}")
    # For example, print the first few points of the trajectory
    logger.info("First 5 points of the trajectory:\n" + str(trajectory[:5]))
    
    # Further processing of the trajectory (e.g., for Poincare map) would go here.

if __name__ == "__main__":
    # Use cProfile to profile the main function execution
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    
    # Print the profiling results sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Print top 20 functions by cumulative time
    logger.info("Profiling Statistics:\n" + s.getvalue())
