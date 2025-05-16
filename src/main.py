import cProfile
import io
import pstats


from algorithms.center.manifold import center_manifold_cn, center_manifold_rn
from algorithms.center.polynomial.base import init_index_tables
from log_config import logger
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants
from algorithms.center.utils import format_cm_table

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

    # ---------------- choose equilibrium point --------------------------
    system_EM, system_SE = build_three_body_system()
    L1_EM        = system_EM.get_libration_point(1)   # Earth‑Moon L₁
    L2_EM        = system_EM.get_libration_point(2)   # Earth‑Moon L₂
    L1_SE        = system_SE.get_libration_point(1)   # Sun‑Earth L₁
    L2_SE        = system_SE.get_libration_point(2)   # Sun‑Earth L₂

    # ---------------- centre‑manifold reduction -------------------------
    # H_cm_cn_full is the full list of coefficient arrays, indexed by degree
    H_cm_cn_full = center_manifold_cn(L1_SE, psi, clmo, MAX_DEG)
    H_cm_rn_full = center_manifold_rn(L1_SE, psi, clmo, MAX_DEG)

    # ---------------- pretty print (Table 1 style) ----------------------
    print("Centre-manifold Hamiltonian (deg 2 to 5) in complex NF variables\n")
    print(format_cm_table(H_cm_cn_full, clmo))
    print("\n")
    print("Centre-manifold Hamiltonian (deg 2 to 5) in real NF variables\n")
    print(format_cm_table(H_cm_rn_full, clmo))

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
