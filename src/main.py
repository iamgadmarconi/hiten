import numpy as np
import symengine as se
import sympy as sp
import cProfile
import pstats
import io

from system.libration import L1Point
from system.body import Body
from system.base import System, systemConfig
from utils.constants import Constants
from algorithms.center.polynomial.base import init_index_tables
from algorithms.center.manifold import (
    reduce_center_manifold_arrays,
    coefficients_to_table_arrays,
)

from log_config import logger


def main():

    max_degree = 5
    
    psi, clmo = init_index_tables(max_degree)

    Sun = Body("Sun", 
                Constants.bodies["sun"]["mass"], 
                Constants.bodies["sun"]["radius"], 
                "yellow")

    Earth = Body("Earth", 
                Constants.bodies["earth"]["mass"], 
                Constants.bodies["earth"]["radius"], 
                "blue",
                Sun)

    Moon = Body("Moon", 
                Constants.bodies["moon"]["mass"], 
                Constants.bodies["moon"]["radius"], 
                "gray", 
                Earth)

    distance_EM = Constants.get_orbital_distance("earth", "moon")
    distance_SE = Constants.get_orbital_distance("sun", "earth")

    system_EM = System(systemConfig(Earth, Moon, distance_EM))
    system_SE = System(systemConfig(Sun, Earth, distance_SE))

    Lpoint_EM = system_EM.get_libration_point(2)
    Lpoint_SE = system_SE.get_libration_point(1)

    H_cnr = reduce_center_manifold_arrays(Lpoint_EM, max_degree,
                                        psi=psi, clmo=clmo)
    print(coefficients_to_table_arrays(H_cnr, psi, clmo))


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
