import numpy as np
import symengine as se
import sympy as sp

from system.libration import L1Point
from system.body import Body
from system.base import System, systemConfig
from utils.constants import Constants
from algorithms.center.lie import (
    extract_coeffs_up_to_degree,
    compute_center_manifold,
    reduce_center_manifold,
    coefficients_to_table
)

from log_config import logger

def main():

    degree = 6

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

    # Compute the center manifold
    H_nf, G_total = compute_center_manifold(Lpoint_SE, degree)
    H_nfr = reduce_center_manifold(H_nf, degree)

    # Get formatted table of coefficients
    coeffs = extract_coeffs_up_to_degree(H_nfr, degree)
    logger.info(f"Found {len(coeffs)} coefficients")
    table = coefficients_to_table(coeffs, save=False)
    logger.info(table)


if __name__ == "__main__":
    main()
