import numpy as np
import symengine as se
import sympy as sp

from system.libration import L1Point
from system.body import Body
from system.base import System, systemConfig
from utils.constants import Constants
from algorithms.variables import (
    linear_modes_vars, 
    scale_factors_vars, 
    get_vars, 
    create_symbolic_cn
)
from algorithms.center.lie import (
    extract_coeffs_up_to_degree,
    compute_center_manifold,
    real_normal_center_manifold,
    coefficients_to_table,
)

from log_config import logger


omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)
s1, s2 = get_vars(scale_factors_vars)


def main():

    degree = 4
    
    # Generate all necessary c symbols based on degree
    c_symbols = {c2: c2}  # Start with c2 which is already defined
    for n in range(3, degree+1):
        c_symbols[create_symbolic_cn(n)] = create_symbolic_cn(n)

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

    lambda1_num, omega1_num, omega2_num = Lpoint_SE.linear_modes()
    s1_num, s2_num = Lpoint_SE._scale_factor(lambda1_num, omega1_num, omega2_num)

    c_nums = {}
    for n in range(2, degree+1):
        c_sym = create_symbolic_cn(n)
        c_num = Lpoint_SE._cn(n)
        c_nums[c_sym] = c_num

    # Create substitution dictionary with all c symbols
    subs_dict = {
        lambda1: lambda1_num, 
        omega1: omega1_num, 
        omega2: omega2_num,
        s1: s1_num, 
        s2: s2_num
    }
    # Add all c symbols and their numerical values
    for c_sym, c_val in c_nums.items():
        subs_dict[c_sym] = c_val

    H_rnr = real_normal_center_manifold(Lpoint_SE, degree).subs(subs_dict)

    # Get formatted table of coefficients
    coeffs = extract_coeffs_up_to_degree(H_rnr, 5)
    logger.info(f"Found {len(coeffs)} coefficients")
    table = coefficients_to_table(coeffs, save=False)
    logger.info(table)


if __name__ == "__main__":
    main()
