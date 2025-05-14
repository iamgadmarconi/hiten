import symengine as se
import numpy as np
from numba import njit

from algorithms.center.polynomial.base import (
    init_index_tables, decode_multiindex
)
from algorithms.center.polynomial.conversions import (
    poly2symengine, symengine2poly
)
from algorithms.center.lie import lie_transform
from algorithms.variables import (
    physical_vars, real_normal_vars, canonical_normal_vars,
    get_vars, create_symbolic_cn,
    linear_modes_vars, scale_factors_vars, N_VARS
)

x, y, z, px, py, pz = get_vars(physical_vars)
x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn = get_vars(real_normal_vars)
q1, q2, q3, p1, p2, p3 = get_vars(canonical_normal_vars)
omega1, omega2, lambda1, c2 = get_vars(linear_modes_vars)
s1, s2 = get_vars(scale_factors_vars)

