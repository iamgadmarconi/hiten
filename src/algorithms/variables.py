import symengine as se


x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')
physical_vars = {'x': x, 'y': y, 'z': z, 'px': px, 'py': py, 'pz': pz}

x_rn, y_rn, z_rn = se.symbols('x_rn y_rn z_rn')
px_rn, py_rn, pz_rn = se.symbols('px_rn py_rn pz_rn')
real_normal_vars = {'x_rn': x_rn, 'y_rn': y_rn, 'z_rn': z_rn, 'px_rn': px_rn, 'py_rn': py_rn, 'pz_rn': pz_rn}

q1, q2, q3 = se.symbols('q1 q2 q3')
p1, p2, p3 = se.symbols('p1 p2 p3')
canonical_normal_vars = {'q1': q1, 'q2': q2, 'q3': q3, 'p1': p1, 'p2': p2, 'p3': p3}

omega1, omega2, lambda1, c2 = se.symbols('omega1 omega2 lambda1 c2')
linear_modes_vars = {'omega1': omega1, 'omega2': omega2, 'lambda1': lambda1, 'c2': c2}

eta1, eta2 = se.symbols('eta1 eta2')
eta_vars = {'eta1': eta1, 'eta2': eta2}

s1, s2 = se.symbols('s1 s2')
scale_factors_vars = {'s1': s1, 's2': s2}


def get_vars(vars_dict: dict) -> list:
    return [v for v in vars_dict.values()]

def create_symbolic_cn(n: int) -> se.Symbol:
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if n == 2:
        return c2
    return se.Symbol(f'c{n}')
