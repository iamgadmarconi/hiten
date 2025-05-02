import symengine as se
import numpy as np

from .core import Polynomial
from system.libration import LibrationPoint

x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')
x_rn, y_rn, z_rn = se.symbols('x_rn y_rn z_rn')
px_rn, py_rn, pz_rn = se.symbols('px_rn py_rn pz_rn')
q1, q2, q3 = se.symbols('q1 q2 q3')
p1, p2, p3 = se.symbols('p1 p2 p3')

def _build_T_polynomials(N: int) -> list[se.Basic]:
    """Return [T0 … TN] using the Legendre recurrence (paper eq. 6)."""
    if N == 0:
        return [se.Integer(1)]
    if N == 1:
        return [se.Integer(1), x]
        
    T = [se.Integer(1), x]          # T0, T1
    for n in range(2, N + 1):
        n_ = se.Integer(n)
        Tn = ( (2*n_-1)/n_ ) * x * T[n-1] - ( (n_-1)/n_ ) * (x**2 + y**2 + z**2) * T[n-2]
        T.append(se.expand(Tn))
    return T


def hamiltonian(point: LibrationPoint, max_degree: int = 6) -> Polynomial:
    """
    Construct the Hamiltonian expansion (eq. 4) around the chosen collinear point,
    already translated & scaled so that (x,y,z)=(0,0,0) is the equilibrium.

    Returns a Polynomial in the *physical* coordinates (x,y,z,px,py,pz).
    """
    T = _build_T_polynomials(max_degree)

    c = [None, None]   # c_0, c_1 unused
    for n in range(2, max_degree + 1):
        c.append(point._cn(n))

    U = -se.Add(*[c[n] * T[n] for n in range(2, max_degree + 1)])

    K = se.Rational(1,2)*(px**2 + py**2 + pz**2) + y*px - x*py

    H_phys = se.expand(K + U)

    vars_phys = [x, y, z, px, py, pz]
    return Polynomial(vars_phys, H_phys)


def to_real_normal(point: LibrationPoint, H_phys: Polynomial) -> Polynomial:
    """
    Express the physical CR3BP Hamiltonian around a collinear point in the
    real normal-form variables (x,y,z,px,py,pz) used for the centre-
    manifold normal-form computation (Jorba & Masdemont, 1999).

    Parameters
    ----------
    point : LibrationPoint
        The LibrationPoint object
    H_phys : Polynomial
        The physical Hamiltonian to transform

    Returns
    -------
    Polynomial
        The transformed Hamiltonian in real normal-form variables
    """
    C_num, _ = point.normal_form_transform() # numpy array
    C = se.Matrix(C_num.tolist()) # to SymEngine

    Z_new = se.Matrix([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn])

    subs_dict = {}
    for i, var in enumerate([x, y, z, px, py, pz]):
        expr = 0
        for j in range(6):
            expr += C[i, j] * Z_new[j]
        subs_dict[var] = se.expand(expr)

    H_rn = H_phys.subs(subs_dict).expansion.expression

    return Polynomial([x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn], H_rn)

def to_complex_canonical(point: LibrationPoint, H_real_normal: Polynomial) -> Polynomial:
    """
    Express the physical CR3BP Hamiltonian around a collinear point in the
    complex canonical variables (q1,q2,q3,p1,p2,p3) used for the centre-
    manifold normal-form computation (Jorba & Masdemont, 1999).

    Parameters
    ----------
    point : LibrationPoint
        Must implement `normal_form_transform()` → (C, Cinv) with C \subset \mathbb{R}^{6 \times 6}.
    H_real_normal : Polynomial
        Hamiltonian in the real normal-form variables
        (x_rn, y_rn, z_rn, px_rn, py_rn, pz_rn).

    Returns
    -------
    Polynomial
        Same Hamiltonian written in (q1,q2,q3,p1,p2,p3).
    """

    sqrt2 = se.sqrt(2)
    complex_subs = {
        x_rn : q1,
        y_rn : (q2 + se.I*p2) / sqrt2,
        z_rn : (q3 + se.I*p3) / sqrt2,
        px_rn: p1,
        py_rn: (se.I*q2 + p2) / sqrt2,
        pz_rn: (se.I*q3 + p3) / sqrt2,
    }

    H_cn_expr = H_real_normal.subs(complex_subs).expansion.expression

    return Polynomial([q1, q2, q3, p1, p2, p3], H_cn_expr)

