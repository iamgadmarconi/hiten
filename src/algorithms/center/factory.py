import symengine as se
import numpy as np


from .core import Polynomial
from system.libration import LibrationPoint


x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')


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


def to_complex_canonical(point: LibrationPoint,
                         H_phys: Polynomial) -> Polynomial:
    """
    Express the physical CR3BP Hamiltonian around a collinear point in the
    complex canonical variables (q1,q2,q3,p1,p2,p3) used for the centre-
    manifold normal-form computation (Jorba & Masdemont, 1999).

    Parameters
    ----------
    point : LibrationPoint
        Must implement `normal_form_transform()` → (C, Cinv) with C \subset \mathbb{R}^{6 \times 6}.
    H_phys : Polynomial
        Hamiltonian in the translated/scaled Cartesian variables
        (x, y, z, px, py, pz).

    Returns
    -------
    Polynomial
        Same Hamiltonian written in (q1,q2,q3,p1,p2,p3).
    """

    # ------------------------------------------------------------------
    # 0. Symbols
    # ------------------------------------------------------------------
    x, y, z  = se.symbols('x y z')
    px, py, pz = se.symbols('px py pz')
    q1, q2, q3, p1, p2, p3 = se.symbols('q1 q2 q3 p1 p2 p3')

    # ------------------------------------------------------------------
    # 1. Real normal-form change  Z_old = C^{-1} · Z_new
    # ------------------------------------------------------------------
    C_num, Cinv_num = point.normal_form_transform()       # numpy array
    C = se.Matrix(C_num.tolist())               # to SymEngine
    Cinv = se.Matrix(Cinv_num.tolist())               # to SymEngine

    xt, yt, zt, pxt, pyt, pzt = se.symbols('xt yt zt pxt pyt pzt')
    Z_tilde   = se.Matrix([xt, yt, zt, pxt, pyt, pzt])
    Z_old_exp = (Cinv * Z_tilde).applyfunc(se.expand) # list of 6 expressions

    # ------------------------------------------------------------------
    # 2. Complex canonical substitution  (Eq. 12)
    # ------------------------------------------------------------------
    sqrt2 = se.sqrt(2)
    complex_subs = {
        xt : q1,
        pxt: p1,
        yt : (q2 + se.I*p2) / sqrt2,
        pyt: (se.I*q2 + p2) / sqrt2,
        zt : (q3 + se.I*p3) / sqrt2,
        pzt: (se.I*q3 + p3) / sqrt2,
    }

    # ------------------------------------------------------------------
    # 3. Build full mapping  (x,y,z,px,py,pz) → expressions(q,p)
    # ------------------------------------------------------------------
    subs_dict = {
        old_sym : expr.subs(complex_subs)
        for old_sym, expr in zip((x, y, z, px, py, pz), Z_old_exp)
    }

    # ------------------------------------------------------------------
    # 4. Apply and wrap up
    # ------------------------------------------------------------------
    H_cc_expr = se.expand(H_phys.expression.subs(subs_dict))
    return Polynomial([q1, q2, q3, p1, p2, p3], H_cc_expr)