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

    H_rn = se.expand(H_phys.subs(subs_dict))

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

    # Apply substitutions directly to the Hamiltonian expression
    H_cn_expr = se.expand(H_real_normal.subs(complex_subs))
    return Polynomial([q1, q2, q3, p1, p2, p3], H_cn_expr)

def lie_transform(H_init: Polynomial, lambda1: float, omega1: float, omega2: float, max_deg: int) -> tuple[Polynomial, list[Polynomial]]:
    """Bring *H_init* to the Jorba-Masdemont partial normal form up to
    *max_deg* using a Lie-series expansion.  Returns the transformed
    Hamiltonian and the list of generators [G_3, …, G_max_deg].

    Parameters
    ----------
    H_init : Polynomial
        Hamiltonian already expressed in complex canonical variables
        (q1,q2,q3,p1,p2,p3), with a working ``build_by_degree`` helper.
    max_deg : int
        Highest total degree to normalise and keep.
    lambda1, omega1, omega2 : (mp.mpf | float | symengine)  
        lambda, omega1, omega2 from the linear dynamics.
    """
    # Internal working copy (will be updated in‑place).
    H = H_init.copy()  # implement copy() in Polynomial if missing
    by_deg = H.build_by_degree()
    G_list = []  # store generators for optional post‑analysis

    # Convenience unpack
    q1, q2, q3, p1, p2, p3 = H.variables

    for n in range(3, max_deg + 1):
        # --- 1. Homogeneous component of degree n ---
        Hn_expr = se.expand(sum(monomial.sym for monomial in by_deg.get(n, [])))
        if Hn_expr == 0:
            G_list.append(None)
            continue
        Hn = H.__class__(H.variables, Hn_expr)

        # --- 2. Build the generating function G_n ---
        Gn_terms = []
        for coeff, monom in Hn.iter_terms():
            kq, kp = monom.exponents_qp_split()

            # Term‑selection rule (paper eq. 17): remove only monomials
            # with *different* exponents in the hyperbolic coordinate.
            if kq[0] == kp[0]:
                # This term already lives in the normal form ⇒ keep.
                continue

            # ⟨k_p − k_q , η⟩
            divisor = ((kp[0] - kq[0]) * lambda1 +
                       (kp[1] - kq[1]) * se.I * omega1 +
                       (kp[2] - kq[2]) * se.I * omega2)
            if divisor == 0:
                # Exact resonance (should not occur for centre × saddle).
                continue

            Gn_terms.append(coeff / divisor * monom.sym)

        if not Gn_terms:  # nothing to eliminate at this order
            G_list.append(None)
            continue

        Gn = H.__class__(H.variables, se.expand(sum(Gn_terms)))
        G_list.append(Gn)

        # --- 3. Exponential Lie map via truncated BCH series ---
        # We build exp(L_G) H as H + {G,H} + 1/2{{G,{G,H}}} + … until the
        # new term exceeds *max_deg*.
        delta = Gn.poisson(H)
        k = 1
        while delta.total_degree() <= max_deg and delta.expression != 0:
            H += delta / k
            k += 1
            delta = Gn.poisson(delta)

        # --- 4. Truncate to desired degree and refresh cache ---
        H.truncate(max_deg)
        by_deg = H.build_by_degree()

    return H, G_list