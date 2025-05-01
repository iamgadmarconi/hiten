import symengine as se
import numpy as np


from .core import Polynomial
from system.libration import LibrationPoint


x, y, z  = se.symbols('x y z')
px, py, pz = se.symbols('px py pz')


def _build_T_polynomials(N: int) -> list[se.Basic]:
    """Return [T0 … TN] using the Legendre recurrence (paper eq. 6)."""

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
    Apply the linear symplectic matrix C (eq. 10) and the complex map (eq. 12)
    so that H₂ appears as λ₁q₁p₁ + iω₁q₂p₂ + iω₂q₃p₃.

    Resulting variables order: q1,q2,q3,p1,p2,p3  (all SymEngine symbols).
    """
    # 3.1 symbols
    q1,q2,q3,p1,p2,p3 = se.symbols('q1 q2 q3 p1 p2 p3')

    # 3.2 real → complex map (paper eq. 12)
    subs_complex = {
        x: q1,
        y: (q2 + se.I*p2)/se.sqrt(2),
        z: (q3 + se.I*p3)/se.sqrt(2),
        px: p1,
        py: (se.I*q2 + p2)/se.sqrt(2),
        pz: (se.I*q3 + p3)/se.sqrt(2),
    }

    # 3.3 linear normal-form matrix C
    C, Cinv = point.normal_form_transform()
    # build a SymEngine matrix for substitution:   Z = C ⋅ (x,y,z,px,py,pz)^T
    v_old = se.Matrix([x, y, z, px, py, pz])
    v_new = Cinv @ v_old      # numeric for given μ
    
    # Convert each NumPy array element to a SymEngine expression
    subs_C = {}
    for i, old_var in enumerate(v_old):
        # Handle both NumPy array and other cases
        if isinstance(v_new[i], np.ndarray):
            # Convert NumPy array element to string, then to SymEngine expression
            val = se.sympify(str(v_new[i].item()))
        else:
            val = se.sympify(str(v_new[i]))
        subs_C[old_var] = se.expand(val)

    # apply C first, then complexification
    expr = H_phys.expression.subs(subs_C).subs(subs_complex).expand()

    return Polynomial([q1,q2,q3,p1,p2,p3], expr)
