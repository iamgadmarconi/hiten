import numpy as np
import symengine as se
import sympy as sp

from system.base import System, systemConfig
from system.body import Body
from system.libration import L1Point

from algorithms.propagators import propagate_crtbp

from algorithms.center.factory import hamiltonian, to_complex_canonical
from algorithms.center.core import _lie_transform

def main():
    # 1. pick the correct system and expansion order ------------------------
    mu_ES   = 3.00348959632e-6                      # Earth–Sun mass ratio
    L1_ES    = L1Point(mu_ES)                       # Earth–Sun L1
    degree = 5
    H_full   = hamiltonian(L1_ES, max_degree=degree)   # Section 2 expansion

    # 2. Lie–series normal form (your cached routine) -----------------------
    H_nf     = _lie_transform(H_full, max_degree=degree)  # Section 3.1

    # 3. restrict to the centre manifold  (q1 = p1 = 0) --------------------
    q1, p1 = se.Symbol('q1'), se.Symbol('p1')  # the actual symbols
    H_cm    = H_nf.expression.subs({q1: 0, p1: 0}).expand()

    # 4. switch from complex (q2,p2,q3,p3) to real (Q2,P2,Q3,P3) -----------
    q2, p2, q3, p3 = se.symbols('q2 p2 q3 p3')  # Get symbol objects
    Q2, P2, Q3, P3 = se.symbols('Q2 P2 Q3 P3')  # real canonical pair
    to_real = {
        q2: (Q2 - se.I*P2)/se.sqrt(2),   p2: (Q2 + se.I*P2)/se.sqrt(2),
        q3: (Q3 - se.I*P3)/se.sqrt(2),   p3: (Q3 + se.I*P3)/se.sqrt(2),
    }

    H_real = se.sympify(se.expand(H_cm.subs(to_real)).subs({se.I: 0}))

    # 5. grab the coefficients exactly as Table 1 does ----------------------
    def centre_coeffs(expr, max_deg):
        coeffs = {}
        Q2, P2, Q3, P3 = sp.symbols('Q2 P2 Q3 P3')    # the same names

        for term in expr.as_ordered_terms():          # SymPy method
            c, monom = term.as_coeff_Mul()
            powers   = monom.as_powers_dict()

            k1 = powers.get(Q2, 0)
            k2 = powers.get(P2, 0)
            k3 = powers.get(Q3, 0)
            k4 = powers.get(P3, 0)
            tot = k1 + k2 + k3 + k4

            if 0 < tot <= max_deg:
                coeffs[(k1, k2, k3, k4)] = float(c)
        return coeffs

    tbl = centre_coeffs(H_real, max_deg=degree)
    for k, h in sorted(tbl.items()):
        print(f"{k}: {h:+.8e}")


if __name__ == "__main__":
    main()
