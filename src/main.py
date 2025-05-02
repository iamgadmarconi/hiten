import numpy as np
import symengine as se
import sympy as sp

from system.libration import L1Point
from algorithms.center.factory import hamiltonian, to_complex_canonical
from algorithms.center.core import _lie_transform


def main():
    mu_ES = 3.00348959632e-6
    L1_ES = L1Point(mu_ES)
    degree = 5

    # 1. full RTBP series in (q,p)
    H_full = hamiltonian(L1_ES, max_degree=degree)

    # 2. Lie-series normal form
    H_nf = _lie_transform(H_full, L1_ES.linear_modes(), max_degree=degree)

    # ---- original symbols ------------------------------------------------
    q1, q2, q3, p1, p2, p3 = H_nf.variables        # order guaranteed
    # 3. restrict to centre manifold q1 = p1 = 0
    H_cm = se.expand(H_nf.expression.subs({q1: 0, p1: 0}))

    # 4. complex → real canonical map
    Q2, P2, Q3, P3 = sp.symbols('Q2 P2 Q3 P3')
    to_real = {
        q2: (Q2 - se.I*P2)/se.sqrt(2),   p2: (Q2 + se.I*P2)/se.sqrt(2),
        q3: (Q3 - se.I*P3)/se.sqrt(2),   p3: (Q3 + se.I*P3)/se.sqrt(2),
    }
    H_real_se = se.expand(H_cm.subs(to_real)).subs({se.I: 0})

    H_real = sp.sympify(H_real_se)             #   ← add this

    def centre_coeffs(expr, Q2, P2, Q3, P3, *, max_deg):
        """
        Extract coefficients h_{k1,k2,k3,k4} of
            Q2^k1 P2^k2 Q3^k3 P3^k4     (0 < k_sum ≤ max_deg)
        and return them in a dict keyed by (k1,k2,k3,k4).
        Works with SymPy.
        """
        coeffs = {}
        for term in expr.as_ordered_terms():          # SymPy method
            c, monom = term.as_coeff_Mul()
            powers   = monom.as_powers_dict()

            k1 = powers.get(Q2, 0)
            k2 = powers.get(P2, 0)
            k3 = powers.get(Q3, 0)
            k4 = powers.get(P3, 0)
            if 0 < (tot := k1+k2+k3+k4) <= max_deg:
                coeffs[(k1, k2, k3, k4)] = float(c)
        return coeffs

    # 6. coefficient table
    tbl = centre_coeffs(H_real, Q2, P2, Q3, P3, max_deg=degree)

    for k, h in sorted(tbl.items()):
        print(f"{k}: {h:+.8e}")

if __name__ == "__main__":
    main()
