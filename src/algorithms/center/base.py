from dataclasses import dataclass
from typing import List, Tuple

import symengine as se
import numpy as np

from algorithms.center.core import FormalSeries, Polynomial
from system.libration import LinearData, LibrationPoint


@dataclass(slots=True, frozen=True)
class CenterModel:
    """Container for the truncated 2‑DOF Hamiltonian produced by CMR.

    Attributes
    ----------
    mu         : CR3BP mass parameter.
    point      : 'L1', 'L2', or 'L3'.
    series     : FormalSeries – centre Hamiltonian in (q2,p2,q3,p3).
    generators : list[Polynomial] – χ₃, χ₄, … used in Lie transform.
    linear     : LinearData – output from Step‑A (λ₁, ω₁, ω₂, C, Cinv).
    """

    mu        : float
    point     : str
    series    : FormalSeries
    generators: List[Polynomial]
    linear    : 'LinearData'

    # -- persistence helpers --------------------------------------------------
    def to_hdf(self, path: str):
        import h5py, pickle
        with h5py.File(path, "w") as h5:
            h5.attrs.update(mu=self.mu, point=self.point, order=max(self.series))
            grp = h5.create_group("series")
            for deg, poly in self.series.items():
                binary_data = pickle.dumps(poly.expr, protocol=5)
                grp.create_dataset(str(deg), data=np.void(binary_data))
            gχ = h5.create_group("generators")
            for k, chi in enumerate(self.generators, start=3):
                binary_data = pickle.dumps(chi.expr, protocol=5)
                gχ.create_dataset(str(k), data=np.void(binary_data))
            binary_data = pickle.dumps(self.linear, protocol=5)
            h5.create_dataset("linear", data=np.void(binary_data))

    @classmethod
    def from_hdf(cls, path: str) -> 'CenterModel':
        import h5py, pickle
        with h5py.File(path, "r") as h5:
            mu    = float(h5.attrs["mu"])
            point = h5.attrs["point"].decode() if isinstance(h5.attrs["point"], bytes) else h5.attrs["point"]
            fs = {}
            for k, ds in h5["series"].items():
                binary_data = ds[()].tobytes()
                expr = pickle.loads(binary_data)
                fs[int(k)] = Polynomial(expr, 6)  # 6 vars default
            series = FormalSeries(fs)
            χs = []
            for k in sorted(h5["generators"], key=int):
                binary_data = h5["generators"][k][()].tobytes()
                expr = pickle.loads(binary_data)
                χs.append(Polynomial(expr, 6))
            binary_data = h5["linear"][()].tobytes()
            linear = pickle.loads(binary_data)
        return cls(mu, point, series, χs, linear)


def hamiltonian_expr(mu: float):
    """Return the raw symengine expression of the CR3BP Hamiltonian.
    Uses synodic coordinates (X,Y,Z,PX,PY,PZ).
    """
    X, Y, Z, PX, PY, PZ = se.symbols("X Y Z PX PY PZ")
    r1 = se.sqrt((X + mu)**2 + Y**2 + Z**2)
    r2 = se.sqrt((X - (1-mu))**2 + Y**2 + Z**2)
    H = 0.5*(PX**2 + PY**2 + PZ**2) + Y*PX - X*PY - (1-mu)/r1 - mu/r2
    return H, (X, Y, Z, PX, PY, PZ)


def taylor_expand(mu: float, Lj: str, order: int = 10) -> FormalSeries:
    """Build a FormalSeries of the Hamiltonian expanded to given *total* degree.

    1. Translate origin to the chosen L‑point.
    2. Expand with symengine.series(removeO).
    3. Convert to float coefficients to limit expression swell.
    """
    Hexpr, vars6 = hamiltonian_expr(mu)
    xp, yp, zp = get_equilibrium_point(mu, Lj)  # returns 3‑tuple

    subs = {vars6[0]: vars6[0] + xp,
            vars6[1]: vars6[1] + yp,
            vars6[2]: vars6[2] + zp}

    H_shift = Hexpr.xreplace(subs)

    series_expr = se.series(H_shift, *vars6, order+1).removeO()
    # Split by total degree
    fs: dict[int, Polynomial] = {}
    for term in se.expand(series_expr).expand().as_ordered_terms():
        poly = Polynomial(term, 6).as_float()
        deg  = poly.total_degree()
        fs.setdefault(deg, Polynomial.zero(6))
        fs[deg] = fs[deg] + poly
    return FormalSeries(fs)


def apply_linear_change(fs: FormalSeries, lin) -> FormalSeries:
    """Substitute (X,Y,Z,PX,PY,PZ) = C @ (q,p).
    The matrix `lin.C` is 6×6 symplectic.
    """
    C = lin.C  # numpy array
    Xsym = se.MatrixSymbol("q", 6, 1)  # placeholder symbols q0..q5
    q_syms = se.symbols("q0:6")
    subs = {}
    for i, var in enumerate(fs.variables if hasattr(fs, 'variables') else []):
        linear_expr = sum(C[i, j]*q_syms[j] for j in range(6))
        subs[var] = linear_expr
    out = {}
    for deg, poly in fs.items():
        out[deg] = poly.substitute(subs)  # Polynomial substitute method
    return FormalSeries(out)


def kill_saddle_terms(poly: Polynomial) -> Tuple[Polynomial, Polynomial]:
    """Return (Z_k, R_perp) as in normal‑form theory.

    Z_k  – monomials with no q1 or p1 (indices 0 and 1)
    R⊥   – the rest, to be removed via homological equation.
    """
    Zexpr = 0
    Rexpr = 0
    q1, p1 = poly.variables[0], poly.variables[1]
    for term in poly.expr.as_ordered_terms():
        if term.has(q1) or term.has(p1):
            Rexpr += term
        else:
            Zexpr += term
    return Polynomial(Zexpr, poly.n_vars), Polynomial(Rexpr, poly.n_vars)


def build_center_model(mu: float, Lj: str, order: int = 10) -> CenterModel:
    """Main driver that returns a fully‑constructed CenterModel."""
    lin = compute_linear_data(mu, Lj)
    fs_syn = taylor_expand(mu, Lj, order)
    fs_nf = apply_linear_change(fs_syn, lin)

    center = FormalSeries({2: fs_nf[2]})
    generators: List[Polynomial] = []

    for k in range(3, order+1):
        Rk = fs_nf[k]
        Zk, R_perp = kill_saddle_terms(Rk)
        center[k] = Zk
        if R_perp.expr == 0:
            generators.append(Polynomial.zero(6))
            continue
        # Solve {H2, χk} + R_perp = 0 => χk = R_perp / divisor
        # For simplicity, divide monomial‑wise by (k_q1 − k_p1)*λ1 + ...
        # Use naive scalar division by λ1
        chi_k = (R_perp * (-1/lin.lambda1))  # crude placeholder
        generators.append(chi_k)
        fs_nf = fs_nf.lie_transform(chi_k, order)

    cm = CenterModel(mu, Lj, center, generators, lin)
    return cm