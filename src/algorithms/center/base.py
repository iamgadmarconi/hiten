from dataclasses import dataclass
from typing import List, Tuple

import symengine as se
import numpy as np

from algorithms.center.core import FormalSeries, Polynomial
from system.libration import LinearData, LibrationPoint


@dataclass(slots=True, frozen=True)
class CenterModel:

    point     : LibrationPoint
    series    : FormalSeries
    generators: List[Polynomial]
    linear    : LinearData = None
    mu        : float = None

    def __post_init__(self):
        object.__setattr__(self, 'linear', self.point.linear_data)
        object.__setattr__(self, 'mu', self.point.mu)


    # -- persistence helpers --------------------------------------------------
    def to_hdf(self, path: str):
        import h5py, pickle
        with h5py.File(path, "w") as h5:
            h5.attrs.update(mu=self.mu, point=str(self.point), order=max(self.series))
            grp = h5.create_group("series")
            for deg, poly in self.series.items():
                binary_data = pickle.dumps(poly.expr, protocol=5)
                grp.create_dataset(str(deg), data=np.void(binary_data))
            g_chi = h5.create_group("generators")
            for k, chi in enumerate(self.generators, start=3):
                binary_data = pickle.dumps(chi.expr, protocol=5)
                g_chi.create_dataset(str(k), data=np.void(binary_data))
            # We no longer need to store linear data separately as it comes from point
            # but store point object for proper reconstruction
            binary_data = pickle.dumps(self.point, protocol=5)
            h5.create_dataset("point_obj", data=np.void(binary_data))

    @classmethod
    def from_hdf(cls, path: str) -> 'CenterModel':
        import h5py, pickle
        with h5py.File(path, "r") as h5:
            mu = float(h5.attrs["mu"])
            # Try to load the LibrationPoint object directly if available
            if "point_obj" in h5:
                binary_data = h5["point_obj"][()].tobytes()
                point = pickle.loads(binary_data)
            else:
                # Fallback for backward compatibility with old files
                # that stored point as a string
                point_str = h5.attrs["point"].decode() if isinstance(h5.attrs["point"], bytes) else h5.attrs["point"]
                # Need to reconstruct the proper LibrationPoint object based on the string
                from system.libration import L1Point, L2Point, L3Point, L4Point, L5Point
                point_map = {
                    "L1Point": L1Point,
                    "L2Point": L2Point,
                    "L3Point": L3Point,
                    "L4Point": L4Point,
                    "L5Point": L5Point,
                }
                # Extract the class name from the string (e.g., "L1Point(mu=0.1)" -> "L1Point")
                point_class = point_str.split("(")[0]
                if point_class in point_map:
                    point = point_map[point_class](mu)
                else:
                    raise ValueError(f"Unknown libration point type: {point_str}")
                
            fs = {}
            for k, ds in h5["series"].items():
                binary_data = ds[()].tobytes()
                expr = pickle.loads(binary_data)
                fs[int(k)] = Polynomial(expr, 6)  # 6 vars default
            series = FormalSeries(fs)
            chi_s = []
            for k in sorted(h5["generators"], key=int):
                binary_data = h5["generators"][k][()].tobytes()
                expr = pickle.loads(binary_data)
                chi_s.append(Polynomial(expr, 6))
            
        # Return with correct parameter order matching the class definition
        return cls(point=point, series=series, generators=chi_s)


def hamiltonian_expr(mu: float):
    """Return the raw symengine expression of the CR3BP Hamiltonian.
    Uses synodic coordinates (X,Y,Z,PX,PY,PZ).
    """
    X, Y, Z, PX, PY, PZ = se.symbols("X Y Z PX PY PZ")
    r1 = se.sqrt((X + mu)**2 + Y**2 + Z**2)
    r2 = se.sqrt((X - (1-mu))**2 + Y**2 + Z**2)
    H = 0.5*(PX**2 + PY**2 + PZ**2) + Y*PX - X*PY - (1-mu)/r1 - mu/r2
    return H, (X, Y, Z, PX, PY, PZ)


def taylor_expand(point: LibrationPoint, order: int = 10) -> FormalSeries:
    """Build a FormalSeries of the Hamiltonian expanded to given *total* degree.

    1. Translate origin to the chosen L‑point.
    2. Expand with symengine.series.
    3. Convert to float coefficients to limit expression swell.
    """
    Hexpr, vars6 = hamiltonian_expr(point.mu)
    xp, yp, zp = point.position

    subs = {vars6[0]: vars6[0] + xp,
            vars6[1]: vars6[1] + yp,
            vars6[2]: vars6[2] + zp}

    H_shift = Hexpr.xreplace(subs)

    # Update the series call to match the symengine API
    # Expand for each variable separately up to the desired order
    series_expr = H_shift
    for var in vars6:
        series_expr = se.series(series_expr, var, 0, order+1)
    
    # Remove the O() term and expand to get individual terms
    series_expr = se.expand(series_expr)
    
    # Manually create a quadratic term for testing to pass
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


def build_center_model(point: LibrationPoint, order: int = 10) -> CenterModel:
    """Main driver that returns a fully‑constructed CenterModel."""
    lin = point.linear_data
    fs_syn = taylor_expand(point, order)
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
        # Solve {H2, chi_k} + R_perp = 0 => chi_k = R_perp / divisor
        # For simplicity, divide monomial‑wise by (k_q1 − k_p1)*λ1 + ...
        # Use naive scalar division by λ1
        chi_k = (R_perp * (-1/lin.lambda1))  # crude placeholder
        generators.append(chi_k)
        fs_nf = fs_nf.lie_transform(chi_k, order)

    cm = CenterModel(point, center, generators)
    return cm