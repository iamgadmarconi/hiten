from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, MutableMapping, Tuple

import numpy as np
import symengine as se


def symplectic_dot(grad: np.ndarray) -> np.ndarray:
    """Convert \nabla H = (∂H/∂q, ∂H/∂p) into the Hamiltonian vector field.
    Assumes 3 DOF (len = 6)."""
    if grad.size != 6:
        raise ValueError("symplectic_dot expects a 6‑vector of gradients.")
    dq = grad[3:]
    dp = -grad[:3]
    return np.concatenate((dq, dp))


class FormalSeries(MutableMapping[int, "Polynomial"]):
    """Sparse homogeneous power series.

    Keys are *total* degree (>= 2).  Value type is assumed to behave like
    `Polynomial` (symengine expression wrapper) but *not* imported here to
    keep this module independent.
    """

    def __init__(self, mapping: Dict[int, "Polynomial"] | None = None):
        self._data: Dict[int, "Polynomial"] = {}
        if mapping:
            for k, v in mapping.items():
                self[k] = v  # uses __setitem__ validation

    # --- mutable‑mapping protocol -------------------------------------------
    def __getitem__(self, key: int):
        return self._data[key]

    def __setitem__(self, key: int, value: "Polynomial"):
        if not isinstance(key, int) or key < 0:
            raise KeyError("Degree must be a non‑negative integer.")
        self._data[key] = value

    def __delitem__(self, key: int):
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # ---------------------------------------------------------------------
    def degrees(self) -> List[int]:
        return sorted(self._data.keys())

    # ---------------------------------------------------------------------
    def truncate(self, max_degree: int) -> "FormalSeries":
        """Return a shallow copy containing only terms ≤ max_degree."""
        return FormalSeries({k: v for k, v in self._data.items() if k <= max_degree})

    # ---------------------------------------------------------------------
    @staticmethod
    def poisson_pair(series1: "FormalSeries", series2: "FormalSeries", degree: int):
        """Return the homogeneous degree-`degree` part of {S1, S2}."""
        # Leibniz: look at all pairs k + j - 2 == degree (since PB adds degrees‑2)
        res = None
        for k, pk in series1._data.items():
            j = degree + 2 - k
            if j in series2._data:
                term = pk.poisson_bracket(series2[j])
                res = term if res is None else res + term
        # Return None if res is None or if it's a zero polynomial
        return None if res is None or res.expr == se.sympify(0) else res

    # ---------------------------------------------------------------------
    def lie_transform(self, chi: "Polynomial", k_max: int) -> "FormalSeries":
        """Apply exp(L_χ) truncated to degree *k_max*.

        Assumes χ is homogeneous of degree ≥ 3.
        """
        out = FormalSeries(self._data.copy())
        # Baker–Campbell–Hausdorff truncated form:
        #   H ← H + {χ, H} + 1/2{{χ,χ},H}+ ...
        # but we exploit the graded structure and use recursive update
        for d in range(2, k_max + 1):
            if (d in out._data):
                bracket = chi.poisson_bracket(out[d])
                if bracket and bracket.expr != se.sympify(0):
                    out[d] = out[d] + bracket
        return out

    # ---------------------------------------------------------------------
    def __str__(self):
        terms = ", ".join(f"deg{d}" for d in self.degrees())
        return f"FormalSeries({terms})"

    __repr__ = __str__


@dataclass(slots=True)
class Hamiltonian:
    series: FormalSeries
    mu: float
    coords: str = "synodic"  # 'synodic', 'normal', 'center'
    order: int = field(init=False)

    def __post_init__(self):
        self.order = max(self.series.degrees()) if len(self.series) else 0

    # ---------------------------------------------------------------------
    def quadratic(self):
        return self.series[2]

    # ---------------------------------------------------------------------
    def evaluate(self, x: np.ndarray, float_only: bool = True) -> float:
        """Evaluate ∑ H_k(x).

        Parameters
        ----------
        x : np.ndarray length = n_vars
        float_only : if True, cast result to float (double‑precision).
        """
        val = sum(p.evaluate(x) for p in self.series.values())
        return float(val) if float_only else val

    # ---------------------------------------------------------------------
    def gradient(self, x: np.ndarray) -> np.ndarray:
        return sum(p.gradient(x) for p in self.series.values())

    # ---------------------------------------------------------------------
    def vector_field(self, x: np.ndarray) -> np.ndarray:
        return symplectic_dot(self.gradient(x))

    # ---------------------------------------------------------------------
    def poisson(self, other: "Hamiltonian") -> "Hamiltonian":
        max_deg = self.order + other.order - 2
        data: Dict[int, "Polynomial"] = {}
        for d in range(2, max_deg + 1):
            val = FormalSeries.poisson_pair(self.series, other.series, d)
            if val is not None:
                data[d] = val
        return Hamiltonian(FormalSeries(data), self.mu, self.coords)

    # ---------------------------------------------------------------------
    def change_variables(self, transform) -> "Hamiltonian":
        """Return a new Hamiltonian H∘T where `transform` is a callable that
        maps a Polynomial to another Polynomial (e.g. linear C⁻¹ map, Lie
        transform, centre-manifold injection…)."""
        new_series = FormalSeries({k: transform(p) for k, p in self.series.items()})
        return Hamiltonian(new_series, self.mu, self.coords)

    # ---------------------------------------------------------------------
    # I/O helpers ----------------------------------------------------------
    def to_hdf(self, path: str):
        import h5py
        with h5py.File(path, "w") as f:
            f.attrs["mu"] = self.mu
            f.attrs["coords"] = self.coords
            grp = f.create_group("series")
            for d, poly in self.series._data.items():
                grp.create_dataset(str(d), data=str(poly.expr))  # naive store

    @staticmethod
    def from_hdf(path: str) -> "Hamiltonian":
        import symengine as se
        import h5py
        from algorithms.center.polynomials import Polynomial  # correct import

        with h5py.File(path, "r") as f:
            mu = float(f.attrs["mu"])
            coords = f.attrs["coords"]
            data: Dict[int, Polynomial] = {}
            for name, ds in f["series"].items():
                deg = int(name)
                expr = se.sympify(ds[()].decode())
                data[deg] = Polynomial(expr)
        return Hamiltonian(FormalSeries(data), mu, coords)

    # ---------------------------------------------------------------------
    def __str__(self):
        return f"Hamiltonian(order≤{self.order}, coords='{self.coords}', μ={self.mu})"

    __repr__ = __str__
