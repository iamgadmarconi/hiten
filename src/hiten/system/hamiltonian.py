"""Base classes for Hamiltonian representations in the CR3BP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import sympy as sp

from hiten.algorithms.polynomial.conversion import poly2sympy
from hiten.algorithms.polynomial.operations import _polynomial_evaluate
from hiten.algorithms.types.services.hamiltonian import (
    _HamiltonianServices,
    get_hamiltonian_services,
)
from hiten.algorithms.types.core import _HitenBase


class Hamiltonian(_HitenBase):
    """User-facing container delegating Hamiltonian numerics to adapters."""

    def __init__(
        self,
        poly_H: list[np.ndarray],
        degree: int,
        ndof: int = 3,
        name: str = "Hamiltonian",
    ) -> None:
        if degree <= 0:
            raise ValueError("degree must be a positive integer")

        if ndof != 3:
            raise NotImplementedError("Polynomial kernel only supports 3 degrees of freedom")

        self._services = get_hamiltonian_services()

        self._poly_H: list[np.ndarray] = poly_H
        self._degree: int = degree
        self._ndof: int = ndof
        self._name: str = name
        self._psi, self._clmo = self._services.dynamics.init_tables(degree)
        self._encode_dict_list = self._services.dynamics.build_encode_dict(self._clmo)
        self._hamsys = self._services.dynamics.build_hamsys(
            poly_H,
            degree,
            self._psi,
            self._clmo,
            self._encode_dict_list,
            ndof,
            name,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def poly_H(self) -> list[np.ndarray]:
        return self._poly_H

    @property
    def degree(self) -> int:
        return self._degree

    @property
    def ndof(self) -> int:
        return self._ndof

    def __len__(self) -> int:
        return len(self._poly_H)

    def __getitem__(self, key):
        return self._poly_H[key]

    def __call__(self, coords: np.ndarray) -> float:
        return _polynomial_evaluate(self._poly_H, coords, self._clmo)

    @property
    def jacobian(self) -> np.ndarray:
        return self._hamsys.jac_H

    @property
    def hamsys(self):
        if self._hamsys is None:
            self._hamsys = self._services.dynamics.build_hamsys(
                self._poly_H,
                self._degree,
                self._psi,
                self._clmo,
                self._encode_dict_list,
                self._ndof,
                self._name,
            )
        return self._hamsys

    def save(self, filepath: str | Path, **kwargs) -> None:
        self._services.persistence.save(self, filepath, **kwargs)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        services = get_hamiltonian_services()
        ham = services.persistence.load(filepath, **kwargs)
        ham._services = services
        return ham

    def to_state(self, target_form: Union[type["Hamiltonian"], str], **kwargs) -> "Hamiltonian":
        return self._services.conversion.convert(self, target_form, **kwargs)

    @classmethod
    def from_state(cls, other: "Hamiltonian", **kwargs) -> "Hamiltonian":
        services = other._services
        result = services.conversion.convert(other, cls, **kwargs)
        return cls(result.poly_H, result.degree, result.ndof, result.name, services=services)

    @staticmethod
    def register_conversion(
        src: str,
        dst: str,
        converter: Callable,
        required_context: list,
        default_params: dict,
    ) -> None:
        services = get_hamiltonian_services()
        services.conversion.register(src, dst, converter, required_context, default_params)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', degree={self.degree}, "
            f"blocks={len(self)})"
        )
    
    def __str__(self) -> str:
        q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
        return poly2sympy(self._poly_H, [q1, q2, q3, p1, p2, p3], self._psi, self._clmo)

    def __bool__(self):
        return bool(self._poly_H)


class LieGeneratingFunction:
    """Class for Lie generating functions in canonical transformations."""

    def __init__(
        self,
        poly_G: list[np.ndarray],
        poly_elim: list[np.ndarray],
        degree: int,
        ndof: int = 3,
        name: str | None = None,
        *,
        services: _HamiltonianServices | None = None,
    ) -> None:
        self._services = services or get_hamiltonian_services()
        self._poly_G = poly_G
        self._poly_elim = poly_elim
        self._degree = degree
        self._ndof = ndof
        self._name = name
        self._psi, self._clmo = self._services.dynamics.init_tables(degree)
        self._encode_dict_list = self._services.dynamics.build_encode_dict(self._clmo)

    @property
    def poly_G(self) -> list[np.ndarray]:
        """Return the packed coefficient blocks `[G_0, G_2, ..., G_N]`."""
        return self._poly_G
    
    @property
    def degree(self) -> int:
        """Return the maximum total degree *N* represented in *poly_G*."""
        return self._degree

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""
        return self._ndof

    @property
    def poly_elim(self) -> list[np.ndarray]:
        return self._poly_elim

    @property
    def name(self) -> str:
        return self._name



