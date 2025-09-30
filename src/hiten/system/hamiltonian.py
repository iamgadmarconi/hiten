"""Base classes for Hamiltonian representations in the CR3BP."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import numpy as np
import sympy as sp

from hiten.algorithms.polynomial.conversion import poly2sympy
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.hamiltonian import (
    _HamiltonianDynamicsService, _HamiltonianPersistenceService,
    _HamiltonianServices, _LieGeneratingFunctionPersistenceService,
    _LieGeneratingFunctionServices)


class Hamiltonian(_HitenBase):
    """User-facing container delegating Hamiltonian numerics to adapters.
    
    Parameters
    ----------
    poly_H : list[np.ndarray]
        The polynomial Hamiltonian blocks.
    degree : int
        The degree of the Hamiltonian.
    ndof : int, default 3
        The number of degrees of freedom.
    name : str, default "Hamiltonian"
        The name of the Hamiltonian.
    """

    def __init__(self, poly_H: list[np.ndarray], degree: int, ndof: int = 3, name: str = "Hamiltonian") -> None:
        if degree <= 0:
            raise ValueError("degree must be a positive integer")

        if ndof != 3:
            raise NotImplementedError("Polynomial kernel only supports 3 degrees of freedom")

        self._poly_H: list[np.ndarray] = poly_H
        self._degree: int = degree
        self._ndof: int = ndof
        self._name: str = name

        services = _HamiltonianServices.default(self)
        super().__init__(services)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', degree={self.degree}, "
            f"blocks={len(self)})"
        )

    def __str__(self) -> str:
        q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
        return str(poly2sympy(self.poly_H, [q1, q2, q3, p1, p2, p3], self.dynamics.psi, self.dynamics.clmo))

    def __bool__(self):
        return bool(self.poly_H)

    def __len__(self) -> int:
        return len(self.poly_H)

    def __getitem__(self, key):
        return self.poly_H[key]

    def __call__(self, coords: np.ndarray) -> float:
        """Evaluate the Hamiltonian at the given coordinates."""
        return self.dynamics.evaluate(coords)

    @property
    def name(self) -> str:
        """Return the name of the Hamiltonian."""
        return self._name

    @property
    def degree(self) -> int:
        """Return the degree of the Hamiltonian."""
        return self._degree

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""
        return self._ndof

    @property
    def hamsys(self):
        """Return the Hamiltonian system."""
        return self.dynamics.hamsys

    @property
    def jacobian(self) -> np.ndarray:
        """Return the Jacobian of the Hamiltonian."""
        return self.dynamics.jac_H

    @property
    def poly_H(self) -> list[np.ndarray]:
        """Return the polynomial Hamiltonian blocks."""
        return self.dynamics.poly_H

    @classmethod
    def from_state(cls, other: "Hamiltonian", **kwargs) -> "Hamiltonian":
        """Convert another Hamiltonian to this class using the dynamics service."""
        # Create a temporary dynamics service to handle the conversion
        temp_dynamics = _HamiltonianDynamicsService(other)
        return temp_dynamics.from_state(other, cls, **kwargs)

    def to_state(self, target_form: Union[type["Hamiltonian"], str], **kwargs) -> "Hamiltonian":
        """Convert this Hamiltonian to another form using the dynamics service."""
        return self.dynamics.to_state(target_form, **kwargs)

    def __setstate__(self, state):
        """Restore the Hamiltonian instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of poly_H, degree, ndof, and name.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the Hamiltonian.
        """
        super().__setstate__(state)
        self._setup_services(_HamiltonianServices.default(self))

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "Hamiltonian":
        """Load a Hamiltonian from a file (new instance).
        
        Parameters
        ----------
        filepath : str or Path
            The path to the file to load the Hamiltonian from.
        **kwargs
            Additional keyword arguments for the load operation.
        """
        return cls._load_with_services(
            filepath, 
            _HamiltonianPersistenceService(),
            _HamiltonianServices.default, 
            **kwargs
        )

    @staticmethod
    def register_conversion(
        src: str,
        dst: str,
        converter: Callable,
        required_context: list,
        default_params: dict,
    ) -> None:
        """Register a conversion function using the shared conversion service."""
        from hiten.algorithms.types.services.hamiltonian import \
            _SHARED_REGISTRY
        _SHARED_REGISTRY.register_conversion(src, dst, converter, required_context, default_params)


class LieGeneratingFunction(_HitenBase):
    """Class for Lie generating functions in canonical transformations.
    
    Parameters
    ----------
    poly_G : list[np.ndarray]
        The polynomial G blocks.
    poly_elim : list[np.ndarray]
        The polynomial elimination blocks.
    degree : int
        The degree of the Lie generating function.
    ndof : int, default 3
        The number of degrees of freedom.
    name : str, default "LieGeneratingFunction"
        The name of the Lie generating function.
    """

    def __init__(
        self,
        poly_G: list[np.ndarray],
        poly_elim: list[np.ndarray],
        degree: int,
        ndof: int = 3,
        name: str = "LieGeneratingFunction",
    ) -> None:
        self._poly_G = poly_G
        self._poly_elim = poly_elim
        self._degree = degree
        self._ndof = ndof
        self._name = name

        services = _LieGeneratingFunctionServices.default(self)
        super().__init__(services)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', degree={self.degree}, "
            f"blocks={len(self)})"
        )
    
    def __str__(self) -> str:
        return str(poly2sympy(self.poly_G, self.poly_elim, self.dynamics.psi, self.dynamics.clmo))
    
    def __bool__(self):
        return bool(self.poly_G)
    
    def __len__(self) -> int:
        return len(self.poly_G)
    
    def __getitem__(self, key):
        return self.poly_G[key]
    
    def __call__(self, coords: np.ndarray) -> float:
        """Evaluate the Lie generating function at the given coordinates."""
        return self.dynamics.evaluate(coords)

    @property
    def poly_G(self) -> list[np.ndarray]:
        """Return the packed coefficient blocks `[G_0, G_2, ..., G_N]`."""
        return self.dynamics.poly_G
    
    @property
    def degree(self) -> int:
        """Return the maximum total degree *N* represented in *poly_G*."""
        return self.dynamics.degree

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""
        return self.dynamics.ndof

    @property
    def poly_elim(self) -> list[np.ndarray]:
        """Return the polynomial elimination blocks."""
        return self.dynamics.poly_elim

    @property
    def name(self) -> str:
        """Return the name of the Lie generating function."""
        return self.dynamics.name

    def __setstate__(self, state):
        """Restore the LieGeneratingFunction instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of poly_G, poly_elim, degree, ndof, and name.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the LieGeneratingFunction.
        """
        super().__setstate__(state)
        self._setup_services(_LieGeneratingFunctionServices.default(self))

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        """Load a LieGeneratingFunction from a file (new instance).
        
        Parameters
        ----------
        filepath : str or Path
            The path to the file to load the LieGeneratingFunction from.
        **kwargs
            Additional keyword arguments for the load operation.
        """
        return cls._load_with_services(
            filepath, 
            _LieGeneratingFunctionPersistenceService(),
            _LieGeneratingFunctionServices.default, 
            **kwargs
        )
