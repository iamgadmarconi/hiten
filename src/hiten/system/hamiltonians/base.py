from pathlib import Path
from typing import Callable, Dict, Tuple, Union

import numpy as np
import sympy as sp

from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.polynomial.conversion import poly2sympy
from hiten.algorithms.polynomial.operations import _polynomial_evaluate


class Hamiltonian:
    """Abstract container for a specific polynomial Hamiltonian representation.
    """

    def __init__(self, poly_H: list[np.ndarray], degree: int, ndof: int=3, name: str = "Hamiltonian"):
        if degree <= 0:
            raise ValueError("degree must be a positive integer")

        self._poly_H: list[np.ndarray] = poly_H
        self._degree: int = degree
        self._ndof: int = ndof
        self._psi, self._clmo = _init_index_tables(degree)
        self._encode_dict_list = _create_encode_dict_from_clmo(self._clmo)
        self._name: str = name
        self._hamsys = self._build_hamsys()

    @property
    def name(self) -> str:
        return self._name

    @property
    def poly_H(self) -> list[np.ndarray]:
        """Return the packed coefficient blocks `[H_0, H_2, ..., H_N]`."""
        return self._poly_H

    @property
    def degree(self) -> int:
        """Return the maximum total degree *N* represented in *poly_H*."""
        return self._degree

    @property
    def ndof(self) -> int:
        """Return the number of degrees of freedom."""
        return self._ndof

    def __len__(self) -> int:
        return len(self._poly_H)

    def __getitem__(self, key):
        return self._poly_H[key]

    def __call__(self, coords: np.ndarray) -> float:
        """Evaluate the Hamiltonian at the supplied phase-space coordinate.
        """
        return _polynomial_evaluate(self._poly_H, coords, self._clmo)
    
    @property
    def jacobian(self) -> np.ndarray:
        return self._hamsys.jac_H

    @property
    def hamsys(self):
        """Return a runtime :class:`_HamiltonianSystem`, build lazily."""
        if self._hamsys is None:
            self._hamsys = self._build_hamsys()
        return self._hamsys

    def _build_hamsys(self):
        """Sub-classes must convert *poly_H* into a `_HamiltonianSystem`."""
        return create_hamiltonian_system(self._poly_H, self._degree, self._psi, self._clmo, self._encode_dict_list, self._ndof, self.name)

    @classmethod
    def from_state(cls, other: "Hamiltonian", **kwargs) -> "Hamiltonian":
        """Create *cls* from *other* by applying the appropriate transform."""
        if other.name == cls.name:
            return cls(other.poly_H, other.degree, other._ndof)

        key = (other.name, cls.name)
        try:
            converter, required_context, default_params = _CONVERSION_REGISTRY[key]
        except KeyError as exc:
            raise NotImplementedError(
                f"No conversion path registered from '{other.name}' to '{cls.name}'."
            ) from exc

        # Validate required context
        missing = [key for key in required_context if key not in kwargs]
        if missing:
            raise ValueError(f"Missing required context for conversion {other.name} -> {cls.name}: {missing}")

        # Merge defaults with user-provided parameters
        final_kwargs = {**default_params, **kwargs}
        result = converter(other, **final_kwargs)
        
        # Handle tuple return (Hamiltonian, generating_functions)
        return cls._parse_transform(result, kwargs, cls)

    def to_state(self, target_form: Union[type["Hamiltonian"], str], **kwargs) -> "Hamiltonian":
        """Convert *self* into *target_form* via conversion or ``target_cls.from_state``."""
        # Handle string form names
        if isinstance(target_form, str):
            target_name = target_form
            # Create a temporary Hamiltonian class for the target form
            class _Hamiltonian(Hamiltonian):
                name = target_name
        else:
            target_name = target_form.name
            if isinstance(self, target_form):
                return self
        
        key = (self.name, target_name)
        if key in _CONVERSION_REGISTRY:
            converter, required_context, default_params = _CONVERSION_REGISTRY[key]
            
            # Validate required context
            missing = [key for key in required_context if key not in kwargs]
            if missing:
                raise ValueError(f"Missing required context for conversion {self.name} -> {target_name}: {missing}")
            
            # Merge defaults with user-provided parameters
            final_kwargs = {**default_params, **kwargs}
            result = converter(self, **final_kwargs)
            
            return Hamiltonian._parse_transform(result, kwargs, target_name)

        # If no direct conversion, try using from_state
        if isinstance(target_form, type):
            return target_form.from_state(self, **kwargs)
        else:
            raise NotImplementedError(f"No conversion path from {self.name} to {target_name}")
    
    @staticmethod
    def _parse_transform(result, kwargs, target_name):
            if isinstance(result, tuple):
                new_ham, generating_functions = result
                # Store generating functions in pipeline if available
                pipeline = kwargs.get("_pipeline")
                if pipeline is not None:
                    pipeline._store_generating_functions(target_name, generating_functions)
                return new_ham
            else:
                return result

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

    def save(self, filepath: str | Path, **kwargs) -> None:
        """Serialise this Hamiltonian to *filepath* (HDF5)."""
        from hiten.utils.io.hamiltonian import save_hamiltonian

        save_hamiltonian(self, filepath, **kwargs)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        """Load a Hamiltonian from *filepath* and return it.

        The *cls* argument is ignored - the loader determines the correct
        concrete class from the file metadata and returns an instance of that
        class.
        """
        from hiten.utils.io.hamiltonian import load_hamiltonian

        return load_hamiltonian(filepath, **kwargs)


class LieGeneratingFunction:

    def __init__(self, poly_G: list[np.ndarray], poly_elim: list[np.ndarray], degree: int, ndof: int=3, name: str = None):
        self._poly_G: list[np.ndarray] = poly_G
        self._poly_elim: list[np.ndarray] = poly_elim
        self._degree: int = degree
        self._ndof: int = ndof
        self._name: str = name
        self._psi, self._clmo = _init_index_tables(degree)
        self._encode_dict_list = _create_encode_dict_from_clmo(self._clmo)

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


# Mapping: (src_name, dst_name) -> (converter_func, required_context, default_params)
# Converter functions can return either Hamiltonian or (Hamiltonian, LieGeneratingFunction)
_CONVERSION_REGISTRY: Dict[Tuple[str, str], Tuple[Callable[..., "Hamiltonian | tuple[Hamiltonian, LieGeneratingFunction]"], list, dict]] = {}



