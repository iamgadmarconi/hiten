from typing import Callable, Dict, Tuple

import numpy as np
import sympy as sp

from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _init_index_tables)
from hiten.algorithms.polynomial.conversion import poly2sympy
from hiten.algorithms.polynomial.operations import (_polynomial_evaluate,
                                                    _polynomial_jacobian)


class Hamiltonian:
    """Abstract container for a specific polynomial Hamiltonian representation.
    """

    name: str = "abstract"

    def __init__(self, poly_H: list[np.ndarray], degree: int, ndof: int=3):
        if degree <= 0:
            raise ValueError("degree must be a positive integer")

        self._poly_H: list[np.ndarray] = poly_H
        self._degree: int = degree
        self._ndof: int = ndof
        self._psi, self._clmo = _init_index_tables(degree)
        self._encode_dict_list = _create_encode_dict_from_clmo(self._clmo)

        self._hamsys = None

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
        return _polynomial_jacobian(self._poly_H, self._degree, self._psi, self._clmo, self._encode_dict_list)

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
            return cls(other.poly_H, other.max_degree, other._ndof)

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
        return converter(other, **final_kwargs)

    def to_state(self, target_cls: type["Hamiltonian"], **kwargs) -> "Hamiltonian":
        """Convert *self* into *target_cls* via ``target_cls.from_state``."""
        if isinstance(self, target_cls):
            return self

        key = (self.name, target_cls.name)
        if key in _CONVERSION_REGISTRY:
            converter, required_context, default_params = _CONVERSION_REGISTRY[key]
            
            # Validate required context
            missing = [key for key in required_context if key not in kwargs]
            if missing:
                raise ValueError(f"Missing required context for conversion {self.name} -> {target_cls.name}: {missing}")
            
            # Merge defaults with user-provided parameters
            final_kwargs = {**default_params, **kwargs}
            return converter(self, **final_kwargs)

        return target_cls.from_state(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', max_degree={self.max_degree}, "
            f"blocks={len(self)})"
        )
    
    def __str__(self) -> str:
        q1, q2, q3, p1, p2, p3 = sp.symbols("q1 q2 q3 p1 p2 p3")
        return poly2sympy(self._poly_H, [q1, q2, q3, p1, p2, p3], self._psi, self._clmo)

    def __bool__(self):
        return bool(self._poly_H)
    

class LieGeneratingFunction:

    name: str = "lie_generating_function"

    def __init__(self, poly_G: list[np.ndarray], poly_elim: list[np.ndarray], degree: int, ndof: int=3):
        self._poly_G: list[np.ndarray] = poly_G
        self._poly_elim: list[np.ndarray] = poly_elim
        self._degree: int = degree
        self._ndof: int = ndof
        
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


# Mapping: (src_name, dst_name) -> (converter_func, required_context, default_params)
_CONVERSION_REGISTRY: Dict[Tuple[str, str], Tuple[Callable[..., "Hamiltonian"], list, dict]] = {}


def register_conversion(src_name: str, dst: "type[Hamiltonian] | str", 
                       required_context: list = None,
                       default_params: dict = None):
    """Decorator to register a *src* -> *dst* conversion.

    Parameters
    ----------
    src_name : str
        The ``Hamiltonian.name`` of the *source* representation.
    dst : type[Hamiltonian] | str
        Either the **class** for the destination representation *or* its
        ``name`` string.  Allowing a string lets you register conversions
        before defining dedicated subclasses.
    required_context : list, optional
        List of required context keys (e.g., ["point"] for LibrationPoint).
        These must be provided in the conversion call.
    default_params : dict, optional
        Default parameter values (e.g., {"tol": 1e-12}).

    Example
    -------
    >>> @register_conversion("physical", "real_modal", 
    ...                     required_context=["point"],
    ...                     default_params={"tol": 1e-12})
    ... def _physical_to_real(src: Hamiltonian, **kwargs) -> Hamiltonian:
    ...     point = kwargs["point"]
    ...     tol = kwargs.get("tol", 1e-12)
    ...     ...
    """

    dst_name: str
    if isinstance(dst, str):
        dst_name = dst
    else:
        dst_name = dst.name

    def _decorator(func):
        _CONVERSION_REGISTRY[(src_name, dst_name)] = (
            func, 
            required_context or [], 
            default_params or {}
        )
        return func

    return _decorator

