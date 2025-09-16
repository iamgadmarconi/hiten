"""Base classes for Hamiltonian representations in the CR3BP.

This module provides the fundamental classes for representing and manipulating
Hamiltonian functions in the circular restricted three-body problem. It includes
the base Hamiltonian class and Lie generating function class for canonical
transformations.

Notes
-----
The Hamiltonian class supports conversion between different representations
through a registry-based system. All polynomial coefficients are stored as
NumPy arrays for efficient computation.
"""

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
    """
    Abstract container for a specific polynomial Hamiltonian representation.
    
    This class provides the base functionality for representing and manipulating
    polynomial Hamiltonian functions in the circular restricted three-body problem.
    It supports evaluation, conversion between different representations, and
    serialization to/from files.
    
    Parameters
    ----------
    poly_H : list[np.ndarray]
        Packed coefficient blocks [H_0, H_2, ..., H_N] representing the Hamiltonian
    degree : int
        Maximum total degree N represented in poly_H
    ndof : int, optional
        Number of degrees of freedom, by default 3
    name : str, optional
        Name of the Hamiltonian representation, by default "Hamiltonian"
        
    Attributes
    ----------
    poly_H : list[np.ndarray]
        Packed coefficient blocks [H_0, H_2, ..., H_N]
    degree : int
        Maximum total degree N represented in poly_H
    ndof : int
        Number of degrees of freedom
    name : str
        Name of the Hamiltonian representation
    jacobian : np.ndarray
        Jacobian matrix of the Hamiltonian
    hamsys : :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`
        Runtime Hamiltonian system for evaluation
        
    Notes
    -----
    The Hamiltonian is represented as a polynomial in canonical coordinates
    (q1, q2, q3, p1, p2, p3) with coefficients stored in packed format for
    efficient computation.
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
        """
        Evaluate the Hamiltonian at the supplied phase-space coordinate.
        
        Parameters
        ----------
        coords : np.ndarray
            Phase-space coordinates (q1, q2, q3, p1, p2, p3)
            
        Returns
        -------
        float
            Value of the Hamiltonian at the given coordinates
            
        Notes
        -----
        The coordinates should be in the canonical coordinate system
        (q1, q2, q3, p1, p2, p3) for proper evaluation.
        """
        return _polynomial_evaluate(self._poly_H, coords, self._clmo)
    
    @property
    def jacobian(self) -> np.ndarray:
        return self._hamsys.jac_H

    @property
    def hamsys(self):
        """Return a runtime :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`, build lazily."""
        if self._hamsys is None:
            self._hamsys = self._build_hamsys()
        return self._hamsys

    def _build_hamsys(self):
        """Sub-classes must convert *poly_H* into a :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`."""
        return create_hamiltonian_system(self._poly_H, self._degree, self._psi, self._clmo, self._encode_dict_list, self._ndof, self.name)

    @classmethod
    def from_state(cls, other: "Hamiltonian", **kwargs) -> "Hamiltonian":
        """
        Create a new Hamiltonian from another by applying the appropriate transform.
        
        Parameters
        ----------
        other : :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            Source Hamiltonian to transform from
        **kwargs
            Additional parameters required for the transformation
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            New Hamiltonian in the target representation
            
        Raises
        ------
        NotImplementedError
            If no conversion path is registered from other.name to cls.name
        ValueError
            If required context parameters are missing
            
        Notes
        -----
        This method uses the conversion registry to find and apply the
        appropriate transformation between different Hamiltonian representations.
        """
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
        """
        Convert this Hamiltonian into the target form.
        
        Parameters
        ----------
        target_form : type[:class:`~hiten.system.hamiltonians.base.Hamiltonian`] or str
            Target Hamiltonian class or name to convert to
        **kwargs
            Additional parameters required for the transformation
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            New Hamiltonian in the target representation
            
        Raises
        ------
        NotImplementedError
            If no conversion path is available to the target form
        ValueError
            If required context parameters are missing
            
        Notes
        -----
        This method first tries to find a direct conversion path in the
        registry. If none is found, it attempts to use the target class's
        from_state method.
        """
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
        """
        Parse the result of a Hamiltonian transformation.
        
        Parameters
        ----------
        result : :class:`~hiten.system.hamiltonians.base.Hamiltonian` or tuple
            Result of the transformation, either a Hamiltonian or a tuple
            containing (Hamiltonian, LieGeneratingFunction)
        kwargs : dict
            Keyword arguments passed to the transformation
        target_name : str
            Name of the target Hamiltonian representation
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The transformed Hamiltonian
            
        Notes
        -----
        If the result is a tuple containing generating functions, they are
        stored in the pipeline if available for later use.
        """
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
        """
        Serialize this Hamiltonian to a file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the Hamiltonian to (HDF5 format)
        **kwargs
            Additional parameters for serialization
            
        Notes
        -----
        The Hamiltonian is saved in HDF5 format for efficient storage and
        retrieval. The file includes all polynomial coefficients and metadata.
        """
        from hiten.utils.io.hamiltonian import save_hamiltonian

        save_hamiltonian(self, filepath, **kwargs)

    @classmethod
    def load(cls, filepath: str | Path, **kwargs):
        """
        Load a Hamiltonian from a file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to load the Hamiltonian from (HDF5 format)
        **kwargs
            Additional parameters for loading
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            Loaded Hamiltonian instance
            
        Notes
        -----
        The cls argument is ignored - the loader determines the correct
        concrete class from the file metadata and returns an instance of that
        class. The file must be in HDF5 format as saved by the save method.
        """
        from hiten.utils.io.hamiltonian import load_hamiltonian

        return load_hamiltonian(filepath, **kwargs)


class LieGeneratingFunction:
    """
    Class for Lie generating functions in canonical transformations.
    
    This class represents a Lie generating function G(q, p) that generates
    canonical transformations preserving the Hamiltonian structure of the
    system. It is used in normal form calculations and center manifold
    reductions.
    
    Parameters
    ----------
    poly_G : list[np.ndarray]
        Packed coefficient blocks [G_0, G_2, ..., G_N] representing the generating function
    poly_elim : list[np.ndarray]
        Packed coefficient blocks for elimination terms
    degree : int
        Maximum total degree N represented in poly_G
    ndof : int, optional
        Number of degrees of freedom, by default 3
    name : str, optional
        Name of the generating function, by default None
        
    Attributes
    ----------
    poly_G : list[np.ndarray]
        Packed coefficient blocks [G_0, G_2, ..., G_N]
    poly_elim : list[np.ndarray]
        Packed coefficient blocks for elimination terms
    degree : int
        Maximum total degree N represented in poly_G
    ndof : int
        Number of degrees of freedom
    name : str
        Name of the generating function
        
    Notes
    -----
    Lie generating functions are essential for canonical transformations
    that preserve the Hamiltonian structure. They are used in normal form
    calculations to simplify the Hamiltonian representation.
    """

    def __init__(self, poly_G: list[np.ndarray], poly_elim: list[np.ndarray], degree: int, ndof: int=3, name: str = None):
        """
        Initialize a Lie generating function.
        
        Parameters
        ----------
        poly_G : list[np.ndarray]
            Packed coefficient blocks [G_0, G_2, ..., G_N] representing the generating function
        poly_elim : list[np.ndarray]
            Packed coefficient blocks for elimination terms
        degree : int
            Maximum total degree N represented in poly_G
        ndof : int, optional
            Number of degrees of freedom, by default 3
        name : str, optional
            Name of the generating function, by default None
            
        Notes
        -----
        The polynomial coefficients are stored in packed format for efficient
        computation. The degree parameter determines the maximum order of
        terms in the polynomial representation.
        """
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



