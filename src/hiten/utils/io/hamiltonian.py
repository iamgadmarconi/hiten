"""Input/output utilities for Hamiltonian data.

This module provides functions for serializing and deserializing Hamiltonian
objects using pickle, which preserves object relationships automatically.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pickle

from hiten.utils.io.common import _ensure_dir

if TYPE_CHECKING:
    from hiten.system.hamiltonian import Hamiltonian
    from hiten.system.hamiltonian import LieGeneratingFunction


def save_hamiltonian(ham: "Hamiltonian", path: str | Path, **kwargs) -> None:
    """Serialize a Hamiltonian object to a pickle file.

    Parameters
    ----------
    ham : :class:`~hiten.system.hamiltonian.Hamiltonian`
        The Hamiltonian object to serialize.
    path : str or pathlib.Path
        File path where to save the Hamiltonian data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
        
    Examples
    --------
    >>> from hiten.system.hamiltonian import Hamiltonian
    >>> import numpy as np
    >>> ham = Hamiltonian([np.array([1.0, 2.0])], degree=2, ndof=3, name="test")
    >>> save_hamiltonian(ham, "my_hamiltonian.pkl")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(ham, f)


def load_hamiltonian(path: str | Path, **kwargs) -> "Hamiltonian":
    """Load a Hamiltonian object from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the Hamiltonian data.
    **kwargs
        Additional keyword arguments (currently unused).
        
    Returns
    -------
    :class:`~hiten.system.hamiltonian.Hamiltonian`
        The reconstructed Hamiltonian object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Examples
    --------
    >>> ham = load_hamiltonian("my_hamiltonian.pkl")
    >>> print(f"Loaded Hamiltonian: {ham.name}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


def save_lie_generating_function(lgf: "LieGeneratingFunction", path: str | Path, **kwargs) -> None:
    """Serialize a LieGeneratingFunction object to a pickle file.

    Parameters
    ----------
    lgf : :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
        The LieGeneratingFunction object to serialize.
    path : str or pathlib.Path
        File path where to save the LieGeneratingFunction data.
    **kwargs
        Additional arguments (ignored for pickle serialization).
        
    Examples
    --------
    >>> from hiten.system.hamiltonian import LieGeneratingFunction
    >>> import numpy as np
    >>> lgf = LieGeneratingFunction([np.array([1.0, 2.0])], [np.array([0.5])], degree=2, ndof=3, name="test")
    >>> save_lie_generating_function(lgf, "my_lgf.pkl")
    """
    path = Path(path)
    _ensure_dir(path.parent)

    with open(path, "wb") as f:
        pickle.dump(lgf, f)


def load_lie_generating_function(path: str | Path, **kwargs) -> "LieGeneratingFunction":
    """Load a LieGeneratingFunction object from a pickle file.
    
    Parameters
    ----------
    path : str or pathlib.Path
        File path containing the LieGeneratingFunction data.
    **kwargs
        Additional keyword arguments (currently unused).
        
    Returns
    -------
    :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
        The reconstructed LieGeneratingFunction object.
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
        
    Examples
    --------
    >>> lgf = load_lie_generating_function("my_lgf.pkl")
    >>> print(f"Loaded LieGeneratingFunction: {lgf.name}")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    
    with open(path, "rb") as f:
        return pickle.load(f)


def load_hamiltonian_inplace(obj: "Hamiltonian", path: str | Path) -> None:
    """Populate an existing Hamiltonian object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.hamiltonian.Hamiltonian`
        The Hamiltonian object to populate.
    path : str or pathlib.Path
        File path containing the Hamiltonian data.
    """
    tmp = load_hamiltonian(path)
    obj.__dict__.update(tmp.__dict__)


def load_lie_generating_function_inplace(obj: "LieGeneratingFunction", path: str | Path) -> None:
    """Populate an existing LieGeneratingFunction object from a pickle file.

    Parameters
    ----------
    obj : :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
        The LieGeneratingFunction object to populate.
    path : str or pathlib.Path
        File path containing the LieGeneratingFunction data.
    """
    tmp = load_lie_generating_function(path)
    obj.__dict__.update(tmp.__dict__)