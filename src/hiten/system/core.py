"""Abstract base class for Hiten classes.

This module provides the abstract base class for all Hiten classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class _HitenBase(ABC):
    """Abstract base class for Hiten classes.
    """

    def __init__(self):
        self._cache = {}

    def cache_get(self, key: Any, default: Any = None) -> Any:
        """Get item from cache.

        Parameters
        ----------
        key : Any
            The cache key.
        default : Any, optional
            The default value to return if the key is not found.
            
        Returns
        -------
        Any
            The cached value or the default value if the key is not found.
        """
        return self._cache.get(key, default)
    
    def cache_set(self, key: Any, value: Any) -> Any:
        """Set item in cache.

        Parameters
        ----------
        key : Any
            The cache key.
        value : Any
            The value to cache.
            
        Returns
        -------
        Any
            The cached value.
        """
        self._cache[key] = value
        return value
    
    def cache_clear(self) -> None:
        """Clear cache.

        This method resets all cached properties to None, forcing them to be
        recomputed on next access.
        """
        self._cache.clear()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __getstate__(self):
        """Get state for pickling.
        """
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        """Set state after unpickling.
        """
        self.__dict__.update(state)
        if not hasattr(self, "_cache") or self._cache is None:
            self._cache = {}

    @abstractmethod
    def save(self, file_path: str | Path, **kwargs) -> None:
        """Save the object to a file.

        Parameters
        ----------
        file_path : str or Path
            The path to the file to save the object to.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, file_path: str | Path, **kwargs) -> "_HitenBase":
        """Load the object from a file.
        
        Parameters
        ----------
        file_path : str or Path
            The path to the file to load the object from.
        **kwargs
            Additional keyword arguments passed to the load method.
            
        Returns
        -------
        :class:`~hiten.system.base._HitenBase`
            The loaded object.
        """
        ...

    def to_csv(self, file_path: str | Path, **kwargs) -> None:
        """Save the object to a CSV file.

        Parameters
        ----------
        file_path : str or Path
            The path to the file to save the object to.
        **kwargs
            Additional keyword arguments passed to the save method.
        """
        ...

    def to_df(cls, **kwargs) -> pd.DataFrame:
        """Convert the object to a pandas DataFrame.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the to_df method.
            
        Returns
        -------
        pandas.DataFrame
            The converted object.
        """
        ...
