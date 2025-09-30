"""Base serialization utilities for handling Numba objects and other problematic types.

This module provides a base class with serialization methods that can be mixed into
other classes to handle pickling of objects containing Numba-compiled functions,
typed containers, and other non-serializable objects.
"""

from __future__ import annotations

from typing import Any


class _SerializeBase:
    """Mixin class providing serialization utilities for objects with Numba dependencies.
    
    This class provides methods to handle serialization of objects that may contain
    Numba-compiled functions, typed containers, or other objects that cannot be pickled
    directly. It converts problematic objects to serializable formats during pickling
    and handles reconstruction during unpickling.
    
    The main methods are:
    - `_make_serializable()`: Converts objects to pickle-safe formats
    - `__getstate__()`: Handles object state serialization
    - `__setstate__()`: Handles object state deserialization
    
    Notes
    -----
    This class is designed to be used as a mixin with other base classes. It handles
    the common case where objects contain Numba-typed containers (like `numba.typed.List`)
    that cannot be pickled directly due to internal `_nrt_python._MemInfo` objects.
    
    Examples
    --------
    >>> class MyClass(_SerializeBase):
    ...     def __init__(self):
    ...         from numba.typed import List
    ...         self.data = List()
    ...         self.data.append([1, 2, 3])
    >>> 
    >>> obj = MyClass()
    >>> # This will now work without pickling errors
    >>> import pickle
    >>> pickled = pickle.dumps(obj)
    >>> restored = pickle.loads(pickled)
    """
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert an object to a serializable format, handling Numba objects and other problematic types.
        
        This method recursively processes objects to convert them into formats that can be
        safely pickled. It specifically handles:
        
        - Numba typed containers (converts to regular Python containers)
        - Numba compiled functions (returns None, can be reconstructed)
        - Numba memory info objects (returns None)
        - Nested containers (recursively processes contents)
        - Basic serializable types (passes through unchanged)
        
        Parameters
        ----------
        obj : Any
            The object to convert to serializable format
            
        Returns
        -------
        Any
            The object in a serializable format, or a placeholder if conversion is not possible.
            Returns None for objects that cannot be serialized but can be reconstructed.
        """
        if obj is None:
            return None
            
        # Handle Numba typed containers - convert to regular Python containers
        try:
            from numba.typed import List
            if isinstance(obj, List):
                # Convert numba.typed.List to regular Python list
                return [self._make_serializable(item) for item in obj]
        except ImportError:
            pass
            
        # Handle Numba objects - these cannot be pickled
        if hasattr(obj, '__module__') and obj.__module__ and 'numba' in obj.__module__:
            return None  # Return None for Numba objects, they can be reconstructed
            
        # Handle Numba compiled functions
        if hasattr(obj, '__name__') and hasattr(obj, '__module__') and obj.__module__ and 'numba' in obj.__module__:
            return None  # Return None for Numba functions, they can be reconstructed
            
        # Handle Numba memory info objects
        if hasattr(obj, '__class__') and 'nrt_python' in str(type(obj)):
            return None  # Return None for Numba memory objects
            
        # Handle dictionaries that might contain Numba objects
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
            
        # Handle lists that might contain Numba objects
        if isinstance(obj, (list, tuple)):
            if isinstance(obj, tuple):
                return tuple(self._make_serializable(item) for item in obj)
            else:
                return [self._make_serializable(item) for item in obj]
                
        # Handle basic serializable types without testing pickle
        if isinstance(obj, (str, int, float, bool, bytes, type(None))):
            return obj
            
        # Handle numpy arrays and basic numpy types
        try:
            import numpy as np
            if isinstance(obj, (np.ndarray, np.integer, np.floating, np.bool_, np.complexfloating)):
                return obj
        except ImportError:
            pass
            
        # Handle objects with __dict__ - be more conservative
        if hasattr(obj, '__dict__'):
            # Only exclude objects that are clearly problematic for serialization
            if hasattr(obj, '__class__'):
                module_name = getattr(obj.__class__, '__module__', '')
                
                # Only skip objects that are clearly Numba-related or have known issues
                if ('numba' in module_name.lower() or 
                    'nrt_python' in str(type(obj))):
                    return None
                    
                # For other objects with __dict__, try to return as-is
                # Don't test with pickle.dumps to avoid recursion
                return obj
        
        # For other types, return as-is without testing pickle
        # This avoids the recursion issue while preserving functionality
        return obj

    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling, handling Numba objects properly.
        
        This method creates a dictionary containing the object's state that can be
        safely pickled. It processes all attributes through `_make_serializable()`
        to handle Numba objects and other problematic types.
        
        Returns
        -------
        dict[str, Any]
            Dictionary containing the serializable state of the object
        """
        state = self.__dict__.copy()
        
        # Convert Numba objects to serializable format
        for key, value in state.items():
            state[key] = self._make_serializable(value)
        
        return state
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state after unpickling, reconstructing Numba objects if needed.
        
        This method restores the object's state from a dictionary created by
        `__getstate__()`. It handles the reconstruction of objects that were
        converted during serialization.
        
        Parameters
        ----------
        state : dict[str, Any]
            Dictionary containing the serializable state of the object
        """
        self.__dict__.update(state)
        
        # Reset compiled RHS to None so it gets recompiled when needed
        if hasattr(self, '_rhs_compiled'):
            self._rhs_compiled = None
