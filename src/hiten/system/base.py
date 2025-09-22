"""High-level abstractions for the Circular Restricted Three-Body Problem (CR3BP).

This module bundles the physical information of a binary system, computes the
mass parameter mu, instantiates the underlying vector field via
:func:`~hiten.algorithms.dynamics.rtbp.rtbp_dynsys`, and pre-computes the five
classical Lagrange (libration) points.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import rtbp_dynsys, variational_dynsys
from hiten.algorithms.utils.coordinates import _get_mass_parameter
from hiten.system.body import Body
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import L1Point, L2Point, L3Point
from hiten.system.libration.triangular import L4Point, L5Point
from hiten.utils.constants import Constants
from hiten.utils.io.system import save_system, load_system
from hiten.utils.log_config import logger


class _HitenBase(ABC):
    """Abstract base class for Hiten classes.
    """

    def __init__(self):
        self._cache = {}

    def cache_get(self, key: any, default: any = None) -> any:
        """Get item from cache.

        Parameters
        ----------
        key : any
            The cache key.
        default : any, optional
            The default value to return if the key is not found.
            
        Returns
        -------
        any
            The cached value or the default value if the key is not found.
        """
        return self._cache.get(key, default)
    
    def cache_set(self, key: any, value: any) -> any:
        """Set item in cache.

        Parameters
        ----------
        key : any
            The cache key.
        value : any
            The value to cache.
            
        Returns
        -------
        any
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


class System(_HitenBase):
    """
    Lightweight wrapper around the CR3BP dynamical system.

    The class stores the physical properties of the primaries, computes the
    dimensionless mass parameter mu = m2 / (m1 + m2), instantiates
    the CR3BP vector field through :func:`~hiten.algorithms.dynamics.rtbp.rtbp_dynsys`,
    and caches the five Lagrange points.

    Parameters
    ----------
    primary : :class:`~hiten.system.body.Body`
        Primary gravitating body.
    secondary : :class:`~hiten.system.body.Body`
        Secondary gravitating body.
    distance : float
        Characteristic separation between the bodies in km.

    Attributes
    ----------
    mu : float
        Mass parameter mu (dimensionless).
    libration_points : dict[int, LibrationPoint]
        Mapping from integer identifiers {1,...,5} to the corresponding
        libration point objects.
    dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
        Underlying vector field instance compatible with the integrators
        defined in :mod:`~hiten.algorithms.integrators`.
    var_dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
        Underlying variational equations system.

    Notes
    -----
    The heavy computations reside in the dynamical system and individual
    libration point classes; this wrapper simply orchestrates them.
    """
    def __init__(self, primary: Body, secondary: Body, distance: float):
        logger.info(f"Initializing System with primary='{primary.name}', secondary='{secondary.name}', distance={distance:.4e}")
        
        self._primary = primary
        self._secondary = secondary
        self._distance = distance
        self._mu: float = _get_mass_parameter(primary.mass, secondary.mass)
        self._dynsys = rtbp_dynsys(self.mu, name=f"RTBP_{self.primary.name}_{self.secondary.name}")
        self._var_dynsys = variational_dynsys(self.mu, name=f"VarEq_{self.primary.name}_{self.secondary.name}")

        self._cache = {
            1: L1Point(self),
            2: L2Point(self),
            3: L3Point(self),
            4: L4Point(self),
            5: L5Point(self)
        }
        logger.info(f"Computed {len(self.libration_points)} Libration points.")

    def __str__(self) -> str:
        return f"{self.secondary.name} orbiting {self.primary.name}"

    def __repr__(self) -> str:
        return f"System(primary={self.primary!r}, secondary={self.secondary!r}, distance={self.distance}), mu={self.mu:.6e}"

    @property
    def primary(self) -> Body:
        """Primary gravitating body.
        
        Returns
        -------
        :class:`~hiten.system.body.Body`
            The primary gravitating body.
        """
        return self._primary

    @property
    def secondary(self) -> Body:
        """Secondary gravitating body.
        
        Returns
        -------
        :class:`~hiten.system.body.Body`
            The secondary gravitating body.
        """
        return self._secondary

    @property
    def distance(self) -> float:
        """Characteristic separation between the bodies.
        
        Returns
        -------
        float
            The characteristic separation between the bodies in km.
        """
        return self._distance

    @property
    def mu(self) -> float:
        """Mass parameter mu.
        
        Returns
        -------
        float
            The mass parameter mu = m2 / (m1 + m2) (dimensionless).
        """
        return self._mu

    @property
    def libration_points(self) -> Dict[int, LibrationPoint]:
        """Mapping from integer identifiers {1,...,5} to libration point objects.
        
        Returns
        -------
        dict[int, LibrationPoint]
            Dictionary mapping integer identifiers {1,...,5} to libration point objects.
        """
        return self._cache
        
    @property
    def dynsys(self):
        """Underlying vector field instance.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying vector field instance.
        """
        return self._dynsys

    @property
    def var_dynsys(self):
        """Underlying variational equations system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying variational equations system.
        """
        return self._var_dynsys

    def get_libration_point(self, index: int) -> LibrationPoint:
        """
        Access a pre-computed libration point.

        Parameters
        ----------
        index : int
            Identifier of the desired point in {1, 2, 3, 4, 5}.

        Returns
        -------
        :class:`~hiten.system.libration.base.LibrationPoint`
            Requested libration point instance.

        Raises
        ------
        ValueError
            If index is not in the valid range.

        Examples
        --------
        >>> sys = System(primary, secondary, distance)
        >>> L1 = sys.get_libration_point(1)
        """
        point: Optional[LibrationPoint] = self.cache_get(index)
        if point is None:
            raise ValueError(f"Invalid Libration point index: {index}. Must be 1, 2, 3, 4, or 5.")
        return point

    def propagate(
        self,
        initial_conditions: Sequence[float],
        tf: float = 2 * np.pi,
        *,
        steps: int = 1000,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order: int = 8,
        **kwargs
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Propagate arbitrary initial conditions in the CR3BP.

        This helper is a thin wrapper around
        :func:`~hiten.algorithms.dynamics.rtbp._propagate_dynsys` that avoids
        the need to instantiate a :class:`~hiten.system.orbits.base.PeriodicOrbit`.

        Parameters
        ----------
        initial_conditions : Sequence[float]
            Six-element state vector [x, y, z, vx, vy, vz] expressed in
            canonical CR3BP units (nondimensional).
        tf : float, default 2*pi
            Final time for integration in nondimensional units.
        steps : int, default 1000
            Number of output nodes in the returned trajectory.
        method : {"fixed", "adaptive", "symplectic"}, default "adaptive"
            Integration backend to employ (Hiten integrators).
        order : int, default 8
            Formal order of the integrator when applicable.
        **kwargs
            Additional keyword arguments passed to the integrator.

        Returns
        -------
        tuple (times, states)
            times : numpy.ndarray, shape (steps,)
                Array holding the sampling instants in nondimensional units.
            states : numpy.ndarray, shape (steps, 6)
                Array with the propagated trajectory [x, y, z, vx, vy, vz].
        """

        forward = kwargs.get("forward", 1)

        sol = _propagate_dynsys(
            dynsys=self._dynsys,
            state0=initial_conditions,
            t0=0.0,
            tf=tf,
            forward=forward,
            steps=steps,
            method=method,
            order=order,
        )

        return sol.times, sol.states

    @classmethod
    def from_bodies(cls, primary_name: str, secondary_name: str) -> "System":
        """
        Factory method to build a :class:`~hiten.system.base.System` directly from body names.

        This helper retrieves the masses, radii and characteristic orbital
        distance of the selected primary/secondary pair from
        :class:`~hiten.utils.constants.Constants` and instantiates the
        corresponding :class:`~hiten.system.body.Body` objects before finally returning the
        fully-initialised :class:`~hiten.system.base.System` instance.

        Parameters
        ----------
        primary_name : str
            Name of the primary body (case-insensitive, e.g. "earth").
        secondary_name : str
            Name of the secondary body orbiting the primary (e.g. "moon").

        Returns
        -------
        :class:`~hiten.system.base.System`
            Newly created CR3BP system.
            
        Raises
        ------
        ValueError
            If the body names are not found in the constants database.
        """
        # Normalise the identifiers so that the lookup in *Constants* is
        # case-insensitive while preserving the original capitalisation for
        # display purposes.
        p_key = primary_name.lower()
        s_key = secondary_name.lower()

        # Retrieve physical parameters from the constants catalogue
        try:
            p_mass = Constants.get_mass(p_key)
            p_radius = Constants.get_radius(p_key)
            s_mass = Constants.get_mass(s_key)
            s_radius = Constants.get_radius(s_key)
            distance = Constants.get_orbital_distance(p_key, s_key)
        except KeyError as exc:
            # Re-raise with a clearer error message for the end-user
            raise ValueError(
                f"Unknown body or orbital distance for pair '{primary_name}', '{secondary_name}'."
            ) from exc

        # Instantiate the bodies - the secondary orbits the primary.
        primary = Body(primary_name.capitalize(), p_mass, p_radius)
        secondary = Body(secondary_name.capitalize(), s_mass, s_radius, _parent_input=primary)

        # Create and return the CR3BP system
        return cls(primary, secondary, distance)

    @classmethod
    def from_mu(cls, mu: float) -> "System":
        """Factory method to build a :class:`~hiten.system.base.System` 
        directly from the mass parameter.
        
        Parameters
        ----------
        mu : float
            Mass parameter mu = m2 / (m1 + m2) (dimensionless).
            
        Returns
        -------
        :class:`~hiten.system.base.System`
            Newly created CR3BP system with the specified mass parameter.
        """
        primary = Body("Primary", 1-mu, 1.0e-3)
        secondary = Body("Secondary", mu, 1.0e-3)
        distance = 1.0
        return cls(primary, secondary, distance)

    def __getstate__(self):
        """Custom state extractor to enable pickling.

        The underlying dynamical system instance stored in _dynsys often
        contains numba-compiled objects that cannot be serialised. We exclude
        it from the pickled representation and recreate it automatically when
        the object is re-loaded.
        
        Returns
        -------
        dict
            Dictionary containing the serializable state of the System.
        """
        state = self.__dict__.copy()
        # Remove the compiled dynamical system before pickling
        if "_dynsys" in state:
            state["_dynsys"] = None
        if "_var_dynsys" in state:
            state["_var_dynsys"] = None
        return state

    def __setstate__(self, state):
        """Restore the System instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of mu and the names of the primary and
        secondary bodies.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the System.
        """
        from hiten.algorithms.dynamics.rtbp import rtbp_dynsys
        from hiten.algorithms.dynamics.rtbp import variational_dynsys
        # Restore the plain attributes
        self.__dict__.update(state)

        if self.__dict__.get("_dynsys") is None:
            self._dynsys = rtbp_dynsys(self.mu, name=self.primary.name + "_" + self.secondary.name)
            self._var_dynsys = variational_dynsys(self.mu, name=self.primary.name + "_" + self.secondary.name)

    def save(self, file_path: str | Path, **kwargs) -> None:
        """Save this System to a file."""
        from hiten.utils.io.system import save_system
        save_system(self, file_path)

    @classmethod
    def load(cls, file_path: str | Path, **kwargs) -> "System":
        """Load a System from a file (new instance)."""
        from hiten.utils.io.system import load_system
        return load_system(file_path)

    def load_inplace(self, file_path: str | Path) -> "System":
        """Load data into this System instance from a file (in place)."""
        from hiten.utils.io.system import load_system_inplace
        load_system_inplace(self, file_path)
        return self