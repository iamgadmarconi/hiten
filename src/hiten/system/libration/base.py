"""Abstract helpers to model Libration points of the Circular Restricted Three-Body Problem (CR3BP).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.dynamics.hamiltonian import _HamiltonianSystem
from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp
from hiten.algorithms.dynamics.utils.energy import (crtbp_energy,
                                                    energy_to_jacobi)
from hiten.algorithms.dynamics.utils.linalg import eigenvalue_decomposition
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.center import CenterManifold
    from hiten.system.orbits.base import PeriodicOrbit

# Constants for stability analysis mode
CONTINUOUS_SYSTEM = 0
DISCRETE_SYSTEM = 1


@dataclass(slots=True)
class LinearData:
    """
    Container with linearised CR3BP invariants.

    Parameters
    ----------
    mu : float
        Mass ratio mu = m2/(m1+m2) of the primaries (dimensionless).
    point : str
        Identifier of the libration point ('L1', 'L2' or 'L3').
    lambda1 : float | None
        Real hyperbolic eigenvalue lambda1 > 0 associated with the
        saddle behaviour along the centre-saddle subspace (nondimensional units).
    omega1 : float
        First elliptic frequency omega1 > 0 of the centre subspace (nondimensional units).
    omega2 : float
        Second elliptic frequency omega2 > 0 of the centre subspace (nondimensional units).
    omega3: float | None
        Vertical frequency omega3 of the centre subspace (nondimensional units).
    C : numpy.ndarray, shape (6, 6)
        Symplectic change-of-basis matrix such that C^(-1)AC is in real
        Jordan canonical form, with A the Jacobian of the vector
        field evaluated at the libration point.
    Cinv : numpy.ndarray, shape (6, 6)
        Precomputed inverse of C.

    Notes
    -----
    The record is immutable thanks to slots=True; all fields are plain
    numpy.ndarray or scalars so the instance can be safely cached
    and shared among different computations.
    """
    mu: float
    point: str        # 'L1', 'L2', 'L3'
    lambda1: float | None
    omega1: float
    omega2: float
    omega3: float | None
    C: np.ndarray     # 6x6 symplectic transform
    Cinv: np.ndarray  # inverse


class LibrationPoint(ABC):
    """
    Abstract base class for Libration points of the CR3BP.

    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        Parent CR3BP model providing the mass ratio mu and utility
        functions.

    Attributes
    ----------
    mu : float
        Mass ratio mu of the primaries (copied from system, dimensionless).
    system : :class:`~hiten.system.base.System`
        Reference to the owner system.
    position : numpy.ndarray, shape (3,)
        Cartesian coordinates in the synodic rotating frame (nondimensional units).
        Evaluated on first access and cached thereafter.
    energy : float
        Dimensionless mechanical energy evaluated via
        :func:`~hiten.algorithms.dynamics.utils.energy.crtbp_energy`.
    jacobi_constant : float
        Jacobi integral CJ = -2E corresponding to energy (dimensionless).
    is_stable : bool
        True if all eigenvalues returned by 
        :meth:`~hiten.system.libration.base.LibrationPoint.analyze_stability` lie
        inside the unit circle (discrete case) or have non-positive real
        part (continuous case).
    eigenvalues : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Arrays of stable, unstable and centre eigenvalues.
    eigenvectors : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Bases of the corresponding invariant subspaces.
    linear_data : :class:`~hiten.system.libration.base.LinearData`
        Record with canonical invariants and symplectic basis returned by the
        normal-form computation.

    Notes
    -----
    The class is abstract. Concrete subclasses must implement:

    - :meth:`~hiten.system.libration.base.LibrationPoint.idx`
    - :meth:`~hiten.system.libration.base.LibrationPoint._calculate_position`
    - :meth:`~hiten.system.libration.base.LibrationPoint._get_linear_data`
    - :meth:`~hiten.system.libration.base.LibrationPoint.normal_form_transform`

    Heavy algebraic objects produced by the centre-manifold normal-form
    procedure are cached inside a dedicated
    :class:`~hiten.system.center.CenterManifold` instance to avoid memory
    bloat.

    Examples
    --------
    >>> from hiten.system.base import System
    >>> sys = System(mu=0.0121505856)   # Earth-Moon system
    >>> L1 = sys.libration_points['L1']
    >>> L1.position
    array([...])
    """
    
    def __init__(self, system: "System"):
        self.system = system
        self.mu = system.mu
        self._position = None
        self._stability_info = None
        self._linear_data = None
        self._energy = None
        self._jacobi_constant = None
        self._cache = {}
        self._cm_registry = {}

        self._dynsys = system.dynsys
        self._var_eq_system = system.var_dynsys
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    @property
    def dynsys(self):
        """Underlying vector field instance.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`
            The dynamical system instance for this libration point.
        """
        return self._dynsys
    
    @property
    def var_eq_system(self):
        """Underlying variational equations system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.base._DynamicalSystem`
            The variational equations system for this libration point.
        """
        return self._var_eq_system

    @property
    @abstractmethod
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (1-5 for L1-L5).
        """
        pass

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the Libration point in the rotating frame.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            3D vector [x, y, z] representing the position in nondimensional units.
        """
        if self._position is None:
            self._position = self._calculate_position()
        return self._position
    
    @property
    def energy(self) -> float:
        """
        Get the energy of the Libration point.
        
        Returns
        -------
        float
            The mechanical energy in nondimensional units.
        """
        if self._energy is None:
            self._energy = self._compute_energy()
        return self._energy
    
    @property
    def jacobi_constant(self) -> float:
        """
        Get the Jacobi constant of the Libration point.
        
        Returns
        -------
        float
            The Jacobi constant in nondimensional units.
        """
        if self._jacobi_constant is None:
            self._jacobi_constant = self._compute_jacobi_constant()
        return self._jacobi_constant
    
    @property
    def is_stable(self) -> bool:
        """
        Check if the Libration point is linearly stable.

        A libration point is considered stable if its linear analysis yields no
        unstable eigenvalues. The check is performed on the continuous-time
        system by default.
        
        Returns
        -------
        bool
            True if the libration point is linearly stable.
        """
        if self._stability_info is None:
            # The default mode for analyze_stability is CONTINUOUS_SYSTEM,
            # which correctly classifies eigenvalues based on their real part
            # for determining stability.
            self.analyze_stability()
        
        unstable_eigenvalues = self._stability_info[1]
        return len(unstable_eigenvalues) == 0

    @property
    def linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LinearData`
            The linear data containing eigenvalues and eigenvectors.
        """
        if self._linear_data is None:
            self._linear_data = self._get_linear_data()
        return self._linear_data

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvalues of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvalues, unstable_eigenvalues, center_eigenvalues)
            Each array contains eigenvalues in nondimensional units.
        """
        if self._stability_info is None:
            self.analyze_stability() # Ensure stability is analyzed
        sn, un, cn, _, _, _ = self._stability_info
        return (sn, un, cn)
    
    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the eigenvectors of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvectors, unstable_eigenvectors, center_eigenvectors)
            Each array contains eigenvectors as column vectors.
        """
        if self._stability_info is None:
            self.analyze_stability() # Ensure stability is analyzed
        _, _, _, Ws, Wu, Wc = self._stability_info
        return (Ws, Wu, Wc)

    def cache_get(self, key) -> any:
        """
        Get item from cache.
        
        Parameters
        ----------
        key : any
            The cache key.
            
        Returns
        -------
        any
            The cached value or None if not found.
        """
        return self._cache.get(key)
    
    def cache_set(self, key, value) -> any:
        """
        Set item in cache and return the value.
        
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
        """
        Clear all cached data, including computed properties.
        
        This method resets all cached properties to None, forcing them to be
        recomputed on next access.
        """
        self._cache.clear()
        self._position = None
        self._stability_info = None
        self._linear_data = None
        self._energy = None
        self._jacobi_constant = None
        logger.debug(f"Cache cleared for {type(self).__name__}")

    def _compute_energy(self) -> float:
        """
        Compute the energy of the Libration point.
        
        Returns
        -------
        float
            The mechanical energy in nondimensional units.
        """
        state = np.concatenate([self.position, [0, 0, 0]])
        return crtbp_energy(state, self.mu)

    def _compute_jacobi_constant(self) -> float:
        """
        Compute the Jacobi constant of the Libration point.
        
        Returns
        -------
        float
            The Jacobi constant in nondimensional units.
        """
        return energy_to_jacobi(self.energy)

    @abstractmethod
    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of the Libration point.
        
        This is an abstract method that must be implemented by subclasses.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            3D vector [x, y, z] representing the position in nondimensional units.
        """
        pass

    @abstractmethod
    def _get_linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        :class:`~hiten.system.libration.base.LinearData`
            The linear data containing eigenvalues and eigenvectors.
        """
        pass

    def analyze_stability(self, discrete: int = CONTINUOUS_SYSTEM, delta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze the stability properties of the Libration point.
        
        Parameters
        ----------
        discrete : int, optional
            Classification mode for eigenvalues:
            - CONTINUOUS_SYSTEM (0): continuous-time system (classify by real part sign)
            - DISCRETE_SYSTEM (1): discrete-time system (classify by magnitude relative to 1)
        delta : float, optional
            Tolerance for classification (dimensionless).
            
        Returns
        -------
        tuple
            (sn, un, cn, Ws, Wu, Wc) containing:
            - sn: stable eigenvalues (nondimensional units)
            - un: unstable eigenvalues (nondimensional units)
            - cn: center eigenvalues (nondimensional units)
            - Ws: eigenvectors spanning stable subspace
            - Wu: eigenvectors spanning unstable subspace
            - Wc: eigenvectors spanning center subspace
        """
        # Check cache first
        cache_key = ('stability_analysis', discrete, delta)
        cached = self.cache_get(cache_key)
        if cached is not None:
            logger.debug(f"Using cached stability analysis for {type(self).__name__}")
            self._stability_info = cached  # Update instance variable for property access
            return cached
        
        mode_str = "Continuous" if discrete == CONTINUOUS_SYSTEM else "Discrete"
        logger.info(f"Analyzing stability for {type(self).__name__} (mu={self.mu}), mode={mode_str}, delta={delta}.")
        pos = self.position
        A = _jacobian_crtbp(pos[0], pos[1], pos[2], self.mu)
        
        logger.debug(f"Jacobian calculated at position {pos}:\n{A}")

        # Perform eigenvalue decomposition and classification
        stability_info = eigenvalue_decomposition(A, discrete, delta)
        
        # Cache and store in instance variable
        self._stability_info = stability_info
        self.cache_set(cache_key, stability_info)
        
        sn, un, cn, _, _, _ = stability_info
        logger.info(f"Stability analysis complete: {len(sn)} stable, {len(un)} unstable, {len(cn)} center eigenvalues.")
        
        return stability_info

    def get_center_manifold(self, degree: int) -> "CenterManifold":
        """
        Return (and lazily construct) a CenterManifold of given degree.

        Heavy polynomial data (Hamiltonians in multiple coordinate systems,
        Lie generators, etc.) are cached inside the returned CenterManifold,
        not in the LibrationPoint itself.
        
        Parameters
        ----------
        degree : int
            The maximum degree of the center manifold expansion.
            
        Returns
        -------
        :class:`~hiten.system.center.CenterManifold`
            The center manifold instance.
        """
        from hiten.system.center import CenterManifold

        if degree not in self._cm_registry:
            self._cm_registry[degree] = CenterManifold(self, degree)
        return self._cm_registry[degree]

    def hamiltonian(self, max_deg: int) -> dict:
        """
        Return all Hamiltonian representations from the associated CenterManifold.

        Parameters
        ----------
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
            
        Returns
        -------
        dict
            Dictionary with keys: 'physical', 'real_normal', 'complex_normal', 
            'normalized', 'center_manifold_complex', 'center_manifold_real'.
            Each value is a list of coefficient arrays.
        """
        cm = self.get_center_manifold(max_deg)
        cm.compute()  # ensures all representations are cached

        reprs = {}
        for label in (
            'physical',
            'real_normal',
            'complex_normal',
            'normalized',
            'center_manifold_complex',
            'center_manifold_real',
        ):
            data = cm.cache_get(('hamiltonian', max_deg, label))
            if data is not None:
                reprs[label] = [arr.copy() for arr in data]
        return reprs

    def hamiltonian_system(self, form: str, max_deg: int) -> _HamiltonianSystem:
        """
        Return the Hamiltonian system for the given form.
        
        Parameters
        ----------
        form : str
            The Hamiltonian form identifier.
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
            
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`
            The Hamiltonian system instance.
        """
        cm = self.get_center_manifold(max_deg)
        return cm._get_hamsys(form)

    def generating_functions(self, max_deg: int):
        """
        Return the Lie-series generating functions from CenterManifold.
        
        Parameters
        ----------
        max_deg : int
            The maximum degree of the generating function expansion.
            
        Returns
        -------
        list
            List of generating function coefficient arrays.
        """
        cm = self.get_center_manifold(max_deg)
        cm.compute()  # ensure they exist
        data = cm.cache_get(('generating_functions', max_deg))
        return [] if data is None else [g.copy() for g in data]

    @abstractmethod
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the normal form transform for the Libration point.
        
        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix
            and Cinv is its inverse.
        """
        pass

    def __getstate__(self):
        """
        Custom state extractor to enable pickling.

        We remove attributes that may contain unpickleable Numba runtime
        objects (e.g., the compiled variational dynamics system) and restore
        them on unpickling.
        
        Returns
        -------
        dict
            The object state dictionary with unpickleable objects removed.
        """
        state = self.__dict__.copy()
        # Remove the compiled RHS system which cannot be pickled
        if '_var_eq_system' in state:
            state['_var_eq_system'] = None
        # Remove potential circular/self references to center manifolds
        if '_cm_registry' in state:
            state['_cm_registry'] = {}
        return state

    def __setstate__(self, state):
        """
        Restore object state after unpickling.

        The variational dynamics system is re-constructed because it was
        omitted during pickling (it contains unpickleable Numba objects).
        
        Parameters
        ----------
        state : dict
            The object state dictionary from pickling.
        """
        # Restore the plain attributes
        self.__dict__.update(state)
        # Recreate the compiled variational dynamics system on demand
        from hiten.algorithms.dynamics.rtbp import variational_dynsys
        self._var_eq_system = variational_dynsys(
            self.mu, name=f"CR3BP Variational Equations for {self.__class__.__name__}")

        # Ensure _cm_registry exists after unpickling
        if not hasattr(self, '_cm_registry') or self._cm_registry is None:
            self._cm_registry = {}

    def create_orbit(self, family: str | type["PeriodicOrbit"], /, **kwargs) -> "PeriodicOrbit":
        """
        Create a periodic orbit family anchored at this libration point.

        The helper transparently instantiates the appropriate concrete
        subclass of :class:`~hiten.system.orbits.base.PeriodicOrbit` and
        returns it.  The mapping is based on the family string or directly
        on a subclass type::

            L1 = system.get_libration_point(1)
            orb1 = L1.create_orbit("halo", amplitude_z=0.03, zenith="northern")
            orb2 = L1.create_orbit("lyapunov", amplitude_x=0.05)

        Parameters
        ----------
        family : str or :class:`~hiten.system.orbits.base.PeriodicOrbit` subclass
            Identifier of the orbit family or an explicit subclass type.
            Accepted strings (case-insensitive): "halo", "lyapunov",
            "vertical_lyapunov" and "generic".  If a subclass is
            passed, it is instantiated directly.
        **kwargs
            Forwarded verbatim to the underlying orbit constructor.

        Returns
        -------
        :class:`~hiten.system.orbits.base.PeriodicOrbit`
            Newly created orbit instance.
        """
        from hiten.system.orbits.base import GenericOrbit, PeriodicOrbit
        from hiten.system.orbits.halo import HaloOrbit
        from hiten.system.orbits.lyapunov import LyapunovOrbit
        from hiten.system.orbits.vertical import VerticalOrbit

        # Direct class provided
        if isinstance(family, type) and issubclass(family, PeriodicOrbit):
            orbit_cls = family
            return orbit_cls(self, **kwargs)

        # String identifier provided
        if not isinstance(family, str):
            raise TypeError("family must be either a string identifier or a PeriodicOrbit subclass")

        key = family.lower().strip()
        mapping: dict[str, type[PeriodicOrbit]] = {
            "halo": HaloOrbit,
            "lyapunov": LyapunovOrbit,
            "vertical_lyapunov": VerticalOrbit,
            "vertical": VerticalOrbit,
            "generic": GenericOrbit,
        }

        if key not in mapping:
            raise ValueError(
                f"Unknown orbit family '{family}'. Available options: {', '.join(mapping.keys())} "
                "or pass a PeriodicOrbit subclass directly."
            )

        orbit_cls = mapping[key]
        return orbit_cls(self, **kwargs)
