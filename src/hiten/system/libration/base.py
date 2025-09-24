"""Abstract helpers to model Libration points of the Circular Restricted Three-Body Problem (CR3BP).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.common.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.dynamics.hamiltonian import _HamiltonianSystem
from hiten.algorithms.dynamics.rtbp import jacobian_dynsys, variational_dynsys
from hiten.algorithms.linalg.base import StabilityProperties
from hiten.algorithms.linalg.config import _EigenDecompositionConfig
from hiten.algorithms.linalg.interfaces import _LibrationPointInterface
from hiten.algorithms.linalg.types import _ProblemType, _SystemType
from hiten.algorithms.types.core import _HitenBase
from hiten.utils.io.libration import (load_libration_point,
                                      load_libration_point_inplace,
                                      save_libration_point)

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.center import CenterManifold
    from hiten.system.orbits.base import PeriodicOrbit


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


class LibrationPoint(_HitenBase, ABC):
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
        :func:`~hiten.algorithms.common.energy.crtbp_energy`.
    jacobi_constant : float
        Jacobi integral CJ = -2E corresponding to energy (dimensionless).
    is_stable : bool
        True if all eigenvalues returned by 
        :meth:`~hiten.system.libration.base.LibrationPoint.is_stable` lie
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
        super().__init__()
        self._system = system
        self._mu = system.mu

        self._linear_data: LinearData | None = None
        self._cm_registry = {}
        self._stability_properties: StabilityProperties | None = None
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def cache_clear(self) -> None:
        """
        Clear all cached data, including computed properties.
        
        This method resets all cached properties to None, forcing them to be
        recomputed on next access.
        """
        super().cache_clear()
        self._stability_properties = None
        self._linear_data = None

    @property
    def system(self) -> "System":
        """The system this libration point belongs to."""
        return self._system
    
    @property
    def mu(self) -> float:
        """The mass parameter of the system."""
        return self._mu

    @property
    def dynsys(self) -> _DynamicalSystem:
        """Underlying vector field instance.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying vector field instance.
        """
        return self.system.dynsys

    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """Underlying variational equations system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying variational equations system.
        """
        return self.system.var_dynsys

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """Underlying Jacobian evaluation system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying Jacobian evaluation system.
        """
        return self.system.jacobian_dynsys

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
        cached = self.cache_get(('position',))
        if cached is None:
            cached = self.cache_set(('position',), self._calculate_position())
        return cached
    
    @property
    def energy(self) -> float:
        """
        Get the energy of the Libration point.
        
        Returns
        -------
        float
            The mechanical energy in nondimensional units.
        """
        cached = self.cache_get(('energy',))
        if cached is None:
            state = np.concatenate([self.position, np.array([0.0, 0.0, 0.0])])
            cached = self.cache_set(('energy',), crtbp_energy(state, self.mu))
        return cached
    
    @property
    def jacobi_constant(self) -> float:
        """
        Get the Jacobi constant of the Libration point.
        
        Returns
        -------
        float
            The Jacobi constant in nondimensional units.
        """
        cached = self.cache_get(('jacobi_constant',))
        if cached is None:
            cached = self.cache_set(('jacobi_constant',), energy_to_jacobi(self.energy))
        return cached
    
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
        config = _EigenDecompositionConfig(
            problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
            system_type=_SystemType.CONTINUOUS,
        )
        problem = _LibrationPointInterface(config=config).create_problem(self)
        if self._stability_properties is None:
            self._stability_properties = StabilityProperties.with_default_engine(config=config)
        self._stability_properties.compute(problem.A, system_type=config.system_type, problem_type=config.problem_type)
        return self._stability_properties.is_stable

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
        config = _EigenDecompositionConfig(
            problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
            system_type=_SystemType.CONTINUOUS,
        )
        problem = _LibrationPointInterface(config=config).create_problem(self)
        if self._stability_properties is None:
            self._stability_properties = StabilityProperties.with_default_engine(config=config)
        self._stability_properties.compute(problem.A, system_type=config.system_type, problem_type=config.problem_type)
        return self._stability_properties.eigenvalues
    
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
        config = _EigenDecompositionConfig(
            problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
            system_type=_SystemType.CONTINUOUS,
        )
        problem = _LibrationPointInterface(config=config).create_problem(self)
        if self._stability_properties is None:
            self._stability_properties = StabilityProperties.with_default_engine(config=config)
        self._stability_properties.compute(problem.A, system_type=config.system_type, problem_type=config.problem_type)
        return self._stability_properties.eigenvectors

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
        center_manifold = self.get_center_manifold(max_deg)
        center_manifold.compute()

        reprs = {}
        for label in (
            'physical',
            'real_normal',
            'complex_normal',
            'normalized',
            'center_manifold_complex',
            'center_manifold_real',
        ):
            data = center_manifold.cache_get(('hamiltonian', max_deg, label))
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
        center_manifold = self.get_center_manifold(max_deg)
        return center_manifold._get_hamsys(form)

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
        center_manifold = self.get_center_manifold(max_deg)
        center_manifold.compute()
        data = center_manifold.cache_get(('generating_functions', max_deg))
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
        state = super().__getstate__()

        if '_var_eq_system' in state:
            state['_var_eq_system'] = None

        if '_jacobian_system' in state:
            state['_jacobian_system'] = None

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

        super().__setstate__(state)
        self._var_eq_system = variational_dynsys(
            self.mu, name=f"CR3BP Variational Equations for {self.__class__.__name__}")

        self._jacobian_system = jacobian_dynsys(
            self.mu, name=f"CR3BP Jacobian for {self.__class__.__name__}")

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

        if isinstance(family, type) and issubclass(family, PeriodicOrbit):
            orbit_cls = family
            return orbit_cls(self, **kwargs)

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

    def save(self, file_path: str | Path, **kwargs) -> None:
        save_libration_point(self, Path(file_path), **kwargs)

    @classmethod
    def load(cls, file_path: str | Path, **kwargs) -> "LibrationPoint":
        system: System = kwargs.get("system")
        return load_libration_point(Path(file_path), system)

    def load_inplace(self, file_path: str | Path) -> "LibrationPoint":
        load_libration_point_inplace(self, Path(file_path))
        return self