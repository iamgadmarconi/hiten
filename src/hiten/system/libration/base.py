r"""
hiten.system.libration.base
=====================

Abstract helpers to model Libration points of the Circular Restricted Three-Body Problem (CR3BP).

The module introduces two primary abstractions:

* :pyclass:`LinearData` - an immutable record storing the salient linear characteristics (eigenfrequencies and canonical basis) of the flow linearised at a libration point.
* :pyclass:`LibrationPoint` - an abstract base class encapsulating geometry, energetic properties, linear stability analysis and lazy construction of centre-manifold normal forms. 
   Concrete subclasses implement the specific coordinates of the collinear (:math:`L_1`, :math:`L_2`, :math:`L_3`) and triangular (:math:`L_4`, :math:`L_5`) points.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.dynamics.rtbp import _jacobian_crtbp, variational_dynsys
from hiten.algorithms.dynamics.utils.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.dynamics.utils.linalg import eigenvalue_decomposition
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.base import System

# Constants for stability analysis mode
CONTINUOUS_SYSTEM = 0
DISCRETE_SYSTEM = 1


@dataclass(slots=True)
class LinearData:
    r"""
    Container with linearised CR3BP invariants.

    Parameters
    ----------
    mu : float
        Mass ratio :math:`\mu := m_2/(m_1+m_2)` of the primaries.
    point : str
        Identifier of the libration point (``'L1'``, ``'L2'`` or ``'L3'``).
    lambda1 : float
        Real hyperbolic eigenvalue :math:`\lambda_1>0` associated with the
        saddle behaviour along the centre-saddle subspace.
    omega1 : float
        First elliptic frequency :math:`\omega_1>0` of the centre subspace.
    omega2 : float
        Second elliptic frequency :math:`\omega_2>0` of the centre subspace.
    C : numpy.ndarray, shape (6, 6)
        Symplectic change-of-basis matrix such that :math:`C^{-1}AC` is in real
        Jordan canonical form, with :math:`A` the Jacobian of the vector
        field evaluated at the libration point.
    Cinv : numpy.ndarray, shape (6, 6)
        Precomputed inverse of :pyattr:`C`.

    Notes
    -----
    The record is *immutable* thanks to ``slots=True``; all fields are plain
    :pyclass:`numpy.ndarray` or scalars so the instance can be safely cached
    and shared among different computations.
    """
    mu: float
    point: str        # 'L1', 'L2', 'L3'
    lambda1: float
    omega1: float
    omega2: float
    C: np.ndarray     # 6×6 symplectic transform
    Cinv: np.ndarray  # inverse


class LibrationPoint(ABC):
    r"""
    Abstract base class for Libration points of the CR3BP.

    Parameters
    ----------
    system : hiten.system.base.System
        Parent CR3BP model providing the mass ratio :math:`\mu` and utility
        functions.

    Attributes
    ----------
    mu : float
        Mass ratio :math:`\mu` of the primaries (copied from *system*).
    system : hiten.system.base.System
        Reference to the owner hiten.system.
    position : numpy.ndarray, shape (3,)
        Cartesian coordinates in the synodic rotating frame. Evaluated on
        first access and cached thereafter.
    energy : float
        Dimensionless mechanical energy evaluated via
        :pyfunc:`hiten.algorithms.dynamics.hiten.utils.energy.crtbp_energy`.
    jacobi_constant : float
        Jacobi integral :math:`C_J = -2E` corresponding to
        :pyattr:`energy`.
    is_stable : bool
        True if all eigenvalues returned by :pyfunc:`analyze_stability` lie
        inside the unit circle (discrete case) or have non-positive real
        part (continuous case).
    eigenvalues : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Arrays of stable, unstable and centre eigenvalues.
    eigenvectors : tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Bases of the corresponding invariant subspaces.
    linear_data : LinearData
        Record with canonical invariants and symplectic basis returned by the
        normal-form computation.

    Notes
    -----
    The class is *abstract*. Concrete subclasses must implement:

    * :pyfunc:`idx`
    * :pyfunc:`_calculate_position`
    * :pyfunc:`_get_linear_data`
    * :pyfunc:`normal_form_transform`

    Heavy algebraic objects produced by the centre-manifold normal-form
    procedure are cached inside a dedicated
    :pyclass:`hiten.system.center.CenterManifold` instance to avoid memory
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
        self._var_eq_system = variational_dynsys(self.mu, name=f"CR3BP Variational Equations for {self.__class__.__name__}")
    
    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    @property
    def idx(self) -> int:
        pass

    @property
    def position(self) -> np.ndarray:
        r"""
        Get the position of the Libration point in the rotating frame.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        if self._position is None:
            self._position = self._calculate_position()
        return self._position
    
    @property
    def energy(self) -> float:
        r"""
        Get the energy of the Libration point.
        """
        if self._energy is None:
            self._energy = self._compute_energy()
        return self._energy
    
    @property
    def jacobi_constant(self) -> float:
        r"""
        Get the Jacobi constant of the Libration point.
        """
        if self._jacobi_constant is None:
            self._jacobi_constant = self._compute_jacobi_constant()
        return self._jacobi_constant
    
    @property
    def is_stable(self) -> bool:
        r"""
        Check if the Libration point is stable.
        """
        if self._stability_info is None:
            self.analyze_stability() 
        
        indices = self._stability_info[0] 
        return np.all(np.abs(indices) <= 1.0 + 1e-9)

    @property
    def linear_data(self) -> LinearData:
        r"""
        Get the linear data for the Libration point.
        """
        if self._linear_data is None:
            self._linear_data = self._get_linear_data()
        return self._linear_data

    def _compute_energy(self) -> float:
        r"""
        Compute the energy of the Libration point.
        """
        state = np.concatenate([self.position, [0, 0, 0]])
        return crtbp_energy(state, self.mu)

    def _compute_jacobi_constant(self) -> float:
        r"""
        Compute the Jacobi constant of the Libration point.
        """
        return energy_to_jacobi(self.energy)

    def analyze_stability(self, discrete: int = CONTINUOUS_SYSTEM, delta: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Analyze the stability properties of the Libration point.
        
        Parameters
        ----------
        discrete : int, optional
            Classification mode for eigenvalues:
            * CONTINUOUS_SYSTEM (0): continuous-time system (classify by real part sign)
            * DISCRETE_SYSTEM (1): discrete-time system (classify by magnitude relative to 1)
        delta : float, optional
            Tolerance for classification
            
        Returns
        -------
        tuple
            (sn, un, cn, Ws, Wu, Wc) containing:
            - sn: stable eigenvalues
            - un: unstable eigenvalues
            - cn: center eigenvalues
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

    def cache_get(self, key) -> any:
        r"""
        Get item from cache.
        """
        return self._cache.get(key)
    
    def cache_set(self, key, value) -> any:
        r"""
        Set item in cache and return the value.
        """
        self._cache[key] = value
        return value
    
    def cache_clear(self) -> None:
        r"""
        Clear all cached data.
        """
        self._cache.clear()
        logger.debug(f"Cache cleared for {type(self).__name__}")

    def get_center_manifold(self, max_degree: int):
        r"""
        Return (and lazily construct) a CenterManifold of given degree.

        Heavy polynomial data (Hamiltonians in multiple coordinate systems,
        Lie generators, etc.) are cached *inside* the returned CenterManifold,
        not in the LibrationPoint itself.
        """
        from hiten.system.center import CenterManifold

        if max_degree not in self._cm_registry:
            self._cm_registry[max_degree] = CenterManifold(self, max_degree)
        return self._cm_registry[max_degree]

    def hamiltonian(self, max_deg: int) -> dict:
        r"""
        Return all Hamiltonian representations from the associated CenterManifold.

        Keys: 'physical', 'real_normal', 'complex_normal', 'normalized',
        'center_manifold_complex', 'center_manifold_real'.
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

    def generating_functions(self, max_deg: int):
        r"""
        Return the Lie-series generating functions from CenterManifold.
        """
        cm = self.get_center_manifold(max_deg)
        cm.compute()  # ensure they exist
        data = cm.cache_get(('generating_functions', max_deg))
        return [] if data is None else [g.copy() for g in data]

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Get the eigenvalues of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvalues, unstable_eigenvalues, center_eigenvalues)
        """
        if self._stability_info is None:
            self.analyze_stability() # Ensure stability is analyzed
        sn, un, cn, _, _, _ = self._stability_info
        return (sn, un, cn)
    
    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Get the eigenvectors of the linearized system at the Libration point.
        
        Returns
        -------
        tuple
            (stable_eigenvectors, unstable_eigenvectors, center_eigenvectors)
        """
        if self._stability_info is None:
            self.analyze_stability() # Ensure stability is analyzed
        _, _, _, Ws, Wu, Wc = self._stability_info
        return (Ws, Wu, Wc)
    
    @abstractmethod
    def _calculate_position(self) -> np.ndarray:
        r"""
        Calculate the position of the Libration point.
        
        This is an abstract method that must be implemented by subclasses.
        
        Returns
        -------
        ndarray
            3D vector [x, y, z] representing the position
        """
        pass

    @abstractmethod
    def _get_linear_data(self) -> LinearData:
        r"""
        Get the linear data for the Libration point.
        """
        pass

    @abstractmethod
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Get the normal form transform for the Libration point.
        """
        pass