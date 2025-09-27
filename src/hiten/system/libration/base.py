"""Abstract helpers to model Libration points of the Circular Restricted Three-Body Problem (CR3BP).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.dynamics.hamiltonian import _HamiltonianSystem
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.libration import (
    _LibrationPersistenceService, _LibrationServices)

if TYPE_CHECKING:
    from hiten.system.base import System
    from hiten.system.center import CenterManifold
    from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction
    from hiten.system.orbits.base import PeriodicOrbit


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
        self._system = system
        services = _LibrationServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mu={self.mu:.6e})"

    @property
    def system(self) -> "System":
        """The system this libration point belongs to."""
        return self._system
    
    @property
    def mu(self) -> float:
        """The mass parameter of the system."""
        return self._system.mu

    @property
    def dynsys(self):
        """Underlying vector field instance.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying vector field instance.
        """
        return self.dynamics.dynsys

    @property
    def var_dynsys(self):
        """Underlying variational equations system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying variational equations system.
        """
        return self.dynamics.var_dynsys

    @property
    def jacobian_dynsys(self):
        """Underlying Jacobian evaluation system.
        
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
            The underlying Jacobian evaluation system.
        """
        return self.dynamics.jacobian_dynsys

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
    def energy(self) -> float:
        """
        Get the energy of the Libration point.
        
        Returns
        -------
        float
            The mechanical energy in nondimensional units.
        """
        return self.dynamics.energy
    
    @property
    def jacobi(self) -> float:
        """
        Get the Jacobi constant of the Libration point.
        
        Returns
        -------
        float
            The Jacobi constant in nondimensional units.
        """
        return self.dynamics.jacobi
    
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
        return self.dynamics.is_stable

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
        return self.dynamics.eigenvalues
    
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
        return self.dynamics.eigenvectors

    def create_orbit(self, family: str | type["PeriodicOrbit"], /, **kwargs) -> "PeriodicOrbit":
        return self.dynamics.create_orbit(family, **kwargs)

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
        return self.dynamics.center_manifold(degree)

    def hamiltonian(self, max_deg: int, form: str = "physical") -> "Hamiltonian":
        """
        Return a Hamiltonian object from the associated CenterManifold.

        Parameters
        ----------
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
        form : str
            The Hamiltonian form to get coefficients for. Default is "physical".
            Available forms: 'physical', 'real_normal', 'complex_normal', 
            'normalized', 'center_manifold_complex', 'center_manifold_real'.
            
        Returns
        -------
        Hamiltonian
            The Hamiltonian object with the specified form and degree.
        """
        return self.dynamics.hamiltonian(max_deg, form)

    def hamiltonians(self, max_deg: int) -> dict[str, "Hamiltonian"]:
        """
        Return all Hamiltonian representations from the associated CenterManifold.

        Parameters
        ----------
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
            
        Returns
        -------
        dict[str, Hamiltonian]
            Dictionary with keys: 'physical', 'real_normal', 'complex_normal', 
            'normalized', 'center_manifold_complex', 'center_manifold_real'.
            Each value is a Hamiltonian object.
        """
        return self.dynamics.hamiltonians(max_deg)

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
        return self.dynamics.hamsys(max_deg, form)

    def generating_functions(self, max_deg: int) -> list["LieGeneratingFunction"]:
        """
        Return the Lie-series generating functions from CenterManifold.
        
        Parameters
        ----------
        max_deg : int
            The maximum degree of the generating function expansion.
            
        Returns
        -------
        list[LieGeneratingFunction]
            List of LieGeneratingFunction objects.
        """
        return self.dynamics.generating_functions(max_deg)

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the Libration point in the rotating frame.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            3D vector [x, y, z] representing the position in nondimensional units.
        """
        return self._services.dynamics.position

    @property
    def linear_data(self):
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        :class:`~hiten.algorithms.types.services.libration._LinearData`
            The linear data containing eigenvalues and eigenvectors.
        """
        return self._services.dynamics.linear_data

    @property
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the normal form transform for the Libration point.
        
        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix
            and Cinv is its inverse.
        """
        return self._services.dynamics.normal_form_transform

    def __setstate__(self, state):
        """Restore adapter wiring after unpickling."""
        super().__setstate__(state)
        self._setup_services(_LibrationServices.default(self))

    def load_inplace(self, filepath: str | Path) -> "LibrationPoint":
        self.persistence.load_inplace(self, filepath)
        self.dynamics.reset()
        return self

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "LibrationPoint":
        return cls._load_with_services(
            filepath, 
            _LibrationPersistenceService(), 
            _LibrationServices.default, 
            **kwargs
        )
