"""High-level utilities for computing a polynomial normal form of the centre
manifold around a collinear libration point of the spatial circular
restricted three body problem (CRTBP).

All heavy algebra is performed symbolically on packed coefficient arrays.
Only NumPy is used so the implementation is portable and fast.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.

References
----------
Jorba, A. (1999). "A Methodology for the Numerical Computation of Normal Forms, Centre
Manifolds and First Integrals of Hamiltonian Systems".

Zhang, H. Q., Li, S. (2001). "Improved semi-analytical computation of center
manifolds near collinear libration points".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.center import (
    _CenterManifoldPersistenceService, _CenterManifoldServices)

if TYPE_CHECKING:
    from hiten.system.hamiltonian import Hamiltonian
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.maps.center import CenterManifoldMap


class CenterManifold(_HitenBase):
    """Centre manifold normal-form builder orchestrating adapter services."""

    def __init__(self, point: "LibrationPoint", degree: int):
        self._point = point
        self._max_degree = degree
        services = _CenterManifoldServices.default(self)
        super().__init__(services)

    @property
    def point(self) -> "LibrationPoint":
        return self.dynamics.point

    @property
    def degree(self) -> int:
        return self.dynamics.degree

    @degree.setter
    def degree(self, value: int) -> None:
        self.dynamics.degree = value

    def hamiltonian(self, degree: int) -> "Hamiltonian":
        return self.dynamics.hamiltonian(degree)

    def __str__(self) -> str:
        return f"CenterManifold(point={self.point}, degree={self.degree})"

    def __repr__(self) -> str:
        return self.__str__()

    def compute(self, form: str = "center_manifold_real") -> "Hamiltonian":
        return self.dynamics.pipeline.get_hamiltonian(form)

    def coefficients(self,form: str = "center_manifold_real", degree = None) -> str:
        return self.dynamics.format_coefficients(self.dynamics.pipeline.get_hamiltonian(form), degree)

    def to_synodic(self, cm_point, energy: Optional[float] = None, section_coord: str = "q3", tol: float = 1e-14):
        return self.dynamics.cm_point_to_synodic(cm_point, energy=energy, section_coord=section_coord, tol=tol)

    def to_cm(self, synodic_6d, tol=1e-14):
        return self.dynamics.synodic_to_cm(synodic_6d, tol=tol)

    def poincare_map(self, energy: float) -> "CenterManifoldMap":
        return self.dynamics.get_map(energy)

    def __setstate__(self, state):
        """Restore adapter wiring after unpickling."""
        super().__setstate__(state)
        self._point = state["_point"]
        self._max_degree = state["_max_degree"]
        self._setup_services(_CenterManifoldServices.default(self))

    @classmethod
    def load(cls, dir_path: str, **kwargs) -> "CenterManifold":
        """
        Load a :class:`~hiten.system.center.CenterManifold` instance from a directory.

        This class method deserializes a CenterManifold object and its
        associated Poincare maps that were saved with the save method.

        Parameters
        ----------
        dir_path : str or path-like object
            The path to the directory from which to load the data.
        **kwargs
            Additional keyword arguments for the load operation.

        Returns
        -------
        :class:`~hiten.system.center.CenterManifold`
            The loaded CenterManifold instance with its Poincare maps.
        """
        return cls._load_with_services(
            dir_path, 
            _CenterManifoldPersistenceService(), 
            lambda cm: _CenterManifoldServices.default(cm), 
            **kwargs
        )
