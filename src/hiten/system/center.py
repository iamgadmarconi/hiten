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

from hiten.algorithms.types.adapters.center import _CenterManifoldServices
from hiten.algorithms.types.core import _HitenBase
from hiten.system.libration.base import LibrationPoint
from hiten.utils.io.center import load_center_manifold, save_center_manifold

if TYPE_CHECKING:
    from hiten.algorithms.poincare.centermanifold.base import CenterManifoldMap


class CenterManifold(_HitenBase):
    """Centre manifold normal-form builder orchestrating adapter services."""

    def __init__(
        self,
        point: LibrationPoint,
        degree: int,
        *,
        services: Optional[_CenterManifoldServices] = None,
    ) -> None:
        self._point = point
        self._max_degree = degree
        self._services = services or _CenterManifoldServices.from_point(point, degree)

    @property
    def point(self) -> LibrationPoint:
        return self._point

    @property
    def degree(self) -> int:
        return self._max_degree

    @degree.setter
    def degree(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("degree must be a positive integer.")

        if value != self._max_degree:
            self._max_degree = value
            self._services.dynamics.pipeline_for_degree(value)

    @property
    def pipeline(self):
        return self._services.dynamics.pipeline

    def __str__(self) -> str:
        return f"CenterManifold(point={self._point}, degree={self._max_degree})"

    def __repr__(self) -> str:
        return self.__str__()

    def __getstate__(self):
        return {
            "_point": self._point,
            "_max_degree": self._max_degree,
        }

    def __setstate__(self, state):
        self._point = state["_point"]
        self._max_degree = state["_max_degree"]
        self._services = _CenterManifoldServices.from_point(self._point, self._max_degree)

    def compute(self, form: str = "center_manifold_real"):
        pipeline = self._services.dynamics.pipeline
        return pipeline.get_hamiltonian(form).poly_H

    def coefficients(
        self,
        form: str = "center_manifold_real",
        degree = None,
    ) -> str:
        pipeline = self._services.dynamics.pipeline
        ham = pipeline.get_hamiltonian(form)
        return self._services.dynamics.format_coefficients(ham, degree)

    def cache_clear(self):
        self._services.dynamics.clear_caches()

    def to_synodic(self, cm_point, energy: Optional[float] = None, section_coord: str = "q3", tol: float = 1e-14):
        return self._services.dynamics.cm_point_to_synodic(cm_point, energy=energy, section_coord=section_coord, tol=tol)

    def to_cm(self, synodic_6d, tol=1e-14):
        return self._services.dynamics.synodic_to_cm(synodic_6d, tol=tol)

    def poincare_map(self, energy: float, **kwargs) -> "CenterManifoldMap":
        return self._services.dynamics.get_map(self, energy, **kwargs)

    def save(self, dir_path: str, **kwargs):
        """
        Save the :class:`~hiten.system.center.CenterManifold` instance to a directory.

        This method serializes the main object to 'manifold.pkl' and saves
        each associated Poincare map to a separate file within a 'poincare_maps'
        subdirectory.

        Parameters
        ----------
        dir_path : str or path-like object
            The path to the directory where the data will be saved.
        **kwargs
            Additional keyword arguments for the save operation.
        """
        save_center_manifold(self, dir_path, **kwargs)

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
        return load_center_manifold(dir_path, **kwargs)



