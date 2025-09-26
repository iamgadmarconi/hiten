"""High-level utilities for computing invariant tori in the circular restricted
three-body problem.

This module provides comprehensive tools for computing 2D invariant tori that
bifurcate from periodic orbits in the circular restricted three-body problem.
The implementation supports both linear approximation methods and advanced
algorithms like GMOS (Generalized Method of Characteristics) and KKG.

The torus is parameterized by two angles:
- theta1: longitudinal angle along the periodic orbit
- theta2: latitudinal angle in the transverse direction

The torus surface is given by:
u(theta1, theta2) = ubar(theta1) + epsilon * (cos(theta2) * Re(y(theta1)) - sin(theta2) * Im(y(theta1)))

where ubar is the periodic orbit trajectory and y is the complex eigenvector field.

Notes
-----
The module implements both linear approximation methods and advanced algorithms
for computing invariant tori. The linear approximation provides a good starting
point for more sophisticated methods.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.types.services.torus import _TorusServices
from hiten.algorithms.types.core import _HitenBase
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
from hiten.utils.log_config import logger
from hiten.utils.plots import plot_invariant_torus


@dataclass(slots=True, frozen=True)
class Torus:
    """
    Immutable representation of a 2-D invariant torus.

    This class represents a 2D invariant torus in the circular restricted
    three-body problem, parameterized by two angular coordinates theta1 and theta2.
    The torus is defined by a grid of state vectors and fundamental frequencies.

    Parameters
    ----------
    grid : numpy.ndarray
        Real 6-state samples of shape (n_theta1, n_theta2, 6).
        Each point represents a state vector on the torus surface.
    omega : numpy.ndarray
        Fundamental frequencies (omega_1, omega_2) in nondimensional units.
        omega_1 is the longitudinal frequency, omega_2 is the latitudinal frequency.
    C0 : float
        Jacobi constant (fixed along the torus family) in nondimensional units.
    system : System
        Parent CR3BP system (useful for downstream algorithms).

    Notes
    -----
    The torus is parameterized by two angles:
    - theta1: longitudinal angle along the periodic orbit
    - theta2: latitudinal angle in the transverse direction

    The fundamental frequencies determine the quasi-periodic motion on the torus.
    """

    grid: np.ndarray
    omega: np.ndarray
    C0: float
    system: System


class InvariantTori(_HitenBase):
    """
    Linear approximation of a 2-D invariant torus bifurcating from a
    centre component of a periodic orbit.

    This class implements the computation of invariant tori in the circular
    restricted three-body problem using linear approximation methods. The torus
    is constructed from a periodic orbit by analyzing the monodromy matrix
    and computing the associated eigenvector field.

    Parameters
    ----------
    orbit : :class:`~hiten.system.orbits.base.PeriodicOrbit`
        Corrected periodic orbit about which the torus is constructed. The
        orbit must expose a valid period attribute - no propagation is
        performed here; we only integrate the variational equations to
        obtain the state-transition matrices required by the algorithm.

    Mathematical Background
    ----------------------
    The invariant torus is parameterized by two angles:
    - theta1: longitudinal angle along the periodic orbit
    - theta2: latitudinal angle in the transverse direction

    The torus surface is given by:
    u(theta1, theta2) = ubar(theta1) + epsilon * (cos(theta2) * Re(y(theta1)) - sin(theta2) * Im(y(theta1)))

    where ubar is the periodic orbit trajectory and y is the complex eigenvector field.

    References
    ----------
    Szebehely, V. (1967). *Theory of Orbits*. Academic Press.
    """

    def __init__(self, orbit: PeriodicOrbit):
        if orbit.period is None:
            raise ValueError("The generating orbit must be corrected first (period is None).")

        self._orbit = orbit

        self._services = _TorusServices.from_torus(self)
        monodromy, evals, evecs = self._services.dynamics.eigen_data()

        self._monodromy = monodromy
        self._evals = evals
        self._evecs = evecs

        # Internal caches populated via services.
        self._theta1: Optional[np.ndarray] = None
        self._ubar: Optional[np.ndarray] = None
        self._y_series: Optional[np.ndarray] = None
        self._grid: Optional[np.ndarray] = None
        self._v_curve_prev: Optional[np.ndarray] = None
        self._family_tangent: Optional[np.ndarray] = None



    def __str__(self) -> str:
        return f"InvariantTori object for seed orbit={self.orbit} at point={self.libration_point})"

    def __repr__(self) -> str:
        return f"InvariantTori(orbit={self.orbit}, point={self.libration_point})"

    @property
    def orbit(self) -> PeriodicOrbit:
        """Periodic orbit about which the torus is constructed."""
        return self._orbit

    @property
    def libration_point(self) -> LibrationPoint:
        """Libration point anchoring the family."""
        return self.orbit.libration_point

    @property
    def system(self) -> System:
        """Parent CR3BP system."""
        return self.libration_point.system
    
    @property
    def dynsys(self):
        """Dynamical system."""
        return self.system.dynsys

    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """Variational equations system."""
        return self.system.var_dynsys

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """Jacobian evaluation system."""
        return self.system.jacobian_dynsys
    
    @property
    def period(self) -> float:
        """Orbit period."""
        return float(self.orbit.period)
    
    @property
    def jacobi(self) -> float:
        """Jacobi constant."""
        return float(self.orbit.jacobi_constant)

    @property
    def grid(self) -> np.ndarray:
        """Invariant torus grid."""
        return self._services.dynamics.grid

    def compute(
        self,
        *,
        epsilon: float,
        n_theta1: int,
        n_theta2: int,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order: int = 8,
    ) -> np.ndarray:
        """Compute the invariant torus grid.
        
        Parameters
        ----------
        epsilon : float
            Torus amplitude used in the linear approximation.
        n_theta1 : int
            Number of discretisation points along theta1.
        n_theta2 : int
            Number of discretisation points along theta2.

        Returns
        -------
        numpy.ndarray
            Invariant torus grid.

        Notes
        -----
        This method computes the invariant torus grid using the linear approximation.
        The grid is computed using the cached STM samples and the complex eigenvector field.
        The grid is cached for subsequent plotting and state export.
        """

        u_grid = self._services.dynamics.compute_grid(
            epsilon=epsilon,
            n_theta1=n_theta1,
            n_theta2=n_theta2,
            method=method,
            order=order,
        )
        self._grid = u_grid
        return u_grid

    def plot(
        self,
        *,
        figsize: Tuple[int, int] = (10, 8),
        save: bool = False,
        dark_mode: bool = True,
        filepath: str = "invariant_torus.svg",
        **kwargs,
    ):
        """
        Render the invariant torus using :func:`~hiten.utils.plots.plot_invariant_torus`.

        Parameters
        ----------
        figsize : Tuple[int, int], default (10, 8)
            Figure size in inches.
        save : bool, default False
            Whether to save the plot to a file.
        dark_mode : bool, default True
            Whether to use dark mode styling.
        filepath : str, default "invariant_torus.svg"
            File path for saving the plot.
        **kwargs : dict
            Additional keyword arguments accepted by
            :func:`~hiten.utils.plots.plot_invariant_torus`.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure object.
        """
        return plot_invariant_torus(
            self.grid,
            [self.system.primary, self.system.secondary],
            self.system.distance,
            figsize=figsize,
            save=save,
            dark_mode=dark_mode,
            filepath=filepath,
            **kwargs,
        )

    def save(self, path: str | Path, **kwargs) -> None:
        """Persist this torus using the configured services."""
        self._services.persistence.save(self, path, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "InvariantTori":
        """Load an invariant torus from disk using the adapter."""
        services = _TorusServices.from_torus(cls)
        torus = services.persistence.load(path)
        torus._services = services
        return torus

    def load_inplace(self, path: str | Path, **kwargs) -> None:
        """Populate this torus instance from persisted data."""
        self._services.persistence.load_inplace(self, path, **kwargs)
