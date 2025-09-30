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
from typing import Literal, Tuple

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.torus import (_TorusPersistenceService,
                                                   _TorusServices)
from hiten.system.base import System
from hiten.system.libration.base import LibrationPoint
from hiten.system.orbits.base import PeriodicOrbit
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
        self._orbit = orbit

        services = _TorusServices.default(self)
        super().__init__(services)

    def __str__(self) -> str:
        return f"InvariantTori object for seed orbit={self.orbit} at point={self.libration_point})"

    def __repr__(self) -> str:
        return f"InvariantTori(orbit={self.orbit}, point={self.libration_point})"

    @property
    def orbit(self) -> PeriodicOrbit:
        """Periodic orbit about which the torus is constructed."""
        return self.dynamics.orbit

    @property
    def libration_point(self) -> LibrationPoint:
        """Libration point anchoring the family."""
        return self.dynamics.libration_point

    @property
    def system(self) -> System:
        """Parent CR3BP system."""
        return self.dynamics.system
    
    @property
    def dynsys(self):
        """Dynamical system."""
        return self.dynamics.dynsys

    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """Variational equations system."""
        return self.dynamics.var_dynsys

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """Jacobian evaluation system."""
        return self.dynamics.jacobian_dynsys
    
    @property
    def period(self) -> float:
        """Orbit period."""
        return self.dynamics.period
    
    @property
    def jacobi(self) -> float:
        """Jacobi constant."""
        return self.dynamics.jacobi

    @property
    def energy(self) -> float:
        """Orbit energy."""
        return self.dynamics.energy

    @property
    def grid(self) -> np.ndarray:
        """Invariant torus grid."""
        return self.dynamics.grid

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

        u_grid = self.dynamics.compute_grid(
            epsilon=epsilon,
            n_theta1=n_theta1,
            n_theta2=n_theta2,
            method=method,
            order=order,
        )

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


    def __setstate__(self, state):
        """Restore the InvariantTori instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of orbit.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the serialized state of the InvariantTori.
        """
        super().__setstate__(state)
        self._setup_services(_TorusServices.default(self))

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "InvariantTori":
        """Load an invariant torus from disk using the adapter."""
        return cls._load_with_services(
            filepath, 
            _TorusPersistenceService(), 
            _TorusServices.default, 
            **kwargs
        )

    def load_inplace(self, filepath: str, **kwargs) -> None:
        """Load invariant torus data from a file in place."""
        self.persistence.load_inplace(self, filepath)
        self.dynamics.reset()
        return self
