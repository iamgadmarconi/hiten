"""High-level abstractions for the Circular Restricted Three-Body Problem (CR3BP).

This module bundles the physical information of a binary system, computes the
mass parameter mu, instantiates the underlying vector field via
:func:`~hiten.algorithms.dynamics.rtbp.rtbp_dynsys`, and pre-computes the five
classical Lagrange (libration) points.

Notes
-----
All positions and velocities are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Sequence

import numpy as np

from hiten.algorithms.types.core import _HitenBase
from hiten.algorithms.types.services.system import (_SystemPersistenceService,
                                                    _SystemServices)
from hiten.algorithms.types.states import Trajectory
from hiten.system.body import Body
from hiten.system.libration.base import LibrationPoint
from hiten.utils.constants import Constants


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
        self._primary = primary
        self._secondary = secondary
        self._distance = distance
        self._libration_points: Dict[int, LibrationPoint] = {}

        services = _SystemServices.default(self)
        super().__init__(services)

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
        return self.dynamics.primary

    @property
    def secondary(self) -> Body:
        """Secondary gravitating body.
        
        Returns
        -------
        :class:`~hiten.system.body.Body`
            The secondary gravitating body.
        """
        return self.dynamics.secondary

    @property
    def distance(self) -> float:
        """Characteristic separation between the bodies.
        
        Returns
        -------
        float
            The characteristic separation between the bodies in km.
        """
        return self.dynamics.distance

    @property
    def mu(self) -> float:
        """Mass parameter mu.
        
        Returns
        -------
        float
            The mass parameter mu = m2 / (m1 + m2) (dimensionless).
        """
        return self.dynamics.mu

    @property
    def libration_points(self) -> Dict[int, LibrationPoint]:
        """Mapping from integer identifiers {1,...,5} to libration point objects.
        
        Returns
        -------
        dict[int, LibrationPoint]
            Dictionary mapping integer identifiers {1,...,5} to libration point objects.
        """
        return self._libration_points
        
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
        if index not in self._libration_points:
            self._libration_points[index] = self.dynamics.get_point(index)
        return self._libration_points[index]

    def propagate(
        self,
        initial_conditions: Sequence[float],
        tf: float = 2 * np.pi,
        *,
        steps: int = 1000,
        method: Literal["fixed", "adaptive", "symplectic"] = "adaptive",
        order: int = 8,
        forward: int = 1,
    ) -> Trajectory:
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

        Returns
        -------
        :class:`~hiten.algorithms.types.states.Trajectory`
            The propagated trajectory.
        """

        traj = self.dynamics.propagate(
            initial_conditions,
            tf=tf,
            steps=steps,
            method=method,
            order=order,
            forward=forward,
        )

        return traj

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
        p_key = primary_name.lower()
        s_key = secondary_name.lower()
        try:
            p_mass = Constants.get_mass(p_key)
            p_radius = Constants.get_radius(p_key)
            s_mass = Constants.get_mass(s_key)
            s_radius = Constants.get_radius(s_key)
            distance = Constants.get_orbital_distance(p_key, s_key)
        except KeyError as exc:
            raise ValueError(
                f"Unknown body or orbital distance for pair '{primary_name}', '{secondary_name}'."
            ) from exc

        primary = Body(primary_name.capitalize(), p_mass, p_radius)
        secondary = Body(secondary_name.capitalize(), s_mass, s_radius, parent=primary)

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
        super().__setstate__(state)
        self._setup_services(_SystemServices.default(self))

    @classmethod
    def load(cls, filepath: str | Path, **kwargs) -> "System":
        """Load a System from a file (new instance)."""
        return cls._load_with_services(
            filepath, 
            _SystemPersistenceService(), 
            _SystemServices.default, 
            **kwargs
        )
