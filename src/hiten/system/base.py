r"""
hiten.system.base
===========

High-level abstractions for the Circular Restricted Three-Body Problem (CR3BP).

This module bundles the physical information of a binary system, computes the
mass parameter :math:`\mu`, instantiates the underlying vector field via
:pyfunc:`hiten.algorithms.dynamics.rtbp.rtbp_dynsys`, and pre-computes the five
classical Lagrange (libration) points.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from hiten.algorithms.dynamics.base import _propagate_dynsys
from hiten.algorithms.dynamics.rtbp import rtbp_dynsys
from hiten.algorithms.utils.precision import hp
from hiten.system.body import Body
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import L1Point, L2Point, L3Point
from hiten.system.libration.triangular import L4Point, L5Point
from hiten.utils.constants import Constants
from hiten.utils.log_config import logger


class System(object):
    r"""
    Lightweight wrapper around the CR3BP dynamical hiten.system.

    The class stores the physical properties of the primaries, computes the
    dimensionless mass parameter :math:`\mu = m_2 / (m_1 + m_2)`, instantiates
    the CR3BP vector field through :pyfunc:`hiten.algorithms.dynamics.rtbp.rtbp_dynsys`,
    and caches the five Lagrange points.

    Parameters
    ----------
    primary : Body
        Primary gravitating body.
    secondary : Body
        Secondary gravitating body.
    distance : float
        Characteristic separation between the bodies in km.

    Attributes
    ----------
    primary : Body
        Primary gravitating body.
    secondary : Body
        Secondary gravitating body.
    distance : float
        Characteristic separation between the bodies in km.
    mu : float
        Mass parameter :math:`\mu`.
    libration_points : dict[int, LibrationPoint]
        Mapping from integer identifiers {1,...,5} to the corresponding
        libration point objects.
    _dynsys : hiten.algorithms.dynamics.base._DynamicalSystemProtocol
        Underlying vector field instance compatible with the integrators
        defined in :pymod:`hiten.algorithms.integrators`.

    Notes
    -----
    The heavy computations reside in the dynamical system and individual
    libration point classes; this wrapper simply orchestrates them.
    """
    def __init__(self, primary: Body, secondary: Body, distance: float):
        """Initializes the CR3BP system based on the provided configuration."""

        logger.info(f"Initializing System with primary='{primary.name}', secondary='{secondary.name}', distance={distance:.4e}")
        
        self._primary = primary
        self._secondary = secondary
        self._distance = distance

        self._mu: float = self._get_mu()
        logger.info(f"Calculated mass parameter mu = {self.mu:.6e}")

        self._libration_points: Dict[int, LibrationPoint] = self._compute_libration_points()
        logger.info(f"Computed {len(self.libration_points)} Libration points.")

        self._dynsys = rtbp_dynsys(self.mu, name=self.primary.name + "_" + self.secondary.name)

    def __str__(self) -> str:
        return f"System(primary='{self.primary.name}', secondary='{self.secondary.name}', mu={self.mu:.4e})"

    def __repr__(self) -> str:
        return f"System(primary={self.primary!r}, secondary={self.secondary!r}, distance={self.distance})"

    @property
    def primary(self) -> Body:
        """Primary gravitating body."""
        return self._primary

    @property
    def secondary(self) -> Body:
        """Secondary gravitating body."""
        return self._secondary

    @property
    def distance(self) -> float:
        """Characteristic separation between the bodies in km."""
        return self._distance

    @property
    def mu(self) -> float:
        r"""Mass parameter :math:`\mu`."""
        return self._mu

    @property
    def libration_points(self) -> Dict[int, LibrationPoint]:
        """Mapping from integer identifiers {1,...,5} to libration point objects."""
        return self._libration_points
        
    @property
    def dynsys(self):
        """Underlying vector field instance."""
        return self._dynsys

    def _get_mu(self) -> float:
        r"""
        Compute the dimensionless mass parameter.

        Returns
        -------
        float
            Value of :math:`\mu`.

        Notes
        -----
        The calculation is performed in high precision using
        :pyfunc:`hiten.utils.precision.hp` to mitigate numerical cancellation when
        :math:`m_1 \approx m_2`.
        """
        logger.debug(f"Calculating mu: {self.secondary.mass} / ({self.primary.mass} + {self.secondary.mass})")

        # Use Number for critical mu calculation
        primary_mass_hp = hp(self.primary.mass)
        secondary_mass_hp = hp(self.secondary.mass)
        total_mass_hp = primary_mass_hp + secondary_mass_hp
        mu_hp = secondary_mass_hp / total_mass_hp

        mu = float(mu_hp) # Convert back to float for storage
        logger.debug(f"Calculated mu with high precision: {mu}")
        return mu

    def _compute_libration_points(self) -> Dict[int, LibrationPoint]:
        r"""
        Instantiate the five classical libration points.

        Returns
        -------
        dict[int, LibrationPoint]
            Mapping {1,...,5} to :pyclass:`hiten.system.libration.base.LibrationPoint`
            objects.
        """
        logger.debug(f"Computing Libration points for mu={self.mu}")
        points = {
            1: L1Point(self),
            2: L2Point(self),
            3: L3Point(self),
            4: L4Point(self),
            5: L5Point(self)
        }
        logger.debug(f"Finished computing Libration points.")
        return points

    def get_libration_point(self, index: int) -> LibrationPoint:
        r"""
        Access a pre-computed libration point.

        Parameters
        ----------
        index : int
            Identifier of the desired point in {1, 2, 3, 4, 5}.

        Returns
        -------
        LibrationPoint
            Requested libration point instance.

        Raises
        ------
        ValueError
            If *index* is not in the valid range.

        Examples
        --------
        >>> sys = System(primary, secondary, distance)
        >>> L1 = sys.get_libration_point(1)
        """
        point: Optional[LibrationPoint] = self.libration_points.get(index)
        if point is None:
            logger.error(f"Invalid Libration point index requested: {index}. Must be 1-5.")
            raise ValueError(f"Invalid Libration point index: {index}. Must be 1, 2, 3, 4, or 5.")
        logger.debug(f"Retrieving Libration point L{index}")
        return point

    def propagate(
        self,
        initial_conditions: Sequence[float],
        tf: float = 2 * np.pi,
        *,
        steps: int = 1000,
        method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy",
        order: int = 8,
        **kwargs
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        r"""
        Propagate arbitrary initial conditions in the CR3BP.

        This helper is a thin wrapper around
        :func:`hiten.algorithms.dynamics.rtbp._propagate_dynsys` that avoids
        the need to instantiate a :class:`~hiten.system.orbits.base.PeriodicOrbit`.

        Parameters
        ----------
        initial_conditions : Sequence[float]
            Six-element state vector ``[x, y, z, vx, vy, vz]`` expressed in
            canonical CR3BP units.
        steps : int, default 1000
            Number of output nodes in the returned trajectory.
        method : {"rk", "scipy", "symplectic", "adaptive"}, default "scipy"
            Integration backend to employ.
        order : int, default 8
            Formal order of the integrator when applicable (ignored by the
            SciPy backend).

        Returns
        -------
        tuple (times, states)
            *times* - 1-D array of shape ``(steps,)`` holding the sampling
            instants (radial-normalised units).
            *states* - 2-D array of shape ``(steps, 6)`` with the propagated
            trajectory.
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
        r"""
        Factory method to build a :class:`System` directly from body names.

        This helper retrieves the masses, radii and characteristic orbital
        distance of the selected primary/secondary pair from
        :pyclass:`hiten.utils.constants.Constants` and instantiates the
        corresponding :pyclass:`Body` objects before finally returning the
        fully-initialised :class:`System` instance.

        Parameters
        ----------
        primary_name : str
            Name of the primary body (case-insensitive, e.g. ``"earth"``).
        secondary_name : str
            Name of the secondary body orbiting the primary (e.g. ``"moon"``).

        Returns
        -------
        System
            Newly created CR3BP system.
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

    def __getstate__(self):
        """Custom state extractor to enable pickling.

        The underlying dynamical system instance stored in ``_dynsys`` often
        contains numba-compiled objects that cannot be serialised.  We exclude
        it from the pickled representation and recreate it automatically when
        the object is re-loaded.
        """
        state = self.__dict__.copy()
        # Remove the compiled dynamical system before pickling
        if "_dynsys" in state:
            state["_dynsys"] = None
        return state

    def __setstate__(self, state):
        """Restore the System instance after unpickling.

        The heavy, non-serialisable dynamical system is reconstructed lazily
        using the stored value of :pyattr:`mu` and the names of the primary and
        secondary bodies.
        """
        from hiten.algorithms.dynamics.rtbp import rtbp_dynsys

        # Restore the plain attributes
        self.__dict__.update(state)

        # Re-instantiate the CR3BP vector field that had been stripped during pickling
        if self.__dict__.get("_dynsys") is None:
            self._dynsys = rtbp_dynsys(self.mu, name=self.primary.name + "_" + self.secondary.name)
