"""Configuration for center manifold Poincare sections in the CR3BP.

This module provides configuration classes for computing Poincare sections
restricted to center manifolds of collinear libration points in the Circular
Restricted Three-Body Problem (CR3BP).
"""
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from hiten.algorithms.poincare.core.config import (_IntegrationConfig,
                                                   _IterationConfig,
                                                   _ReturnMapBaseConfig,
                                                   _SectionConfig,
                                                   _SeedingConfig)
from hiten.algorithms.utils.exceptions import EngineError
from hiten.utils.log_config import logger


@dataclass(frozen=True)
class _CenterManifoldMapConfig(_ReturnMapBaseConfig, _IntegrationConfig, _IterationConfig, _SeedingConfig):
    """Configuration for center manifold Poincare maps.

    This dataclass combines configuration from multiple base classes to provide
    comprehensive settings for center manifold map computation, including
    integration parameters, seeding strategies, and iteration controls.

    Parameters
    ----------
    seed_strategy : {'single', 'axis_aligned', 'level_sets', 'radial', 'random'}, default='axis_aligned'
        Strategy for generating initial conditions on the center manifold.
        - 'single': Single axis seeding along one coordinate direction
        - 'axis_aligned': Seeding aligned with coordinate axes
        - 'level_sets': Seeding based on level sets of the Hamiltonian
        - 'radial': Radial seeding pattern from the periodic orbit
        - 'random': Random seeding within specified bounds
    seed_axis : {'q2', 'p2', 'q3', 'p3'}, optional
        Coordinate axis for single-axis seeding strategy. Required when
        seed_strategy='single', ignored otherwise.
    section_coord : {'q2', 'p2', 'q3', 'p3'}, default='q3'
        Coordinate defining the Poincare section (set to zero).

    Notes
    -----
    The configuration inherits from multiple base classes:
    - :class:`~hiten.algorithms.poincare.core.config._ReturnMapBaseConfig`: Basic return map settings
    - :class:`~hiten.algorithms.poincare.core.config._IntegrationConfig`: Integration method and parameters
    - :class:`~hiten.algorithms.poincare.core.config._IterationConfig`: Iteration control parameters
    - :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`: Seeding strategy parameters

    All coordinates are in nondimensional units with the primary-secondary
    separation as the length unit.
    """

    seed_strategy: Literal[
        "single",
        "axis_aligned",
        "level_sets",
        "radial",
        "random",
    ] = "axis_aligned"

    seed_axis: Optional[Literal["q2", "p2", "q3", "p3"]] = None
    section_coord: Literal["q2", "p2", "q3", "p3"] = "q3"

    def __post_init__(self):
        if self.seed_strategy == "single" and self.seed_axis is None:
            raise EngineError("seed_axis must be specified when seed_strategy is 'single'")
        if self.seed_strategy != "single" and self.seed_axis is not None:
            logger.warning("seed_axis is ignored when seed_strategy is not 'single'")


class _CenterManifoldSectionConfig(_SectionConfig):
    """Configuration for center manifold Poincare sections.

    This class provides configuration data for Poincare sections defined by
    setting one coordinate to zero in the center manifold phase space.
    It includes mappings between coordinate names and indices, and methods
    for extracting and building state vectors.

    Parameters
    ----------
    section_coord : str
        Coordinate defining the section ('q2', 'p2', 'q3', or 'p3').

    Attributes
    ----------
    section_coord : str
        The coordinate defining the section.
    section_index : int
        Index of the section coordinate in the state vector.
    section_value : float
        Value of the section coordinate (always 0.0).
    plane_coords : tuple[str, str]
        Coordinates spanning the section plane.
    plane_indices : tuple[int, int]
        Indices of the plane coordinates in the state vector.
    missing_coord : str
        Coordinate that must be solved for on the section.
    missing_index : int
        Index of the missing coordinate in the state vector.
    momentum_check_index : int
        Index for momentum direction checking.
    momentum_check_sign : float
        Sign for momentum direction checking.
    deriv_index : int
        Index for derivative computation.
    other_coords : tuple[str, str]
        The two coordinates not in the plane.
    other_indices : tuple[int, int]
        Indices of the other coordinates in the state vector.

    Notes
    -----
    The state vector is ordered as [q1, q2, q3, p1, p2, p3] in the rotating
    synodic frame. For center manifold trajectories, q1=0, so the effective
    state vector is [q2, q3, p2, p3].

    The four supported section types are:
    - q3=0: Plane spanned by (q2, p2), solve for p3
    - p3=0: Plane spanned by (q2, p2), solve for q3  
    - q2=0: Plane spanned by (q3, p3), solve for p2
    - p2=0: Plane spanned by (q3, p3), solve for q2
    """

    _TABLE: dict[str, dict[str, object]] = {
        "q3": dict(
            section_index=2, section_value=0.0,
            plane_coords=("q2", "p2"), plane_indices=(1, 4),
            missing_coord="p3", missing_index=5,
            momentum_check_index=5, momentum_check_sign=+1.0,
            deriv_index=2,
            other_coords=("q3", "p3"), other_indices=(2, 5),
        ),
        "p3": dict(
            section_index=5, section_value=0.0,
            plane_coords=("q2", "p2"), plane_indices=(1, 4),
            missing_coord="q3", missing_index=2,
            momentum_check_index=2, momentum_check_sign=+1.0,
            deriv_index=5,
            other_coords=("q3", "p3"), other_indices=(2, 5),
        ),
        "q2": dict(
            section_index=1, section_value=0.0,
            plane_coords=("q3", "p3"), plane_indices=(2, 5),
            missing_coord="p2", missing_index=4,
            momentum_check_index=4, momentum_check_sign=+1.0,
            deriv_index=1,
            other_coords=("q2", "p2"), other_indices=(1, 4),
        ),
        "p2": dict(
            section_index=4, section_value=0.0,
            plane_coords=("q3", "p3"), plane_indices=(2, 5),
            missing_coord="q2", missing_index=1,
            momentum_check_index=1, momentum_check_sign=+1.0,
            deriv_index=4,
            other_coords=("q2", "p2"), other_indices=(1, 4),
        ),
    }

    def __init__(self, section_coord: str) -> None:
        try:
            cfg = self._TABLE[section_coord]
        except KeyError as exc:
            raise EngineError(f"Unsupported section_coord: {section_coord}") from exc

        # copy into attributes (they are read-only by convention)
        self.section_coord: str = section_coord
        for k, v in cfg.items():
            setattr(self, k, v)

    def get_section_value(self, state: np.ndarray) -> float:
        """Get the value of the section coordinate from a state vector.

        Parameters
        ----------
        state : ndarray, shape (6,)
            State vector [q1, q2, q3, p1, p2, p3].

        Returns
        -------
        float
            Value of the section coordinate.
        """
        return float(state[self.section_index])

    def extract_plane_coords(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract the plane coordinates from a state vector.

        Parameters
        ----------
        state : ndarray, shape (6,)
            State vector [q1, q2, q3, p1, p2, p3].

        Returns
        -------
        tuple[float, float]
            The two coordinates spanning the section plane.
        """
        i, j = self.plane_indices 
        return float(state[i]), float(state[j])

    def extract_other_coords(self, state: np.ndarray) -> Tuple[float, float]:
        """Extract the other coordinates from a state vector.

        Parameters
        ----------
        state : ndarray, shape (6,)
            State vector [q1, q2, q3, p1, p2, p3].

        Returns
        -------
        tuple[float, float]
            The two coordinates not in the section plane.
        """
        i, j = self.other_indices
        return float(state[i]), float(state[j])

    def build_state(
        self,
        plane_vals: Tuple[float, float],
        other_vals: Tuple[float, float],
    ) -> Tuple[float, float, float, float]:
        """Build a center manifold state vector from coordinate values.

        Parameters
        ----------
        plane_vals : tuple[float, float]
            Values for the plane coordinates.
        other_vals : tuple[float, float]
            Values for the other coordinates.

        Returns
        -------
        tuple[float, float, float, float]
            Center manifold state (q2, p2, q3, p3) with the section
            coordinate set to its section value.

        Notes
        -----
        The section coordinate is automatically set to its section value
        (typically 0.0) regardless of the input values.
        """
        q2 = p2 = q3 = p3 = 0.0
        if self.plane_coords == ("q2", "p2"):
            q2, p2 = plane_vals
            q3, p3 = other_vals
        else:
            q3, p3 = plane_vals
            q2, p2 = other_vals

        if self.section_coord == "q2":
            q2 = self.section_value
        elif self.section_coord == "p2":    
            p2 = self.section_value
        elif self.section_coord == "q3":
            q3 = self.section_value
        else:  # "p3"
            p3 = self.section_value
        return q2, p2, q3, p3

    def build_constraint_dict(self, **kwargs) -> dict[str, float]:
        """Build a constraint dictionary for root finding.

        Parameters
        ----------
        **kwargs
            Additional coordinate values to include in the constraints.

        Returns
        -------
        dict[str, float]
            Dictionary mapping coordinate names to their values, including
            the section coordinate set to its section value.

        Notes
        -----
        The section coordinate is automatically included with its section
        value. Only valid coordinate names ('q1', 'q2', 'q3', 'p1', 'p2', 'p3')
        from the keyword arguments are included in the output.
        """
        out: dict[str, float] = {self.section_coord: self.section_value} 
        for k, v in kwargs.items():
            if k in {"q1", "q2", "q3", "p1", "p2", "p3"}:
                out[k] = float(v)
        return out


_SECTION_CACHE: dict[str, _CenterManifoldSectionConfig] = {
    name: _CenterManifoldSectionConfig(name) for name in ("q2", "p2", "q3", "p3")
}

def _get_section_config(section_coord: str) -> _CenterManifoldSectionConfig:
    """Get a cached section configuration for the specified coordinate.

    Parameters
    ----------
    section_coord : str
        Section coordinate identifier ('q2', 'p2', 'q3', or 'p3').

    Returns
    -------
    :class:`~hiten.algorithms.poincare.centermanifold.config._CenterManifoldSectionConfig`
        Cached configuration for the specified section coordinate.

    Raises
    ------
    ValueError
        If section_coord is not one of the supported values.

    Notes
    -----
    This function uses a cache to avoid recreating configuration objects
    for the same section coordinate. The cache is populated at module
    import time with all four supported section types.
    """
    try:
        return _SECTION_CACHE[section_coord]
    except KeyError as exc:
        raise EngineError(f"Unsupported section_coord: {section_coord}") from exc
