"""Configuration classes for Poincare return map implementations.

This module provides configuration dataclasses and protocols for
Poincare return map computations. It defines the configuration
interface that all return map implementations must support.

The configuration system provides a flexible way to tune these
parameters for different dynamical systems and computational
requirements.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable


@dataclass(frozen=True)
class _ReturnMapBaseConfig(ABC):
    """Base configuration for Poincare return map implementations.

    This abstract base class defines the minimal configuration
    parameters that are universally applicable to all return map
    implementations. It contains only orchestration fields that
    control the overall behavior of the computation.

    Parameters
    ----------
    compute_on_init : bool, default=False
        Whether to automatically compute the return map upon
        initialization. If True, the map is computed immediately
        after object creation.
    n_workers : int or None, default=None
        Number of parallel workers for computation. If None,
        uses the default number of workers (typically the number
        of CPU cores).  

    Notes
    -----
    This class serves as the base for all return map configurations.
    Concrete implementations should inherit from this class and
    add their specific configuration parameters.

    All time units are in nondimensional units unless otherwise
    specified.
    """
    compute_on_init: bool = False
    n_workers: int | None = None


@dataclass(frozen=True)
class _IntegrationConfig(ABC):
    """Configuration for numerical integration parameters.

    This abstract base class defines the integration-related
    configuration parameters used by propagating backends and
    engines for numerical integration of trajectories.

    Parameters
    ----------
    dt : float, default=1e-2
        Integration time step (nondimensional units). Smaller
        values provide higher accuracy but require more computation.
    method : {'fixed', 'adaptive', 'symplectic'}, default='fixed'
        Integration method to use:
        - 'fixed': Fixed-step Runge-Kutta methods
        - 'symplectic': Symplectic integrators (preserves Hamiltonian structure)
        - 'adaptive': Adaptive step size methods
    order : int, default=8
        Integration order for Runge-Kutta methods. Higher orders
        provide better accuracy but require more function evaluations.
    c_omega_heuristic : float, default=20.0
        Heuristic parameter for adaptive integration, controlling
        the relationship between step size and frequency content.

    Notes
    -----
    All time units are in nondimensional units unless otherwise
    specified.
    """
    dt: float = 1e-2
    method: Literal["fixed", "adaptive", "symplectic"] = "fixed"
    order: int = 8
    c_omega_heuristic: Optional[float] = 20.0


@dataclass(frozen=True)
class _IterationConfig(ABC):
    """Configuration for iteration control in return map computation.

    This abstract base class defines the iteration-related
    configuration parameters that control how many return map
    steps are computed for each initial condition.

    Parameters
    ----------
    n_iter : int, default=40
        Number of return map iterations to compute for each
        initial condition. More iterations provide a more
        complete picture of the dynamics but require more
        computation.

    Notes
    -----
    The number of iterations determines how many times each
    trajectory is mapped back to the section. This parameter
    should be chosen based on the desired resolution of the
    return map and the computational resources available.

    For chaotic systems, more iterations may be needed to
    reveal the full structure of the attractor.
    """
    n_iter: int = 40


@dataclass(frozen=True)
class _SeedingConfig(ABC):
    """Configuration for seeding strategies in return map computation.

    This abstract base class defines the seeding-related
    configuration parameters that control how initial conditions
    are generated for return map computation.

    Parameters
    ----------
    n_seeds : int, default=20
        Number of initial seeds to generate for return map
        computation. More seeds provide better coverage of
        the section plane but require more computation.

    Notes
    -----
    The seeding strategy determines how initial conditions
    are distributed on the section plane. The number of seeds
    affects the resolution and coverage of the computed return
    map. More seeds provide better statistical coverage but
    increase computational cost.

    The choice of seeding strategy depends on the specific
    dynamical system and the desired analysis goals.
    """
    n_seeds: int = 20


# Backward-compatible umbrella config combining all mixins
@dataclass(frozen=True)
class _ReturnMapConfig(_ReturnMapBaseConfig, _IntegrationConfig, _IterationConfig, _SeedingConfig):
    """Complete configuration for Poincare return map computation.

    This class combines all configuration mixins into a single
    comprehensive configuration class. It inherits from all
    the abstract configuration base classes, providing a
    complete set of parameters for return map computation.

    This class serves as a backward-compatible umbrella
    configuration that includes all available parameters
    for return map computation.

    Notes
    -----
    This class inherits all parameters from:
    - :class:`~hiten.algorithms.poincare.core.config._ReturnMapBaseConfig`: Base orchestration parameters
    - :class:`~hiten.algorithms.poincare.core.config._IntegrationConfig`: Numerical integration parameters
    - :class:`~hiten.algorithms.poincare.core.config._IterationConfig`: Iteration control parameters
    - :class:`~hiten.algorithms.poincare.core.config._SeedingConfig`: Seeding strategy parameters

    All time units are in nondimensional units unless otherwise
    specified.
    """
    pass


class _SectionConfig(ABC):
    """Abstract base class for Poincare section configuration.

    This abstract base class defines the interface for section
    configuration objects that specify the section coordinate
    and plane coordinates for Poincare map computation.

    Parameters
    ----------
    section_coord : str
        The coordinate that defines the section (e.g., "q2", "p2").
        This is the coordinate that is held constant on the section.
    plane_coords : tuple[str, str]
        Tuple of two coordinate labels that define the section plane
        (e.g., ("q2", "p2")). These are the coordinates that vary
        in the section plane.

    Notes
    -----
    This class serves as the base for all section configuration
    implementations. Concrete implementations should inherit from
    this class and add their specific section parameters.

    The section coordinate determines which coordinate is held
    constant, while the plane coordinates determine which two
    coordinates are used to represent points in the section.
    """
    section_coord: str
    plane_coords: Tuple[str, str]


@runtime_checkable
class _EngineConfigLike(Protocol):
    """Protocol for engine configuration objects.

    This protocol defines the interface that engine configuration
    objects must implement. It specifies the minimum set of
    attributes required for engine configuration.

    Attributes
    ----------
    dt : float
        Integration time step (nondimensional units).
    n_iter : int
        Number of return map iterations to compute.
    n_workers : int or None
        Number of parallel workers for computation.

    Notes
    -----
    This protocol is used for type checking and runtime validation
    of engine configuration objects. Any object that implements
    these attributes can be used as an engine configuration.

    The protocol ensures that engine configurations have the
    necessary parameters for numerical integration and iteration
    control.
    """
    dt: float
    n_iter: int
    n_workers: int | None


@runtime_checkable
class _SeedingConfigLike(Protocol):
    """Protocol for seeding configuration objects.

    This protocol defines the interface that seeding configuration
    objects must implement. It specifies the minimum set of
    attributes required for seeding configuration.

    Attributes
    ----------
    n_seeds : int
        Number of initial seeds to generate for return map
        computation.

    Notes
    -----
    This protocol is used for type checking and runtime validation
    of seeding configuration objects. Any object that implements
    this attribute can be used as a seeding configuration.

    The protocol ensures that seeding configurations have the
    necessary parameters for generating initial conditions.
    """
    n_seeds: int
