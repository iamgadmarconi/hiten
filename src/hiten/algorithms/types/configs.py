"""Configuration classes for algorithm structure (compile-time).

These classes define WHAT algorithm you're using and HOW it's structured.
They should be set once when creating a pipeline and should NOT contain
runtime tuning parameters.

For runtime tuning parameters, see options.py.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal, Optional, Sequence

from hiten.utils.log_config import logger


class _HitenBaseConfig(ABC):
    """Base class for compile-time configuration objects.
    
    Config objects define the algorithm structure and are set once when
    creating a pipeline. They should NOT contain runtime tuning parameters
    or problem-specific data.
    
    Examples of what belongs in Config:
    - Algorithm structure (residual_indices, control_indices)
    - Method selection (finite_difference, integration method)
    - Problem definition (target values, event functions)
    
    Examples of what does NOT belong in Config:
    - Runtime tuning (tol, max_attempts, order)
    - Problem-specific data (dynsys, state0, t0, tf)
    - Computed functions (residual_fn, jacobian_fn)
    
    Use _HitenBaseOptions for runtime tuning parameters.
    """

    _version: float = 1.0

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self._validate()
        logger.debug(f"\n{self._display()}")

    def __getattr__(self, name: str):
        """Automatically flatten nested configs by searching for attributes.
        
        If an attribute is not found at the root level, this method searches
        through nested _HitenBaseConfig objects to find it. This allows
        transparent access to nested config parameters.
        
        Parameters
        ----------
        name : str
            Attribute name to search for.
        
        Returns
        -------
        Any
            The attribute value from a nested config.
        
        Raises
        ------
        AttributeError
            If the attribute is not found in this config or any nested configs.
        
        Examples
        --------
        >>> config = CorrectionConfig(
        ...     integration=IntegrationConfig(method="adaptive")
        ... )
        >>> config.method  # Automatically finds integration.method
        'adaptive'
        """
        # Avoid infinite recursion for __dataclass_fields__
        if name == '__dataclass_fields__':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Search through all fields that are _HitenBaseConfig instances
        if hasattr(self, '__dataclass_fields__'):
            for field_name in self.__dataclass_fields__:
                try:
                    field_value = object.__getattribute__(self, field_name)
                    # Check if this field is a nested config
                    if isinstance(field_value, _HitenBaseConfig):
                        # Try to get the attribute from the nested config
                        if hasattr(field_value, name):
                            return getattr(field_value, name)
                except AttributeError:
                    continue
        
        # If not found in any nested config, raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Searched nested configs but '{name}' was not found."
        )

    def _display(self) -> str:
        """Generate human-readable description."""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, _ in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            lines.append(f"  {field_name}: {value}")
        return "\n".join(lines)

    @abstractmethod
    def _validate(self) -> None:
        """Validate the configuration."""
        pass

    def merge(self, **overrides) -> "_HitenBaseConfig":
        """Create new config with specified overrides (immutable).
        
        Parameters
        ----------
        **overrides
            Fields to override. None values are ignored.
        
        Returns
        -------
        _HitenBaseConfig
            New config instance with overrides applied.
        
        Examples
        --------
        >>> config = MyConfig(tol=1e-10, max_attempts=50)
        >>> new_config = config.merge(tol=1e-12)
        >>> new_config.tol  # 1e-12
        >>> config.tol      # 1e-10 (unchanged)
        """
        filtered = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **filtered)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "_HitenBaseConfig":
        """Create configuration from dictionary."""
        import inspect
        sig = inspect.signature(cls)
        valid_keys = set(sig.parameters.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(frozen=True)
class IntegrationConfig(_HitenBaseConfig):
    """Integration method selection (compile-time).
    
    Defines which integration algorithm to use. Set once at pipeline creation.
    
    Parameters
    ----------
    method : {'fixed', 'adaptive', 'symplectic'}, default='adaptive'
        Integration method to use:
        - 'fixed': Fixed-step Runge-Kutta methods
        - 'symplectic': Symplectic integrators (preserves Hamiltonian structure)
        - 'adaptive': Adaptive step size methods
    forward : int, default=1
        Direction flag. Positive values integrate forward in time,
        negative values integrate backward.
    flip_indices : Optional[Sequence[int]], default=None
        Indices of state components whose derivatives should be negated
        when forward < 0. If None, all components are flipped.
    """
    method: Literal["fixed", "adaptive", "symplectic"] = "adaptive"
    forward: int = 1
    flip_indices: Optional[Sequence[int]] = None

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.method not in ["fixed", "adaptive", "symplectic"]:
            raise ValueError(f"Invalid integration method: {self.method}")
        if self.forward not in [-1, 1]:
            raise ValueError(f"Invalid direction flag: {self.forward}")


@dataclass(frozen=True)
class EventConfig(_HitenBaseConfig):
    """Event detection configuration (compile-time).
    
    Defines which event crossings to detect. Set once at pipeline creation.

    Parameters
    ----------
    direction : {-1, 0, 1}, default=0
        Crossing direction to detect:
        - 0: any sign change (g0 * g1 <= 0)
        - +1: only increasing crossings (g0 <= 0 and g1 >= 0)
        - -1: only decreasing crossings (g0 >= 0 and g1 <= 0)
    terminal : bool, default=True
        When True, integration should stop at the first event.
    """
    direction: Literal[-1, 0, 1] = 0
    terminal: bool = True

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.direction not in [-1, 0, 1]:
            raise ValueError(f"Invalid direction: {self.direction}")


@dataclass(frozen=True)
class NumericalConfig(_HitenBaseConfig):
    """Numerical method selection (compile-time).
    
    Defines which numerical methods to use for differentiation and line search.
    Set once at pipeline creation.
    
    Parameters
    ----------
    finite_difference : bool, default=False
        Whether to use finite difference for Jacobian computation.
        Useful for debugging, testing, or when analytic Jacobians are 
        suspected to be incorrect.
    line_search_enabled : bool, default=False
        Whether to enable line search. Line search improves robustness 
        for challenging problems at the cost of additional function evaluations.
    """
    finite_difference: bool = False
    line_search_enabled: bool = False

    def _validate(self) -> None:
        """Validate the configuration."""
        pass  # Boolean fields are always valid


@dataclass(frozen=True)
class RefineConfig(_HitenBaseConfig):
    """Refinement method configuration (compile-time).
    
    Defines which refinement algorithms to use. Set once at pipeline creation.
    
    Parameters
    ----------
    interp_kind : {'linear', 'cubic'}, default='cubic'
        Interpolation method to use for refinement.
    """
    interp_kind: Literal["linear", "cubic"] = "cubic"

    def _validate(self) -> None:
        """Validate the configuration."""
        if self.interp_kind not in ["linear", "cubic"]:
            raise ValueError(f"Invalid interpolation kind: {self.interp_kind}")


@dataclass(frozen=True)
class CorrectionConfig(_HitenBaseConfig):
    """Base configuration for correction algorithms (compile-time structure).

    Composes common configs from types/ with corrector-specific structure.
    These should be set once when creating a pipeline.

    Parameters
    ----------
    integration : :class:`~hiten.algorithms.types.configs.IntegrationConfig`, optional
        Integration method configuration (method, forward, flip_indices).
    numerical : :class:`~hiten.algorithms.types.configs.NumericalConfig`, optional
        Numerical method configuration (finite_difference, line_search_enabled).

    Notes
    -----
    For runtime tuning parameters like `tol`, `max_attempts`, `max_delta`, 
    `fd_step`, `order`, and `steps`, use CorrectionOptions instead.

    Examples
    --------
    >>> # Compile-time: Choose algorithm structure
    >>> config = CorrectionConfig(
    ...     integration=IntegrationConfig(method="adaptive"),
    ...     numerical=NumericalConfig(
    ...         finite_difference=False,
    ...         line_search_enabled=True
    ...     )
    ... )
    >>> # Runtime: Tune algorithm parameters per call
    >>> from hiten.algorithms.types.options import CorrectionOptions
    >>> options = CorrectionOptions()
    >>> tight_options = options.merge(
    ...     convergence=options.convergence.merge(tol=1e-14)
    ... )
    """
    integration: IntegrationConfig = IntegrationConfig()
    numerical: NumericalConfig = NumericalConfig()

    def _validate(self) -> None:
        """Validate the configuration."""
        # Nested configs validate themselves in __post_init__
        pass


@dataclass(frozen=True)
class SectionConfig(_HitenBaseConfig):
    """Poincare map section configuration (compile-time).
    
    Defines the Poincare section geometry. Set once at pipeline creation.
    
    Parameters
    ----------
    section_axis : str
        Which coordinate defines the section (e.g., 'x', 'y', 'z').
    section_offset : float
        Where the section is located along the section_axis.
    plane_coords : tuple[str, str]
        Which coordinates define the plane (e.g., ('y', 'z')).
    direction : {-1, 0, 1} or None, default=None
        Crossing direction to detect:
        - None: both directions
        - +1: positive crossings
        - -1: negative crossings
    """
    section_axis: str
    section_offset: float
    plane_coords: tuple[str, str]
    direction: Optional[Literal[-1, 1]] = None

    def _validate(self) -> None:
        """Validate the configuration."""
        if len(self.plane_coords) != 2:
            raise ValueError(f"plane_coords must have exactly 2 elements, got {len(self.plane_coords)}")
        
        if self.section_axis in self.plane_coords:
            raise ValueError(
                f"section_axis '{self.section_axis}' cannot be in plane_coords {self.plane_coords}"
            )
        
        if self.direction is not None and self.direction not in [-1, 1]:
            raise ValueError(f"direction must be -1, 1, or None, got {self.direction}")
