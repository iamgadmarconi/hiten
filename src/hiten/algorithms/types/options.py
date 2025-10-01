"""Options classes for runtime tuning parameters.

These classes define HOW WELL the algorithm runs and can vary between
method calls without changing the algorithm structure.

For algorithm structure parameters, see configs.py.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Optional

from hiten.utils.log_config import logger


class _HitenBaseOptions(ABC):
    """Base class for runtime options objects.
    
    Options objects contain runtime tuning parameters that can vary between
    method calls without changing the algorithm structure. They should NOT
    contain algorithm structure or problem definitions.
    
    Examples of what belongs in Options:
    - Convergence tuning (tol, max_attempts, max_delta)
    - Integration accuracy (order, steps, dt)
    - Refinement accuracy (segment_refine, tolerances)
    - Resource allocation (n_workers)
    
    Examples of what does NOT belong in Options:
    - Algorithm structure (residual_indices, control_indices)
    - Problem definition (target, event_func)
    - Problem data (dynsys, state0)
    
    Use _HitenBaseConfig for algorithm structure parameters.
    """

    _version: float = 1.0

    def __post_init__(self) -> None:
        """Post-initialization hook."""
        self._validate()
        logger.debug(f"\n{self._display()}")

    def __getattr__(self, name: str):
        """Automatically flatten nested options by searching for attributes.
        
        If an attribute is not found at the root level, this method searches
        through nested _HitenBaseOptions objects to find it. This allows
        transparent access to nested option parameters.
        
        Parameters
        ----------
        name : str
            Attribute name to search for.
        
        Returns
        -------
        Any
            The attribute value from a nested option.
        
        Raises
        ------
        AttributeError
            If the attribute is not found in this option or any nested options.
        
        Examples
        --------
        >>> options = CorrectionOptions(
        ...     integration=IntegrationOptions(order=10)
        ... )
        >>> options.order  # Automatically finds integration.order
        10
        """
        # Avoid infinite recursion for __dataclass_fields__
        if name == '__dataclass_fields__':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Search through all fields that are _HitenBaseOptions instances
        if hasattr(self, '__dataclass_fields__'):
            for field_name in self.__dataclass_fields__:
                try:
                    field_value = object.__getattribute__(self, field_name)
                    # Check if this field is a nested option
                    if isinstance(field_value, _HitenBaseOptions):
                        # Try to get the attribute from the nested option
                        if hasattr(field_value, name):
                            return getattr(field_value, name)
                except AttributeError:
                    continue
        
        # If not found in any nested option, raise AttributeError
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Searched nested options but '{name}' was not found."
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
        """Validate the options."""
        pass

    def merge(self, *args, **overrides) -> "_HitenBaseOptions":
        """Create a new options instance with overrides applied (immutable).

        Supports both direct field overrides and recursive updates using
        dotted keys ("group.subfield=value") or double-underscore keys
        ("group__subfield=value").

        Notes
        -----
        - None values are ignored (kept for backward compatibility).
        - Unknown fields raise a KeyError.

        Examples
        --------
        >>> opts = MyOptions(tol=1e-10)
        >>> opts2 = opts.merge(tol=1e-12)
        >>> # Recursive update into nested options
        >>> corr = CorrectionOptions()
        >>> corr2 = corr.merge("convergence.tol"=1e-14)  # doctest: +SKIP
        >>> corr3 = corr.merge(**{"convergence.tol": 1e-14})
        >>> corr4 = corr.merge(convergence__tol=1e-14)
        """

        # Allow an optional first positional mapping argument for ergonomics
        mapping_arg: Mapping[str, Any] | None = None
        if len(args) > 1:
            raise TypeError("merge() accepts at most one positional mapping argument")
        if len(args) == 1:
            if not isinstance(args[0], Mapping):
                raise TypeError("merge() positional argument must be a mapping of overrides")
            mapping_arg = args[0]

        # Combine mapping arg and kwargs, ignoring None values at leaves
        combined: dict[str, Any] = {}
        if mapping_arg is not None:
            for k, v in mapping_arg.items():
                if v is not None:
                    combined[k] = v
        for k, v in overrides.items():
            if v is not None:
                combined[k] = v

        if not combined:
            return self

        # Build a nested update dict from dotted or double-underscore keys
        nested_updates: dict[str, Any] = {}

        def assign_nested(target: dict, parts: list[str], value: Any) -> None:
            if not parts:
                return
            head, *tail = parts
            if not tail:
                # Leaf assignment; if there is an existing dict and incoming value
                # is also a dict, perform a shallow merge to allow multiple sources
                if isinstance(target.get(head), dict) and isinstance(value, dict):
                    target[head].update(value)  # type: ignore[arg-type]
                else:
                    target[head] = value
                return
            # Intermediate node
            if head not in target or not isinstance(target[head], dict):
                target[head] = {}
            assign_nested(target[head], tail, value)  # type: ignore[arg-type]

        for key, value in combined.items():
            if isinstance(key, str) and ('.' in key or '__' in key):
                sep = '.' if '.' in key else '__'
                parts = [p for p in key.split(sep) if p]
                assign_nested(nested_updates, parts, value)
            else:
                # Treat as a top-level assignment
                nested_updates[key] = value

        # Apply nested updates recursively using dataclass replace()
        def apply_updates(obj: "_HitenBaseOptions", updates: Mapping[str, Any]) -> "_HitenBaseOptions":
            if not hasattr(obj, "__dataclass_fields__"):
                raise TypeError(f"{type(obj).__name__} is not a dataclass-based options object")

            replacements: dict[str, Any] = {}
            for field_name, update_value in updates.items():
                if field_name not in obj.__dataclass_fields__:
                    raise KeyError(
                        f"Unknown options field '{field_name}' for {type(obj).__name__}"
                    )
                current_value = getattr(obj, field_name)
                # Recurse into nested options groups
                if isinstance(update_value, Mapping) and isinstance(current_value, _HitenBaseOptions):
                    replacements[field_name] = apply_updates(current_value, update_value)
                else:
                    replacements[field_name] = update_value
            return replace(obj, **replacements)

        return apply_updates(self, nested_updates)
    
    def to_dict(self) -> dict:
        """Convert options to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "_HitenBaseOptions":
        """Create options from dictionary."""
        import inspect
        sig = inspect.signature(cls)
        valid_keys = set(sig.parameters.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)



@dataclass(frozen=True)
class IntegrationOptions(_HitenBaseOptions):
    """Integration runtime tuning options.
    
    These parameters control integration accuracy and can vary per call.
    They tune HOW WELL the integration runs, not WHAT integration method is used.
    
    Parameters
    ----------
    dt : float, default=1e-2
        Integration time step (nondimensional units).
        Smaller values provide higher accuracy but require more computation.
        Ignored in adaptive integration.
    order : int, default=8
        Integration order for Runge-Kutta methods.
        Higher orders provide better accuracy but require more function evaluations.
    max_steps : int, default=2000
        Maximum number of integration steps.
    c_omega_heuristic : float, default=20.0
        Heuristic parameter for symplectic integration, controlling
        the relationship between step size and frequency content.
        Ignored in fixed and adaptive integration.
    steps : int, default=500
        Number of integration steps for correction algorithms.
    """
    _version: float = 1.0

    dt: float = 1e-2
    order: int = 8
    max_steps: int = 2000
    c_omega_heuristic: float = 20.0
    steps: int = 500

    def _validate(self) -> None:
        """Validate the options."""
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.order <= 0:
            raise ValueError(f"order must be positive, got {self.order}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")


@dataclass(frozen=True)
class EventOptions(_HitenBaseOptions):
    """Event detection runtime tuning options.
    
    These parameters control event detection accuracy and can vary per call.
    
    Parameters
    ----------
    xtol : float, default=1e-12
        Absolute time tolerance for in-step event refinement.
    gtol : float, default=1e-12
        Absolute function tolerance: stop refinement when |g| <= gtol.
    """
    _version: float = 1.0

    xtol: float = 1e-12
    gtol: float = 1e-12

    def _validate(self) -> None:
        """Validate the options."""
        if self.xtol <= 0:
            raise ValueError(f"xtol must be positive, got {self.xtol}")
        if self.gtol <= 0:
            raise ValueError(f"gtol must be positive, got {self.gtol}")


@dataclass(frozen=True)
class ConvergenceOptions(_HitenBaseOptions):
    """Convergence runtime tuning options.
    
    These parameters control iteration convergence and can vary per call.
    They tune HOW WELL the algorithm converges, not WHAT algorithm is used.
    
    Parameters
    ----------
    max_attempts : int, default=50
        Maximum number of iterations to attempt before declaring convergence failure.
        This prevents infinite loops in cases where the algorithm fails to converge.
    tol : float, default=1e-12
        Convergence tolerance for the residual norm. The algorithm terminates
        successfully when the norm of the residual falls below this value.
    max_delta : float, default=1e-2
        Maximum allowed infinity norm of Newton steps.
        Serves as a safeguard against excessively large steps.
    """
    _version: float = 1.0

    max_attempts: int = 50
    tol: float = 1e-12
    max_delta: float = 1e-2

    def _validate(self) -> None:
        """Validate the options."""
        if self.max_attempts <= 0:
            raise ValueError(f"max_attempts must be positive, got {self.max_attempts}")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.max_delta <= 0:
            raise ValueError(f"max_delta must be positive, got {self.max_delta}")


@dataclass(frozen=True)
class NumericalOptions(_HitenBaseOptions):
    """Numerical method runtime tuning options.
    
    These parameters tune numerical methods and can vary per call.
    
    Parameters
    ----------
    fd_step : float, default=1e-8
        Finite difference step size (when finite_difference=True in config).
    line_search_alpha_reduction : float, default=0.5
        Factor to reduce step size in backtracking (when line_search_enabled=True).
    line_search_min_alpha : float, default=1e-4
        Minimum step size before giving up (when line_search_enabled=True).
    line_search_armijo_c : float, default=0.1
        Armijo parameter for sufficient decrease condition (when line_search_enabled=True).
    """
    _version: float = 1.0

    fd_step: float = 1e-8
    line_search_alpha_reduction: float = 0.5
    line_search_min_alpha: float = 1e-4
    line_search_armijo_c: float = 0.1

    def _validate(self) -> None:
        """Validate the options."""
        if self.fd_step <= 0:
            raise ValueError(f"fd_step must be positive, got {self.fd_step}")
        if self.line_search_alpha_reduction <= 0 or self.line_search_alpha_reduction >= 1:
            raise ValueError(
                f"line_search_alpha_reduction must be in (0, 1), got {self.line_search_alpha_reduction}"
            )
        if self.line_search_min_alpha <= 0:
            raise ValueError(f"line_search_min_alpha must be positive, got {self.line_search_min_alpha}")
        if self.line_search_armijo_c <= 0:
            raise ValueError(f"line_search_armijo_c must be positive, got {self.line_search_armijo_c}")


@dataclass(frozen=True)
class RefineOptions(_HitenBaseOptions):
    """Refinement runtime tuning options.
    
    These parameters control refinement accuracy and can vary per call.
    
    Parameters
    ----------
    segment_refine : int, default=20
        Number of segments to refine.
    tol_on_surface : float, default=1e-12
        Tolerance for considering a point to be on the surface.
    dedup_time_tol : float, default=1e-9
        Time tolerance for deduplicating nearby crossings.
    dedup_point_tol : float, default=1e-12
        Point tolerance for deduplicating nearby crossings.
    max_hits_per_traj : int or None, default=None
        Maximum number of hits per trajectory.
    newton_max_iter : int, default=25
        Maximum number of Newton iterations for refinement.
    """
    _version: float = 1.0

    segment_refine: int = 20
    tol_on_surface: float = 1e-12
    dedup_time_tol: float = 1e-9
    dedup_point_tol: float = 1e-12
    max_hits_per_traj: Optional[int] = None
    newton_max_iter: int = 25

    def _validate(self) -> None:
        """Validate the options."""
        if self.segment_refine <= 0:
            raise ValueError(f"segment_refine must be positive, got {self.segment_refine}")
        if self.tol_on_surface <= 0:
            raise ValueError(f"tol_on_surface must be positive, got {self.tol_on_surface}")
        if self.dedup_time_tol <= 0:
            raise ValueError(f"dedup_time_tol must be positive, got {self.dedup_time_tol}")
        if self.dedup_point_tol <= 0:
            raise ValueError(f"dedup_point_tol must be positive, got {self.dedup_point_tol}")
        if self.max_hits_per_traj is not None and self.max_hits_per_traj <= 0:
            raise ValueError(f"max_hits_per_traj must be positive, got {self.max_hits_per_traj}")
        if self.newton_max_iter <= 0:
            raise ValueError(f"newton_max_iter must be positive, got {self.newton_max_iter}")


@dataclass(frozen=True)
class WorkerOptions(_HitenBaseOptions):
    """Worker resource allocation options.
    
    These parameters control computational resources and can vary per call.
    
    Parameters
    ----------
    n_workers : int, default=1
        Number of parallel workers to use in the engine.
    """
    _version: float = 1.0

    n_workers: int = 1

    def _validate(self) -> None:
        """Validate the options."""
        if self.n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {self.n_workers}")


@dataclass(frozen=True)
class CorrectionOptions(_HitenBaseOptions):
    """Combined correction algorithm runtime options.
    
    Combines all runtime tuning options commonly used in correction algorithms.
    This is a convenience class that includes integration, convergence, and numerical options.
    
    Parameters
    ----------
    integration : IntegrationOptions, optional
        Integration tuning parameters.
    convergence : ConvergenceOptions, optional
        Convergence tuning parameters.
    numerical : NumericalOptions, optional
        Numerical method tuning parameters.
    
    Notes
    -----
    Individual option groups can be accessed and overridden independently:
    
    >>> opts = CorrectionOptions()
    >>> tight_opts = opts.merge(convergence=opts.convergence.merge(tol=1e-14))
    """
    _version: float = 1.0

    # Nested option groups
    integration: IntegrationOptions = IntegrationOptions()
    convergence: ConvergenceOptions = ConvergenceOptions()
    numerical: NumericalOptions = NumericalOptions()

    def _validate(self) -> None:
        """Validate the options."""
        # Nested options validate themselves in __post_init__
        pass


@dataclass(frozen=True)
class PoincareMapOptions(_HitenBaseOptions):
    """Combined Poincare map runtime options.
    
    Combines all runtime tuning options commonly used in Poincare map algorithms.
    
    Parameters
    ----------
    integration : IntegrationOptions, optional
        Integration tuning parameters.
    refine : RefineOptions, optional
        Refinement tuning parameters.
    workers : WorkerOptions, optional
        Worker allocation parameters.
    """
    _version: float = 1.0

    integration: IntegrationOptions = IntegrationOptions()
    refine: RefineOptions = RefineOptions()
    workers: WorkerOptions = WorkerOptions()

    def _validate(self) -> None:
        """Validate the options."""
        # Nested options validate themselves in __post_init__
        pass
