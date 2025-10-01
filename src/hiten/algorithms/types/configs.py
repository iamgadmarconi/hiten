from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from hiten.algorithms.types.core import _HitenBaseConfig
from hiten.algorithms.dynamics.protocols import _DynamicalSystemProtocol


@dataclass(frozen=True)
class _IntegrationConfig(_HitenBaseConfig):
    """Configuration for numerical integration parameters.
    
    This configuration class defines the parameters for numerical integration
    used by propagating backends and engines for numerical integration of trajectories.

    Parameters
    ----------
    dynsys : :class:`~hiten.algorithms.dynamics.protocols._DynamicalSystemProtocol`
        Dynamical system to integrate.
    state0 : Sequence[float]
        Initial state vector.
    t0 : float
        Initial time.
    tf : float
        Final time.
    dt : float, default=1e-2
        Integration time step (nondimensional units). Smaller
        values provide higher accuracy but require more computation.
        Ignored in adaptive integration.
    method : {'fixed', 'adaptive', 'symplectic'}, default='fixed'
        Integration method to use:
        - 'fixed': Fixed-step Runge-Kutta methods
        - 'symplectic': Symplectic integrators (preserves Hamiltonian structure)
        - 'adaptive': Adaptive step size methods
    order : int, default=8
        Integration order for Runge-Kutta methods. Higher orders
        provide better accuracy but require more function evaluations.
    c_omega_heuristic : float, default=20.0
        Heuristic parameter for symplectic integration, controlling
        the relationship between step size and frequency content.
        Ignored in fixed and adaptive integration.
    max_steps : int, default=2000
        Maximum number of integration steps.
    forward : int, default=1
        Direction flag. Positive values integrate forward in time,
        negative values integrate backward.
    flip_indices : Optional[Sequence[int]], default=None
        Indices of state components whose derivatives should be negated
        when fwd < 0. If None, all components are flipped.
    event_fn : Optional[Callable[[float, Sequence[float]], float]], default=None
        Event function to use for event detection.
    event_cfg : Optional[:class:`~hiten.algorithms.types.events._EventConfig`], default=None
        Event configuration to use for event detection.
    """
    dynsys: _DynamicalSystemProtocol
    state0: Sequence[float]
    t0: float
    tf: float
    dt: float = 1e-2
    method: Literal["fixed", "adaptive", "symplectic"] = "fixed"
    order: int = 8
    c_omega_heuristic: Optional[float] = 20.0
    max_steps: int = 2000
    forward: int = 1
    flip_indices: Optional[Sequence[int]] = None
    event_fn: Optional[Callable[[float, Sequence[float]], float]] = None
    event_cfg: Optional[_EventConfig] = None