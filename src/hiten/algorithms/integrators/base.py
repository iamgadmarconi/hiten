"""Provide abstract interfaces for numerical time integration.

References
----------
Hairer, E., Norsett, S. P., & Wanner, G. (1993). "Solving Ordinary
Differential Equations I: Non-stiff Problems".
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from hiten.algorithms.dynamics.base import _DynamicalSystemProtocol
from hiten.algorithms.integrators.configs import _EventConfig
from hiten.algorithms.integrators.types import _Solution


class _Integrator(ABC):
    """Define the minimal interface that every concrete integrator must satisfy.

    Parameters
    ----------
    name : str
        Human-readable identifier of the method.
    **options
        Extra keyword arguments left untouched and stored in
        :attr:`~hiten.algorithms.integrators.base._Integrator.options` for later use by subclasses.

    Notes
    -----
    Subclasses *must* implement the abstract members :func:`~hiten.algorithms.integrators.base._Integrator.order` and
    :func:`~hiten.algorithms.integrators.base._Integrator.integrate`.

    Examples
    --------
    Creating a dummy first-order explicit Euler scheme::

        class Euler(_Integrator):
            @property
            def order(self):
                return 1

            def integrate(self, system, y0, t_vals, **kwds):
                y = [y0]
                for t0, t1 in zip(t_vals[:-1], t_vals[1:]):
                    dt = t1 - t0
                    y.append(y[-1] + dt * hiten.system.rhs(t0, y[-1]))
                return _Solution(np.asarray(t_vals), np.asarray(y))
    """
    
    def __init__(self, name: str, **options):
        self.name = name
        self.options = options
    
    @property
    @abstractmethod
    def order(self) -> Optional[int]:
        """Order of accuracy of the integrator.
        
        Returns
        -------
        int or None
            Order of the method, or None if not applicable
        """
        pass
    
    @abstractmethod
    def integrate(
        self,
        system: _DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray,
        *,
        event_fn: "Callable[[float, np.ndarray], float] | None" = None,
        event_cfg: "_EventConfig | None" = None,
        **kwargs
    ) -> _Solution:
        """Integrate the dynamical system from initial conditions.
        
        Parameters
        ----------
        system : _DynamicalSystemProtocol
            The dynamical system to integrate
        y0 : numpy.ndarray
            Initial state vector, shape (hiten.system.dim,)
        t_vals : numpy.ndarray
            Array of time points at which to evaluate the solution
        **kwargs
            Additional integration options
            
        Returns
        -------
        :class:`~hiten.algorithms.integrators.base._Solution`
            Integration results containing times and states
            
        Raises
        ------
        ValueError
            If the system is incompatible with this integrator
        """
        pass
    
    def validate_system(self, system: _DynamicalSystemProtocol) -> None:
        """Check that *system* complies with :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol`.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol`
            Candidate system whose suitability is being tested.

        Raises
        ------
        ValueError
            If the required attribute :attr:`~hiten.algorithms.integrators.base._DynamicalSystem.rhs` is absent.
        """
        if not hasattr(system, 'rhs'):
            raise ValueError(f"System must implement 'rhs' method for {self.name}")
    
    def validate_inputs(
        self,
        system: _DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray
    ) -> None:
        """Validate that the input arguments form a consistent integration task.

        Parameters
        ----------
        system : :class:`~hiten.algorithms.dynamics.base._DynamicalSystemProtocol`
            System to be integrated.
        y0 : numpy.ndarray
            Initial state vector of length :attr:`~hiten.system.dim`.
        t_vals : numpy.ndarray
            Strictly monotonic array of time nodes with at least two entries.

        Raises
        ------
        ValueError
            If any of the following conditions holds:
            - ``len(y0)`` differs from :attr:`~hiten.system.dim`.
            - ``t_vals`` contains fewer than two points.
            - ``t_vals`` is not strictly monotonic.
        """
        self.validate_system(system)
        
        if len(y0) != system.dim:
            raise ValueError(
                f"Initial state dimension {len(y0)} != system dimension {system.dim}"
            )
        
        if len(t_vals) < 2:
            raise ValueError("Must provide at least 2 time points")
        
        # Check that time values are monotonic (either strictly increasing or decreasing)
        dt = np.diff(t_vals)
        # Allow zero-span intervals (all times equal) for short-circuit handling in drivers
        if np.all(dt == 0.0):
            return
        if not (np.all(dt > 0) or np.all(dt < 0)):
            raise ValueError("Time values must be strictly monotonic (either increasing or decreasing)")

    def __str__(self):
        return f"HITEN-{self.name}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', options={self.options})"

    def _maybe_constant_solution(
        self,
        system: _DynamicalSystemProtocol,
        y0: np.ndarray,
        t_vals: np.ndarray,
    ) -> "_Solution | None":
        """Return constant-state solution when span is effectively zero; else None.

        This centralizes the zero-span short-circuit so concrete integrators
        can simply call this helper at the top of their integrate methods.
        """
        if t_vals.size >= 2 and np.isclose(t_vals[0], t_vals[-1]):
            f = system.rhs
            deriv0 = f(t_vals[0], y0)
            states = np.repeat(y0[None, :], repeats=t_vals.size, axis=0)
            derivs = np.repeat(deriv0[None, :], repeats=t_vals.size, axis=0)
            return _Solution(times=t_vals.copy(), states=states, derivatives=derivs)
        return None