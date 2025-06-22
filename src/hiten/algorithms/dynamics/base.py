r"""
hiten.algorithms.dynamics.base
========================

Core abstractions for time-continuous dynamical systems used by the
integrator framework. The definitions collected here allow numerical
integrators to interact with a system through a minimal, yet explicit,
interface that is independent of the underlying physical model.

References
----------
Hairer, E.; Nørsett, S.; Wanner, G. (1993). "Solving Ordinary Differential
Equations I". Springer.

Koon, W. S.; Lo, M. W.; Marsden, J. E.; Ross, S. D. (2011). "Dynamical
Systems, the Three-Body Problem and Space Mission Design".
"""

from abc import ABC, abstractmethod
from typing import Callable, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class DynamicalSystemProtocol(Protocol):
    r"""
    Lightweight structural type understood by the integrator layer.

    The protocol declares the minimal read-only attributes that a concrete
    dynamical system must expose.

    Attributes
    ----------
    dim : int
        Dimension of the state vector.
    rhs : Callable[[float, numpy.ndarray], numpy.ndarray]
        Vector field :math:`f(t,\,\mathbf y)` returning
        :math:`\dot{\mathbf y}` evaluated at time *t* and state
        :math:`\mathbf y`.
    """
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        ...
    
    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        ...
            

class _DynamicalSystem(ABC):
    r"""
    Abstract base class implementing common functionality for concrete
    dynamical systems.

    Parameters
    ----------
    dim : int
        Dimension :math:`n \ge 1` of the state space.

    Attributes
    ----------
    dim : int
        Same as the *dim* parameter.

    Raises
    ------
    ValueError
        If *dim* is not positive.

    Notes
    -----
    Subclasses must override :pyattr:`rhs` with a callable complying with
    :pyclass:`DynamicalSystemProtocol`.
    """
    
    def __init__(self, dim: int):
        """
        Initialize the dynamical hiten.system.
        
        Parameters
        ----------
        dim : int
            Dimension of the state space
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim
    
    @property
    def dim(self) -> int:
        """Dimension of the state space."""
        return self._dim
    
    @property
    @abstractmethod
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:
        pass
    
    def validate_state(self, y: np.ndarray) -> None:
        """Check that *y* has the correct dimension.

        Parameters
        ----------
        y : numpy.ndarray
            State vector.

        Raises
        ------
        ValueError
            If the size of *y* differs from :pyattr:`dim`.
        """
        if len(y) != self.dim:
            raise ValueError(f"State vector dimension {len(y)} != system dimension {self.dim}")


class _DirectedSystem(_DynamicalSystem):
    r"""
    Directional wrapper around another dynamical hiten.system.

    The wrapper permits integration forward or backward in time and can
    selectively negate derivatives of specified state components. This is
    particularly handy for Hamiltonian systems written in
    :math:`\mathbf{q},\,\mathbf{p}` form where momentum variables change sign
    under time reversal.

    Parameters
    ----------
    base_or_dim : _DynamicalSystem | int
        A concrete system instance to be wrapped or, alternatively, the state
        dimension expected from a subclass that will implement
        :pyattr:`rhs`.
    fwd : int, default 1
        Direction flag. A positive value keeps the original direction while a
        negative value integrates backward in time.
    flip_indices : slice | Sequence[int] | None, optional
        Indices of components whose derivatives must be negated when *fwd* is
        negative. If *None*, all components are flipped.

    Attributes
    ----------
    dim : int
        Dimension of the underlying hiten.system.
    _fwd : int
        Normalised copy of *fwd* that is either +1 or -1.

    Raises
    ------
    AttributeError
        If :pyattr:`rhs` is accessed while no base system was supplied and the
        subclass did not implement its own :pyattr:`rhs`.

    Notes
    -----
    The wrapper leaves the original vector field untouched and only
    post-processes its output.
    """

    def __init__(self,
                 base_or_dim: "_DynamicalSystem | int",
                 fwd: int = 1,
                 flip_indices: "slice | Sequence[int] | None" = None):

        if isinstance(base_or_dim, _DynamicalSystem):
            self._base: "_DynamicalSystem | None" = base_or_dim
            dim = base_or_dim.dim
        else:
            self._base = None
            dim = int(base_or_dim)

        super().__init__(dim=dim)

        self._fwd: int = 1 if fwd >= 0 else -1
        self._flip_idx = flip_indices

    @property
    def rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:

        if self._base is None:
            raise AttributeError("`rhs` not implemented: subclass must provide "
                                 "its own implementation when no base system "
                                 "is wrapped.")

        base_rhs = self._base.rhs
        flip_idx = self._flip_idx

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = base_rhs(t, y)

            if self._fwd == -1:
                if flip_idx is None:
                    dy = -dy
                else:
                    dy = dy.copy()
                    dy[flip_idx] *= -1
            return dy

        return _rhs

    def __repr__(self):
        return (f"DirectedSystem(dim={self.dim}, fwd={self._fwd}, "
                f"flip_idx={self._flip_idx})")

    def __getattr__(self, item):
        if self._base is None:
            raise AttributeError(item)
        return getattr(self._base, item)