"""Provide interface classes for domain-specific continuation algorithms.

This module provides interface classes that adapt the generic continuation
engine to specific problem domains in dynamical systems. These interfaces
implement the abstract methods required by the continuation framework for
particular types of solutions (periodic orbits, invariant tori, etc.).

The interfaces serve as mix-ins that provide domain-specific implementations
of instantiation, correction, and parameter extraction methods, allowing
the generic continuation algorithm to work with different solution types.

All coordinates are in nondimensional CR3BP rotating-frame units.

See Also
--------
:mod:`~hiten.algorithms.continuation.engine`
    Continuation engines that these interfaces work with.
:mod:`~hiten.system.orbits`
    Periodic orbit classes used by orbit continuation.
:mod:`~hiten.algorithms.corrector`
    Correction algorithms used by continuation interfaces.
"""

from typing import Callable, Sequence

import numpy as np

from hiten.algorithms.utils.types import SynodicState


class _PeriodicOrbitContinuationInterface:
    """Stateless adapter that builds closures for continuation engines."""

    @staticmethod
    def representation(orbit) -> np.ndarray:
        return np.asarray(orbit.initial_state, dtype=float).copy()

    @staticmethod
    def build_instantiator(seed) -> Callable[[np.ndarray], object]:
        orbit_cls = type(seed)
        libration_point = getattr(seed, "libration_point", None)

        def _instantiate(representation: np.ndarray):
            # Create a new orbit of the same class and libration point
            return orbit_cls(libration_point=libration_point, initial_state=np.asarray(representation, dtype=float))

        return _instantiate

    @staticmethod
    def build_parameter_getter(seed, cfg) -> Callable[[np.ndarray], np.ndarray]:
        # Select continuation parameters from representation, default identity
        state = getattr(cfg, "state", None)
        if state is None:
            def _getter(repr_vec: np.ndarray) -> np.ndarray:
                return np.asarray(repr_vec, dtype=float)
            return _getter

        # Normalize state to list of indices
        if isinstance(state, SynodicState):
            indices = [int(state.value)]
        elif isinstance(state, Sequence):
            indices = [int(s.value) if isinstance(s, SynodicState) else int(s) for s in state]
        else:
            indices = [int(state)]

        idx_arr = np.asarray(indices, dtype=int)

        def _getter(repr_vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(repr_vec, dtype=float)
            return vec[idx_arr]

        return _getter

    @staticmethod
    def build_predictor(seed, cfg) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        # Default predictor adds step to selected indices; if no state provided, add to all components
        state = getattr(cfg, "state", None)
        if state is None:
            def _predictor(last: np.ndarray, step: np.ndarray) -> np.ndarray:
                return np.asarray(last, dtype=float) + np.asarray(step, dtype=float)
            return _predictor

        # Normalize state to list of indices
        if isinstance(state, SynodicState):
            indices = [int(state.value)]
        elif isinstance(state, Sequence):
            indices = [int(s.value) if isinstance(s, SynodicState) else int(s) for s in state]
        else:
            indices = [int(state)]

        idx_arr = np.asarray(indices, dtype=int)

        def _predictor(last: np.ndarray, step: np.ndarray) -> np.ndarray:
            last = np.asarray(last, dtype=float).copy()
            step = np.asarray(step, dtype=float)
            for idx, d in zip(idx_arr, step):
                last[idx] += d
            return last

        return _predictor
