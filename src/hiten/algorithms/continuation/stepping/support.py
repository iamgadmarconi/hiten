"""Support utilities for continuation steppers and backends.

This module defines lightweight protocols that allow backends to expose
optional capabilities (such as secant tangents) to continuation stepping
strategies without hard-coding engine or interface interactions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class _ContinuationStepSupport(Protocol):
    """Base protocol for backend-provided stepper support information.

    Backends may implement this protocol to receive notifications about the
    continuation loop lifecycle. Steppers can leverage these hooks to keep
    auxiliary state (e.g., residual histories, adaptive metrics) in sync with
    backend progress.
    """

    def on_accept(self, prev_repr: np.ndarray, curr_repr: np.ndarray) -> None:
        """Notify support object that a continuation step was accepted."""

    def on_reject(self, prev_repr: np.ndarray, step: np.ndarray) -> None:
        """Notify support object that a continuation step was rejected."""


@runtime_checkable
class _SecantSupport(_ContinuationStepSupport, Protocol):
    """Capability mixin for steppers that require secant tangent access."""

    def get_tangent(self) -> np.ndarray | None:
        """Return the current unit tangent in representation space, if any."""

    def seed(self, tangent: np.ndarray | None) -> None:
        """Seed the support object with an initial tangent vector."""


class _NullStepSupport:
    """Convenience no-op implementation of :class:`_ContinuationStepSupport`."""

    def on_accept(self, prev_repr: np.ndarray, curr_repr: np.ndarray) -> None:
        return None

    def on_reject(self, prev_repr: np.ndarray, step: np.ndarray) -> None:
        return None


class _VectorSpaceSecantSupport(_SecantSupport):
    """Default secant helper tracking tangents in vector space."""

    def __init__(self) -> None:
        self._tangent: np.ndarray | None = None

    def get_tangent(self) -> np.ndarray | None:
        if self._tangent is None:
            return None
        return np.asarray(self._tangent, dtype=float).copy()

    def seed(self, tangent: np.ndarray | None) -> None:
        self._tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def on_accept(self, prev_repr: np.ndarray, curr_repr: np.ndarray) -> None:
        diff = np.asarray(curr_repr, dtype=float) - np.asarray(prev_repr, dtype=float)
        flat = diff.ravel()
        norm = float(np.linalg.norm(flat))
        self._tangent = None if norm == 0.0 else flat / norm

    def on_reject(self, prev_repr: np.ndarray, step: np.ndarray) -> None:
        return None
