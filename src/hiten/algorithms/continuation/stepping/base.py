"""Abstract base class for continuation stepping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(slots=True)
class _StepProposal:
    """Prediction payload returned by continuation steppers."""

    prediction: np.ndarray
    step_hint: Optional[np.ndarray] = None


class _ContinuationStepBase(ABC):
    """Define the protocol for continuation stepping strategies."""

    def __init__(
        self,
        *,
        step_min: float = 1e-10,
        step_max: float = 1.0,
        shrink_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self._step_min = step_min
        self._step_max = step_max
        self._shrink_policy = shrink_policy

    @abstractmethod
    def predict(self, last_solution: object, step: np.ndarray) -> _StepProposal:
        """Generate a prediction for the next solution."""

    def on_accept(
        self,
        *,
        last_solution: object,
        new_solution: np.ndarray,
        step: np.ndarray,
        proposal: _StepProposal,
    ) -> np.ndarray:
        """Hook executed after successful correction; returns next step."""
        next_step = proposal.step_hint if proposal.step_hint is not None else step
        return self._clamp_step(next_step)

    def on_reject(
        self,
        *,
        last_solution: object,
        step: np.ndarray,
        proposal: _StepProposal,
    ) -> np.ndarray:
        """Hook executed after failed correction; returns shrunk step."""
        if self._shrink_policy is not None:
            try:
                new_step = self._shrink_policy(step)
            except Exception:
                new_step = step * 0.5
        else:
            new_step = step * 0.5
        return self._clamp_step(new_step)

    def _clamp_step(self, vec: np.ndarray) -> np.ndarray:
        """Clamp step magnitude to configured bounds."""
        mag = np.clip(np.abs(vec), self._step_min, self._step_max)
        return np.sign(vec) * mag
