"""Abstract base class for continuation stepping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class _StepProposal:
    """Prediction payload returned by continuation steppers."""

    prediction: np.ndarray
    step_hint: Optional[np.ndarray] = None


class _ContinuationStepBase(ABC):
    """Define the protocol for continuation stepping strategies."""

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
    ) -> Optional[np.ndarray]:
        """Hook executed after successful correction; may return next step."""
        return proposal.step_hint

    def on_reject(
        self,
        *,
        last_solution: object,
        step: np.ndarray,
        proposal: _StepProposal,
    ) -> Optional[np.ndarray]:
        """Hook executed after failed correction; may return adjusted step."""
        return None
