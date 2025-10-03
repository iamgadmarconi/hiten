"""Abstract base class for continuation backends."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import numpy as np

from hiten.algorithms.continuation.stepping.support import (
    _ContinuationStepSupport, _NullStepSupport)
from hiten.algorithms.continuation.stepping import (
    _ContinuationStepperFactory, make_natural_stepper)
from hiten.algorithms.continuation.types import ContinuationBackendRequest, ContinuationBackendResponse
from hiten.algorithms.types.core import _HitenBaseBackend


class _ContinuationBackend(_HitenBaseBackend):
    """Base contract for continuation backends."""

    def __init__(
        self,
        *,
        stepper_factory: _ContinuationStepperFactory | None = None,
        support_factory: Callable[[], _ContinuationStepSupport] | None = None,
    ) -> None:
        self._stepper_factory = stepper_factory or make_natural_stepper()
        self._support_factory = support_factory or _NullStepSupport

    @abstractmethod
    def run(
        self,
        *,
        request: ContinuationBackendRequest,
    ) -> ContinuationBackendResponse:
        """Run continuation using structured request/response types."""

    def make_step_support(self) -> _ContinuationStepSupport:
        return self._support_factory()
