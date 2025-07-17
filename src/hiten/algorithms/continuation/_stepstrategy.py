from typing import Protocol

import numpy as np


class StepStrategy(Protocol):

    def __call__(
        self,
        last_solution: object,
        step: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def on_accept(self, *args, **kwargs) -> None: ...

    def on_iteration(self, *args, **kwargs) -> None: ...

    def on_reject(self, *args, **kwargs) -> None: ...

    def on_failure(self, *args, **kwargs) -> None: ...

    def on_success(self, *args, **kwargs) -> None: ...

    def on_initialisation(self, *args, **kwargs) -> None: ...
