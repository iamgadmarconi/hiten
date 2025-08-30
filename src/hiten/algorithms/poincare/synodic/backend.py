import numpy as np
from hiten.algorithms.poincare.core.backend import _ReturnMapBackend
from hiten.algorithms.poincare.core.events import _SurfaceEvent


class _NoOpBackend(_ReturnMapBackend):
    """Placeholder backend; never used by the synodic engine."""

    def __init__(self) -> None:
        # Pass inert placeholders; engine will not use backend.
        class _NullSurface(_SurfaceEvent):
            def value(self, state: np.ndarray) -> float:    
                return 0.0

        super().__init__(dynsys=None, surface=_NullSurface())

    def step_to_section(self, seeds: "np.ndarray", *, dt: float = 1e-2) -> tuple["np.ndarray", "np.ndarray"]:
        raise NotImplementedError("Synodic engine does not propagate seeds")

