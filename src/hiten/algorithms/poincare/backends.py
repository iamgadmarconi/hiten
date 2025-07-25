from abc import ABC, abstractmethod

import numpy as np


class _ReturnMapBackend(ABC):
    """Abstract contract for any return-map computation backend."""

    @abstractmethod
    def compute_section(self, *, recompute: bool = False):
        """Compute / return the section object (backend-specific)."""

    @abstractmethod
    def compute_grid(self, *, recompute: bool = False):
        """Compute / return the grid object (backend-specific)."""

    def points2d(self) -> np.ndarray:
        sec = self.compute_section()
        return sec.points

    def states(self) -> np.ndarray:
        sec = self.compute_section()
        return getattr(sec, "states", np.empty((0, 0)))

    def __len__(self):
        return self.points2d().shape[0]


class _CenterManifoldBackend(_ReturnMapBackend):
    pass