from dataclasses import dataclass
from typing import Tuple

import numpy as np

from hiten.algorithms.poincare.core.base import _Section


@dataclass
class _SectionAdapter:
    """Uniform accessor over a poincaré _Section.

    Provides 2-D points, 6-D states, labels, and geometry metadata (when available).
    """

    section: _Section
    plane_coords: Tuple[str, str]
    normal: np.ndarray | None = None  # affine-plane normal (6-D) if known
    offset: float | None = None       # affine-plane offset if known
    axis: str | int | None = None     # axis name/index if plane is axis-aligned

    def points2d(self) -> np.ndarray:
        return np.asarray(self.section.points, dtype=float)

    def states6d(self) -> np.ndarray:
        return np.asarray(self.section.states, dtype=float)

    def labels(self) -> Tuple[str, str]:
        return tuple(self.section.labels)

    def times1d(self) -> np.ndarray | None:
        ts = getattr(self.section, "times", None)
        if ts is None:
            return None
        return np.asarray(ts, dtype=float)

    def geometry_matches(self, other: "_SectionAdapter", tol: float = 1e-12) -> bool:
        # Projection labels must match
        if tuple(self.labels()) != tuple(other.labels()):
            return False

        # If both carry normals, compare normal vectors and offsets
        if self.normal is not None and other.normal is not None:
            a = np.asarray(self.normal, dtype=float).ravel()
            b = np.asarray(other.normal, dtype=float).ravel()
            if a.shape != b.shape:
                return False
            if not np.allclose(a, b, atol=tol, rtol=0.0):
                return False
            if (self.offset is not None) and (other.offset is not None):
                if not np.isclose(float(self.offset), float(other.offset), atol=tol, rtol=0.0):
                    return False
            return True

        # Else, if both expose axis and offsets, compare those
        if (self.axis is not None) and (other.axis is not None):
            if str(self.axis) != str(other.axis):
                return False
            if (self.offset is not None) and (other.offset is not None):
                if not np.isclose(float(self.offset), float(other.offset), atol=tol, rtol=0.0):
                    return False
            return True

        # Fallback: only labels matched
        return True


