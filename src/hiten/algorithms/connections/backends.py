from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from hiten.algorithms.connections.config import _BallisticConfig
from hiten.algorithms.connections.results import ConnectionResult
from hiten.algorithms.connections.section.base import _SectionAdapter


class _BackendBase(ABC):
    """Shared helpers for connection backends.

    Provides geometry checks, 2D tangent/transversality, velocity mismatch, and
    result construction utilities usable by ballistic/impulsive/low-thrust backends.
    """

    def _check_geometry(self, src: _SectionAdapter, tgt: _SectionAdapter) -> None:
        if not src.geometry_matches(tgt):
            raise ValueError("Source and target sections use different geometry/projection")

    @staticmethod
    def _points(adapter: _SectionAdapter) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
        return adapter.points2d(), adapter.states6d(), adapter.labels()

    @staticmethod
    def _tangent2d(points: np.ndarray, idx: int) -> np.ndarray | None:
        n = points.shape[0]
        if n < 2:
            return None
        i0 = max(0, idx - 1)
        i1 = min(n - 1, idx + 1)
        v = points[i1] - points[i0]
        norm = float(np.hypot(v[0], v[1]))
        if norm == 0.0:
            return None
        return v / norm

    def _transversality_angle(self, ps: np.ndarray, i: int, pt: np.ndarray, j: int) -> float | None:
        vs = self._tangent2d(ps, i)
        vt = self._tangent2d(pt, j)
        if vs is None or vt is None:
            return None
        dot = float(abs(vs[0] * vt[0] + vs[1] * vt[1]))
        dot = max(0.0, min(1.0, dot))
        try:
            import math
            return math.acos(dot)
        except Exception:
            return None

    @staticmethod
    def _dv_mismatch(src_states: np.ndarray, i: int, tgt_states: np.ndarray, j: int) -> tuple[np.ndarray, float]:
        v_src = src_states[i, 3:6]
        v_tgt = tgt_states[j, 3:6]
        dv = v_tgt - v_src
        return dv, float(np.linalg.norm(dv))

    @staticmethod
    def _build_result(
        *,
        ballistic: bool,
        labels: tuple[str, str],
        match_point2d: tuple[float, float] | None,
        angle: float | None,
        dv_list: List[float] | None = None,
        tof: float | None = None,
    ) -> ConnectionResult:
        return ConnectionResult(
            ballistic=ballistic,
            dv_list=[] if dv_list is None else dv_list,
            total_dv=0.0 if not dv_list else float(sum(dv_list)),
            tof=None if tof is None else float(tof),
            match_point2d=None if match_point2d is None else (float(match_point2d[0]), float(match_point2d[1])),
            transversality_angle=angle,
            source_leg=None,
            target_leg=None,
            section_labels=labels,
        )

    @abstractmethod
    def pre_refine(self, *args, **kwargs) -> List[ConnectionResult]:
        ...


@dataclass
class BallisticBackend(_BackendBase):
    cfg: _BallisticConfig

    def pre_refine(
        self,
        src: _SectionAdapter,
        tgt: _SectionAdapter,
        candidates: List[Tuple[int, int]],
    ) -> List[ConnectionResult]:
        """Minimal 2D match -> proto ConnectionResult (no shooting).

        Use the source section point as the meet if within tolerance; otherwise discard.
        """
        out: List[ConnectionResult] = []
        ps, xs, labels = self._points(src)
        pt, xt, _ = self._points(tgt)

        tol2 = float(self.cfg.tol_intersection ** 2)
        for i, j in candidates:
            diff = ps[i] - pt[j]
            d2 = float(diff[0]*diff[0] + diff[1]*diff[1])
            if d2 > tol2:
                continue
            p = ps[i]
            angle = self._transversality_angle(ps, i, pt, j)
            res = self._build_result(
                ballistic=True,
                labels=labels,
                match_point2d=(float(p[0]), float(p[1])),
                angle=angle,
                dv_list=[],
            )
            out.append(res)
        return out


