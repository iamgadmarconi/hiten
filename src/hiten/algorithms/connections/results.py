from dataclasses import dataclass
from typing import Iterator, Literal, Sequence, Tuple

import numpy as np


@dataclass
class _ConnectionResult:
    kind: Literal["impulsive", "ballistic"]
    delta_v: float
    point2d: Tuple[float, float]
    state_u: np.ndarray
    state_s: np.ndarray
    index_u: int
    index_s: int


class ConnectionResults:
    """Lightweight view over a sequence of `_ConnectionResult` with nice printing.

    Behaves like a read-only sequence and renders a compact table on str().
    """

    def __init__(self, results: Sequence[_ConnectionResult] | None):
        self._results: list[_ConnectionResult] = list(results) if results else []

    # Sequence-like methods
    def __len__(self) -> int: 
        return len(self._results)

    def __iter__(self) -> Iterator[_ConnectionResult]: 
        return iter(self._results)

    def __getitem__(self, idx: int) -> _ConnectionResult: 
        return self._results[idx]

    def __bool__(self) -> bool: 
        return bool(self._results)

    # Pretty printing
    def __repr__(self) -> str:
        n_total = len(self._results)
        n_ballistic = sum(1 for r in self._results if r.kind == "ballistic")
        n_impulsive = n_total - n_ballistic
        return (
            f"ConnectionResults(n={n_total}, ballistic={n_ballistic}, impulsive={n_impulsive})"
        )

    def __str__(self) -> str:
        if not self._results:
            return "<no connection results>"

        # Header
        headers = ("#", "kind", "ΔV", "idx_u", "idx_s", "state_u", "state_s")
        rows: list[tuple[str, ...]] = [headers]

        # Build rows with limited precision for readability
        for i, r in enumerate(self._results):
            su = np.asarray(r.state_u).ravel()
            ss = np.asarray(r.state_s).ravel()
            su_str = "[" + ", ".join(f"{v:.6f}" for v in su) + "]"
            ss_str = "[" + ", ".join(f"{v:.6f}" for v in ss) + "]"
            rows.append(
                (
                    str(i),
                    r.kind,
                    f"{r.delta_v:.3e}",
                    str(r.index_u),
                    str(r.index_s),
                    su_str,
                    ss_str,
                )
            )

        # Compute column widths
        col_widths = [max(len(row[c]) for row in rows) for c in range(len(headers))]

        def fmt_row(row: tuple[str, ...]) -> str:
            return "  ".join(cell.rjust(col_widths[i]) for i, cell in enumerate(row))

        # Assemble table
        lines = [fmt_row(rows[0])]
        lines.append("  ".join("-" * w for w in col_widths))
        for row in rows[1:]:
            lines.append(fmt_row(row))

        return "\n".join(lines)
