#!/usr/bin/env python3
"""Symplectic event-handling benchmark and example.

This script mirrors ``event_integrator_benchmark.py`` but focuses on
Hiten's extended symplectic integrators with event detection enabled.
It constructs a truncated pendulum Hamiltonian, integrates it with
event-enabled symplectic schemes, compares the detected event times
against SciPy solvers, and saves an accuracy vs speed plot.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList
from scipy.integrate import solve_ivp

# Make project src importable when running from repository root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hiten.algorithms.dynamics.hamiltonian import create_hamiltonian_system
from hiten.algorithms.integrators import ExtendedSymplectic
from hiten.algorithms.integrators.symplectic import (N_SYMPLECTIC_DOF,
                                                     N_VARS_POLY,
                                                     P_POLY_INDICES,
                                                     Q_POLY_INDICES)
from hiten.algorithms.polynomial.base import (_create_encode_dict_from_clmo,
                                              _encode_multiindex,
                                              _init_index_tables)
from hiten.algorithms.types.configs import EventConfig
from hiten.algorithms.types.options import EventOptions

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EventResult:
    solver_name: str
    problem_name: str
    t_event: Optional[float]
    y_event: Optional[np.ndarray]
    t_error: float
    y_error: Optional[float]
    computation_time: float
    converged: bool = True
    error_message: str = ""


@dataclass
class SymplecticEventProblem:
    name: str
    description: str
    y0_extended: np.ndarray
    y0_canonical: np.ndarray
    t_span: Tuple[float, float]
    grid_size: int
    direction: int


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def event_q1_jit(t: float, y: np.ndarray) -> float:
    """Event function detecting q1 = 0 crossings."""
    return y[0]


def truncated_pendulum_rhs(_t: float, y: np.ndarray) -> List[float]:
    """Return dynamics for the 2D truncated pendulum system."""
    q, p = y
    sin_taylor = q - (q ** 3) / 6.0 + (q ** 5) / 120.0
    return [p, -sin_taylor]


def embed_state(qp_state: np.ndarray) -> np.ndarray:
    """Embed a 2D (q, p) state into the 6D symplectic phase space."""
    state = np.zeros(2 * N_SYMPLECTIC_DOF, dtype=np.float64)
    state[0] = qp_state[0]
    state[N_SYMPLECTIC_DOF] = qp_state[1]
    return state


def build_truncated_pendulum_system(max_degree: int = 6):
    """Construct a polynomial Hamiltonian system for a truncated pendulum."""
    psi_tables, clmo_tables = _init_index_tables(max_degree)
    encode_dict_list = _create_encode_dict_from_clmo(clmo_tables)

    H_blocks = [
        np.zeros(psi_tables[N_VARS_POLY, deg], dtype=np.complex128)
        for deg in range(max_degree + 1)
    ]

    idx_p1 = P_POLY_INDICES[0]
    idx_q1 = Q_POLY_INDICES[0]
    mono = np.zeros(N_VARS_POLY, dtype=np.int64)

    # Kinetic term: p1^2 / 2 (degree 2)
    mono[:] = 0
    mono[idx_p1] = 2
    encoded = _encode_multiindex(mono, 2, encode_dict_list)
    if encoded != -1:
        H_blocks[2][encoded] = 0.5

    # Potential offset: -1 (degree 0)
    mono[:] = 0
    encoded = _encode_multiindex(mono, 0, encode_dict_list)
    if encoded != -1:
        H_blocks[0][encoded] = -1.0

    # Quadratic potential: +q1^2 / 2 (degree 2)
    mono[:] = 0
    mono[idx_q1] = 2
    encoded = _encode_multiindex(mono, 2, encode_dict_list)
    if encoded != -1:
        H_blocks[2][encoded] += 0.5

    # Quartic correction: -q1^4 / 24 (degree 4)
    if max_degree >= 4:
        mono[:] = 0
        mono[idx_q1] = 4
        encoded = _encode_multiindex(mono, 4, encode_dict_list)
        if encoded != -1:
            H_blocks[4][encoded] = -1.0 / 24.0

    # Sextic correction: +q1^6 / 720 (degree 6)
    if max_degree >= 6:
        mono[:] = 0
        mono[idx_q1] = 6
        encoded = _encode_multiindex(mono, 6, encode_dict_list)
        if encoded != -1:
            H_blocks[6][encoded] = 1.0 / 720.0

    H_blocks_typed = NumbaList()
    for arr in H_blocks:
        H_blocks_typed.append(arr.copy())

    system = create_hamiltonian_system(
        H_blocks=H_blocks_typed,
        degree=max_degree,
        psi_table=psi_tables,
        clmo_table=clmo_tables,
        encode_dict_list=encode_dict_list,
        n_dof=N_SYMPLECTIC_DOF,
        name="Truncated Pendulum Hamiltonian",
    )

    return system


def make_event_problems() -> List[SymplecticEventProblem]:
    """Configure example problems for event detection."""
    angle = np.deg2rad(45.0)
    base_state = np.zeros(2 * N_SYMPLECTIC_DOF, dtype=np.float64)
    base_state_neg = base_state.copy()

    # Positive release, expect crossing with negative direction
    base_state[0] = angle
    problem_pos = SymplecticEventProblem(
        name="Pendulum release (+45 deg)",
        description="Pendulum released from +45 degrees, expect q1 -> 0 with negative crossing.",
        y0_extended=base_state.copy(),
        y0_canonical=np.array([angle, 0.0], dtype=np.float64),
        t_span=(0.0, 5.0),
        grid_size=4097,
        direction=-1,
    )

    # Negative release, expect crossing with positive direction
    base_state_neg[0] = -angle
    problem_neg = SymplecticEventProblem(
        name="Pendulum release (-45 deg)",
        description="Pendulum released from -45 degrees, expect q1 -> 0 with positive crossing.",
        y0_extended=base_state_neg.copy(),
        y0_canonical=np.array([-angle, 0.0], dtype=np.float64),
        t_span=(0.0, 5.0),
        grid_size=4097,
        direction=+1,
    )

    return [problem_pos, problem_neg]


def compute_reference_event(problem: SymplecticEventProblem) -> Tuple[float, np.ndarray]:
    """High-accuracy reference using SciPy Radau."""
    event = lambda t, y: y[0]
    event.terminal = True
    event.direction = float(problem.direction)

    sol = solve_ivp(
        truncated_pendulum_rhs,
        problem.t_span,
        problem.y0_canonical,
        method="Radau",
        events=event,
        rtol=1e-12,
        atol=1e-14,
    )

    if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
        t_event = float(sol.t_events[0][0])
        y_event = sol.y_events[0][0].copy()
        return t_event, y_event

    raise RuntimeError(f"Reference solver failed to detect event for problem '{problem.name}'")


def _warmup_symplectic_event(integrator, system, problem: SymplecticEventProblem) -> None:
    """Trigger compilation overhead outside timed region."""
    try:
        warmup_grid = np.linspace(
            problem.t_span[0],
            problem.t_span[0] + 1.0e-2,
            8,
            dtype=np.float64,
        )
        integrator.integrate(
            system,
            problem.y0_extended,
            warmup_grid,
            event_fn=event_q1_jit,
            event_cfg=EventConfig(direction=problem.direction, terminal=True),
            event_options=EventOptions(xtol=1.0e-12, gtol=1.0e-12),
        )
    except Exception:
        pass


def _warmup_scipy_event(method: str, problem: SymplecticEventProblem) -> None:
    """Trigger SciPy solver overhead outside timed region."""
    try:
        event = lambda t, y: y[0]
        event.terminal = True
        event.direction = float(problem.direction)
        warmup_t_span = (problem.t_span[0], problem.t_span[0] + 1.0e-2)
        solve_ivp(
            truncated_pendulum_rhs,
            warmup_t_span,
            problem.y0_canonical,
            method=method,
            events=event,
            rtol=1e-6,
            atol=1e-8,
        )
    except Exception:
        pass


def run_symplectic_event(
    solver_name: str,
    integrator,
    system,
    problem: SymplecticEventProblem,
    t_expected: float,
    y_expected: np.ndarray,
) -> EventResult:
    """Execute symplectic integration with event detection."""
    time_grid = np.linspace(
        problem.t_span[0], problem.t_span[1], problem.grid_size, dtype=np.float64
    )
    event_cfg = EventConfig(direction=problem.direction, terminal=True)

    event_options = EventOptions(xtol=1.0e-12, gtol=1.0e-12)

    _warmup_symplectic_event(integrator, system, problem)

    start = time.perf_counter()
    try:
        sol = integrator.integrate(
            system,
            problem.y0_extended,
            time_grid,
            event_fn=event_q1_jit,
            event_cfg=event_cfg,
        )
        elapsed = time.perf_counter() - start

        t_event = float(sol.times[-1])
        y_event = sol.states[-1].copy()
        t_error = abs(t_event - t_expected)
        y_error = float(np.linalg.norm(y_event - y_expected)) if y_expected is not None else None

        return EventResult(
            solver_name=solver_name,
            problem_name=problem.name,
            t_event=t_event,
            y_event=y_event,
            t_error=t_error,
            y_error=y_error,
            computation_time=elapsed,
            converged=True,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return EventResult(
            solver_name=solver_name,
            problem_name=problem.name,
            t_event=None,
            y_event=None,
            t_error=np.inf,
            y_error=None,
            computation_time=elapsed,
            converged=False,
            error_message=str(exc),
        )


def run_scipy_event(
    solver_name: str,
    method: str,
    solver_opts: Dict,
    problem: SymplecticEventProblem,
    t_expected: float,
    y_expected: np.ndarray,
) -> EventResult:
    """Run SciPy solve_ivp with event detection on the 2D system."""
    event = lambda t, y: y[0]
    event.terminal = True
    event.direction = float(problem.direction)

    _warmup_scipy_event(method, problem)

    start = time.perf_counter()
    try:
        sol = solve_ivp(
            truncated_pendulum_rhs,
            problem.t_span,
            problem.y0_canonical,
            method=method,
            events=event,
            **solver_opts,
        )
        elapsed = time.perf_counter() - start

        if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
            t_event = float(sol.t_events[0][0])
            qp_event = sol.y_events[0][0].copy()
            y_event = embed_state(qp_event)
            t_error = abs(t_event - t_expected)
            y_error = float(np.linalg.norm(y_event - y_expected)) if y_expected is not None else None
            return EventResult(
                solver_name=solver_name,
                problem_name=problem.name,
                t_event=t_event,
                y_event=y_event,
                t_error=t_error,
                y_error=y_error,
                computation_time=elapsed,
                converged=True,
            )

        message = sol.message if hasattr(sol, 'message') else "event not detected"
        return EventResult(
            solver_name=solver_name,
            problem_name=problem.name,
            t_event=None,
            y_event=None,
            t_error=np.inf,
            y_error=None,
            computation_time=elapsed,
            converged=False,
            error_message=message,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return EventResult(
            solver_name=solver_name,
            problem_name=problem.name,
            t_event=None,
            y_event=None,
            t_error=np.inf,
            y_error=None,
            computation_time=elapsed,
            converged=False,
            error_message=str(exc),
        )


def run_benchmark() -> List[EventResult]:
    """Execute the symplectic event benchmark."""
    system = build_truncated_pendulum_system()
    problems = make_event_problems()

    symplectic_schemes = {
        "Symplectic4": ExtendedSymplectic(order=4, c_omega_heuristic=15.0),
        "Symplectic6": ExtendedSymplectic(order=6, c_omega_heuristic=20.0),
        "Symplectic8": ExtendedSymplectic(order=8, c_omega_heuristic=25.0),
    }

    scipy_solvers = {
        "SciPy-RK45": ("RK45", {"rtol": 1.0e-8, "atol": 1.0e-10}),
        "SciPy-DOP853": ("DOP853", {"rtol": 1.0e-9, "atol": 1.0e-11}),
        "SciPy-BDF": ("BDF", {"rtol": 1.0e-8, "atol": 1.0e-10}),
    }

    results: List[EventResult] = []

    print("Symplectic Event Integrator Benchmark")
    print("=" * 70)

    for problem in problems:
        print(f"\nProblem: {problem.name}")
        print(problem.description)
        print("-" * 70)

        t_ref, y_ref_qp = compute_reference_event(problem)
        y_ref_ext = embed_state(y_ref_qp)

        print(f"Reference event time (Radau): {t_ref:.10f} s")

        reference_result = EventResult(
            solver_name="SciPy-Radau (reference)",
            problem_name=problem.name,
            t_event=t_ref,
            y_event=y_ref_ext,
            t_error=0.0,
            y_error=0.0,
            computation_time=0.0,
            converged=True,
        )
        results.append(reference_result)

        for solver_name, integrator in symplectic_schemes.items():
            res = run_symplectic_event(
                solver_name,
                integrator,
                system,
                problem,
                t_ref,
                y_ref_ext,
            )
            results.append(res)
            if res.converged:
                print(
                    f"{solver_name:20s}: t_hit={res.t_event:.10f}, |dt|={res.t_error:.2e}, "
                    f"time={res.computation_time:.4f}s"
                )
            else:
                print(f"{solver_name:20s}: FAILED - {res.error_message}")

        for solver_name, (method, opts) in scipy_solvers.items():
            res = run_scipy_event(
                solver_name,
                method,
                opts,
                problem,
                t_ref,
                y_ref_ext,
            )
            results.append(res)
            if res.converged:
                print(
                    f"{solver_name:20s}: t_hit={res.t_event:.10f}, |dt|={res.t_error:.2e}, "
                    f"time={res.computation_time:.4f}s"
                )
            else:
                print(f"{solver_name:20s}: FAILED - {res.error_message}")

    return results


def print_summary_table(results: List[EventResult]) -> None:
    """Print formatted summary of all runs."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE (Symplectic Events)")
    print("=" * 110)
    print(f"{'Integrator':<22} {'Problem':<32} {'t_hit':<18} {'|dt|':<12} {'|dy|':<12} {'Time (s)':<10}")
    print("-" * 110)

    grouped: Dict[str, List[EventResult]] = {}
    for res in results:
        grouped.setdefault(res.problem_name, []).append(res)

    for problem_name in sorted(grouped):
        for res in sorted(grouped[problem_name], key=lambda r: (not r.converged, r.t_error)):
            if res.converged:
                t_hit_str = f"{res.t_event:.10f}" if res.t_event is not None else "N/A"
                dt_str = f"{res.t_error:.2e}" if np.isfinite(res.t_error) else "inf"
                dy_str = (
                    f"{res.y_error:.2e}"
                    if res.y_error is not None and np.isfinite(res.y_error)
                    else "n/a"
                )
                print(
                    f"{res.solver_name:<22} {res.problem_name:<32} {t_hit_str:<18} "
                    f"{dt_str:<12} {dy_str:<12} {res.computation_time:<10.4f}"
                )
            else:
                print(
                    f"{res.solver_name:<22} {res.problem_name:<32} {'FAILED':<18} "
                    f"{'inf':<12} {'n/a':<12} {res.computation_time:<10.4f}"
                )


def create_performance_plot(results: List[EventResult]) -> None:
    """Save log-log plot of |dt| vs computation time."""
    os.makedirs(os.path.join('_debug', 'results', 'plots'), exist_ok=True)

    names = sorted({res.solver_name for res in results if res.converged and res.t_error >= 0.0})
    if not names:
        print("No successful runs to plot.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(names))))
    cmap = dict(zip(names, colors))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name in names:
        subset = [r for r in results if r.solver_name == name and r.converged and np.isfinite(r.t_error)]
        if not subset:
            continue
        times = [r.computation_time for r in subset]
        terrs = [r.t_error for r in subset]
        ax.loglog(times, terrs, 'o', label=name, color=cmap[name], markersize=8)

    ax.set_xlabel('Computation Time (s)')
    ax.set_ylabel('Absolute Event Time Error |dt|')
    ax.set_title('Symplectic Event Detection: Accuracy vs Speed')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join('_debug', 'results', 'plots', 'symplectic_events_performance.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to {out_path}")


def main() -> None:
    results = run_benchmark()
    print_summary_table(results)
    create_performance_plot(results)
    print("\nSymplectic event benchmark completed.")


if __name__ == "__main__":
    main()

