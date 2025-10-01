"""Event-handling integrator benchmark.

Compares Hiten's event-capable integrator (DOP853) against SciPy solvers on
simple problems with known event times. Reports detection accuracy and speed,
and saves a comparison plot.
"""

import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, types
from scipy.integrate import solve_ivp

# Make project src importable when running from repository root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hiten.algorithms.dynamics.rhs import create_rhs_system
from hiten.algorithms.integrators import RungeKutta
from hiten.algorithms.types.configs import EventConfig

warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class EventBenchmarkResult:
    integrator_name: str
    problem_name: str
    t_event: Optional[float]
    y_event: Optional[np.ndarray]
    t_error: float
    y_error: Optional[float]
    computation_time: float
    converged: bool = True
    error_message: str = ""


# --- Precompiled global RHS and event functions (stable identities) ---
@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def g_y1_jit(t: float, y: np.ndarray) -> float:
    return y[0] - 1.0


@njit(types.float64(types.float64, types.float64[:]), cache=True, fastmath=True)
def g_x0_jit(t: float, y: np.ndarray) -> float:
    return y[0]


@njit(types.float64[:](types.float64, types.float64[:]), cache=True, fastmath=True)
def rhs_inc_jit(t: float, y: np.ndarray) -> np.ndarray:
    out = np.empty(1, dtype=np.float64)
    out[0] = 1.0
    return out


@njit(types.float64[:](types.float64, types.float64[:]), cache=True, fastmath=True)
def rhs_dec_jit(t: float, y: np.ndarray) -> np.ndarray:
    out = np.empty(1, dtype=np.float64)
    out[0] = -1.0
    return out


@njit(types.float64[:](types.float64, types.float64[:]), cache=True, fastmath=True)
def rhs_ho_jit(t: float, y: np.ndarray) -> np.ndarray:
    out = np.empty(2, dtype=np.float64)
    out[0] = y[1]
    out[1] = -y[0]
    return out


# --- Event test problems using precompiled functions ---
def _problems() -> List[Dict]:
    problems: List[Dict] = []

    problems.append({
        "name": "y' = 1, hit y=1 (inc)",
        "rhs": rhs_inc_jit,
        "y0": np.array([0.2], dtype=float),
        "t_span": (0.0, 5.0),
        "event": g_y1_jit,
        "direction": +1,
        "t_expected": lambda y0: 1.0 - float(y0[0]),
        "dim": 1,
    })

    problems.append({
        "name": "y' = -1, hit y=1 (dec)",
        "rhs": rhs_dec_jit,
        "y0": np.array([2.0], dtype=float),
        "t_span": (0.0, 5.0),
        "event": g_y1_jit,
        "direction": -1,
        "t_expected": lambda y0: float(y0[0]) - 1.0,
        "dim": 1,
    })

    problems.append({
        "name": "Harmonic oscillator, hit x=0",
        "rhs": rhs_ho_jit,
        "y0": np.array([1.0, 0.0], dtype=float),
        "t_span": (0.0, 5.0),
        "event": g_x0_jit,
        "direction": -1,
        "t_expected": lambda y0: 0.5 * np.pi,
        "dim": 2,
    })

    return problems


# --- Warm-up helpers (avoid JIT bias in timing) ---
def _make_warmup_times(t0: float) -> np.ndarray:
    return np.array([t0, t0 + 1.0e1], dtype=float)


def _is_fixed_step(integrator) -> bool:
    try:
        return hasattr(integrator, "_integrate_fixed_rk")
    except Exception:
        return False


def _warmup_hiten_event(integrator, system, y0: np.ndarray, event_fn: Callable[[float, np.ndarray], float], direction: int) -> None:
    try:
        warmup_t = _make_warmup_times(0.0)
        if _is_fixed_step(integrator):
            # Use a small grid to trigger step-based event path and refinement
            t_grid = np.linspace(warmup_t[0], warmup_t[-1], 64)
            _ = integrator.integrate(
                system,
                y0,
                t_grid,
                event_fn=event_fn,
                event_cfg=EventConfig(direction=direction, terminal=True),
            )
        else:
            _ = integrator.integrate(
                system,
                y0,
                warmup_t,
                event_fn=event_fn,
                event_cfg=EventConfig(direction=direction, terminal=True),
            )
    except Exception:
        pass


def _warmup_scipy_event(method: str, rhs: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, t0: float, event_fn: Callable[[float, np.ndarray], float], direction: int) -> None:
    try:
        def ev(t, y):
            return event_fn(t, y)
        ev.terminal = True
        ev.direction = float(direction)
        warmup_t = _make_warmup_times(t0)
        _ = solve_ivp(rhs, (warmup_t[0], warmup_t[-1]), y0, method=method, events=ev, rtol=1e-6, atol=1e-8)
    except Exception:
        pass


def run_hiten_event(
    integrator,
    system,
    y0: np.ndarray,
    t_span: Tuple[float, float],
    event_fn: Callable[[float, np.ndarray], float],
    direction: int,
    t_expected: float,
    problem_name: str,
) -> EventBenchmarkResult:
    _warmup_hiten_event(integrator, system, y0, event_fn, direction)
    start = time.perf_counter()
    try:
        # Adaptive integrators only need endpoints; fixed-step benefits from a grid
        if _is_fixed_step(integrator):
            # Use a moderately fine grid for fixed-step methods
            n_grid = 4097
            t_eval = np.linspace(t_span[0], t_span[1], n_grid, dtype=float)
        else:
            t_eval = np.array([t_span[0], t_span[1]], dtype=float)
        sol = integrator.integrate(
            system,
            y0,
            t_eval,
            event_fn=event_fn,
            event_cfg=EventConfig(direction=direction, terminal=True),
        )
        dt = time.perf_counter() - start
        # Event path returns two nodes [t0, t_hit] or [t0, tmax] when no hit
        t_event = float(sol.times[-1])
        y_event = sol.states[-1].copy()
        t_err = abs(t_event - t_expected)
        y_err = float(np.linalg.norm(y_event - y_event))  # always 0 vs itself (placeholder)
        return EventBenchmarkResult(
            integrator_name=str(integrator),
            problem_name=problem_name,
            t_event=t_event,
            y_event=y_event,
            t_error=t_err,
            y_error=y_err,
            computation_time=dt,
            converged=True,
        )
    except Exception as e:
        dt = time.perf_counter() - start
        return EventBenchmarkResult(
            integrator_name=str(integrator),
            problem_name=problem_name,
            t_event=None,
            y_event=None,
            t_error=np.inf,
            y_error=None,
            computation_time=dt,
            converged=False,
            error_message=str(e),
        )


def run_scipy_event(
    name: str,
    method: str,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    event_fn: Callable[[float, np.ndarray], float],
    direction: int,
    t_expected: float,
    problem_name: str,
) -> EventBenchmarkResult:
    _warmup_scipy_event(method, rhs, y0, t_span[0], event_fn, direction)
    def ev(t, y):
        return event_fn(t, y)
    ev.terminal = True
    ev.direction = float(direction)
    start = time.perf_counter()
    try:
        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            method=method,
            events=ev,
            rtol=1e-8,
            atol=1e-10,
        )
        dt = time.perf_counter() - start
        if sol.t_events and len(sol.t_events[0]) > 0:
            t_event = float(sol.t_events[0][0])
            y_event = sol.y_events[0][0].copy() if hasattr(sol, 'y_events') and sol.y_events and len(sol.y_events[0]) > 0 else None
            t_err = abs(t_event - t_expected)
            y_err = float(np.linalg.norm(y_event - y_event)) if y_event is not None else None
            return EventBenchmarkResult(
                integrator_name=name,
                problem_name=problem_name,
                t_event=t_event,
                y_event=y_event,
                t_error=t_err,
                y_error=y_err,
                computation_time=dt,
                converged=True,
            )
        else:
            return EventBenchmarkResult(
                integrator_name=name,
                problem_name=problem_name,
                t_event=None,
                y_event=None,
                t_error=np.inf,
                y_error=None,
                computation_time=dt,
                converged=False,
                error_message=str(sol.message),
            )
    except Exception as e:
        dt = time.perf_counter() - start
        return EventBenchmarkResult(
            integrator_name=name,
            problem_name=problem_name,
            t_event=None,
            y_event=None,
            t_error=np.inf,
            y_error=None,
            computation_time=dt,
            converged=False,
            error_message=str(e),
        )


def run_event_benchmark() -> List[EventBenchmarkResult]:
    results: List[EventBenchmarkResult] = []

    # Hiten integrators (adaptive and fixed-step)
    hiten_integrators = {
        "HITEN-RK45": RungeKutta(order=45, rtol=1e-8, atol=1e-10),
        "HITEN-DOP853": RungeKutta(order=853, rtol=1e-8, atol=1e-10),
        "HITEN-RK4-fixed": RungeKutta(order=4),
        "HITEN-RK6-fixed": RungeKutta(order=6),
        "HITEN-RK8-fixed": RungeKutta(order=8),
    }

    # SciPy integrators with event support
    scipy_integrators = {
        "SciPy-RK45": "RK45",
        "SciPy-DOP853": "DOP853",
        "SciPy-Radau": "Radau",
        "SciPy-BDF": "BDF",
        "SciPy-LSODA": "LSODA",
    }

    print("Event Integrator Benchmark (Hiten vs SciPy)")
    print("=" * 60)

    for prob in _problems():
        print(f"\nProblem: {prob['name']}")
        print("-" * 40)
        system = create_rhs_system(prob["rhs"], dim=prob["dim"], name=prob["name"])  
        t_expected = float(prob["t_expected"](prob["y0"]))

        # Hiten
        for name, integrator in hiten_integrators.items():
            res = run_hiten_event(
                integrator=integrator,
                system=system,
                y0=prob["y0"],
                t_span=prob["t_span"],
                event_fn=prob["event"],
                direction=prob["direction"],
                t_expected=t_expected,
                problem_name=prob["name"],
            )
            results.append(res)
            if res.converged:
                print(f"{str(integrator):15s}: t_hit={res.t_event:.10f}, |dt|={res.t_error:.2e}, time={res.computation_time:.4f}s")
            else:
                print(f"{str(integrator):15s}: FAILED - {res.error_message}")

        # SciPy
        for name, method in scipy_integrators.items():
            res = run_scipy_event(
                name=name,
                method=method,
                rhs=prob["rhs"],
                y0=prob["y0"],
                t_span=prob["t_span"],
                event_fn=prob["event"],
                direction=prob["direction"],
                t_expected=t_expected,
                problem_name=prob["name"],
            )
            results.append(res)
            if res.converged:
                print(f"{name:15s}: t_hit={res.t_event:.10f}, |dt|={res.t_error:.2e}, time={res.computation_time:.4f}s")
            else:
                print(f"{name:15s}: FAILED - {res.error_message}")

    return results


def print_summary_table(results: List[EventBenchmarkResult]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (Events)")
    print("=" * 100)
    print(f"{'Integrator':<15} {'Problem':<30} {'t_hit':<18} {'|dt|':<12} {'Time (s)':<10}")
    print("-" * 100)

    groups: Dict[str, List[EventBenchmarkResult]] = {}
    for r in results:
        groups.setdefault(r.problem_name, []).append(r)

    for problem_name in sorted(groups.keys()):
        for r in sorted(groups[problem_name], key=lambda x: (not x.converged, x.t_error)):
            t_hit_str = f"{r.t_event:.10f}" if r.t_event is not None else "N/A"
            dt_str = f"{r.t_error:.2e}" if np.isfinite(r.t_error) else "inf"
            print(f"{r.integrator_name:<15} {r.problem_name:<30} {t_hit_str:<18} {dt_str:<12} {r.computation_time:<10.4f}")


def create_performance_plot(results: List[EventBenchmarkResult]) -> None:
    os.makedirs('_debug/results/plots', exist_ok=True)

    # Plot |dt| vs time, grouped by integrator names
    names = list({r.integrator_name for r in results})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(names))))
    cmap: Dict[str, np.ndarray] = dict(zip(names, colors))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for name in names:
        subset = [r for r in results if r.integrator_name == name and r.converged]
        if not subset:
            continue
        times = [r.computation_time for r in subset]
        terrs = [r.t_error for r in subset]
        ax.loglog(times, terrs, 'o', label=name, color=cmap[name], markersize=8)

    ax.set_xlabel('Computation Time (s)')
    ax.set_ylabel('Absolute Event Time Error |dt|')
    ax.set_title('Event Detection: Accuracy vs Speed')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out_path = '_debug/results/plots/events_performance_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance plot saved to {out_path}")


def main() -> None:
    results = run_event_benchmark()
    print_summary_table(results)
    create_performance_plot(results)
    print("\nEvent benchmark completed.")


if __name__ == "__main__":
    main()


